"""YAMNet audio event classifier (Tier 2 of the detection pipeline).

YAMNet scores the 521 AudioSet classes independently (multi-label), so groups
are combined with max(), never summed.  Class indices come from the label list
embedded in the model file, so the mapping cannot drift from the model.

Categories returned for a window:
  human_speech      someone is talking, whispering or shouting (stranger evidence)
  crowd             many voices at once (babble); not usable as stranger evidence
  vocal_non_speech  laughing, crying, coughing, breathing... (no reliable voiceprint)
  reject            music, singing, TV/radio, synthetic speech
  uncertain         nothing above confidently; `confidence` is the speech score
"""

import logging
import os
import threading
import zipfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yamnet.tflite")
FRAME_SAMPLES = 15600  # 0.975 s at 16 kHz, the model's fixed input

SPEECH_CLASSES = {"Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Whispering"}
SHOUT_CLASSES = {"Shout", "Bellow", "Yell", "Children shouting", "Screaming"}
CROWD_CLASSES = {"Chatter", "Crowd", "Hubbub, speech noise, speech babble", "Children playing"}
VOCAL_NON_SPEECH_CLASSES = {
    "Babbling", "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
    "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan", "Sigh", "Humming", "Groan",
    "Grunt", "Whistling", "Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Cough",
    "Throat clearing", "Sneeze", "Sniff", "Chewing, mastication", "Burping, eructation", "Hiccup",
}
MEDIA_CLASSES = {
    "Speech synthesizer", "Singing", "Choir", "Yodeling", "Chant", "Mantra", "Child singing",
    "Synthetic singing", "Rapping", "Television", "Radio",
}
# The AudioSet ontology lists "Music" followed by its whole subtree
# (instruments and genres) up to and including "Scary music".
MUSIC_SUBTREE = ("Music", "Scary music")
DISTRESS_CLASSES = {
    "Screaming", "Shout", "Yell", "Children shouting", "Crying, sobbing", "Baby cry, infant cry",
    "Whimper", "Wail, moan",
}

HUMAN_SPEECH_CONFIDENCE_THRESHOLD = 0.60
GROUP_THRESHOLD = 0.50
DISTRESS_SCORE_THRESHOLD = float(os.environ.get("SAFEEAR_DISTRESS_SCORE_THRESHOLD", "0.50"))

_YAMNET_MODEL = None
_YAMNET_LOAD_ATTEMPTED = False
_LOAD_LOCK = threading.Lock()
# A TFLite interpreter is not thread-safe.
_INFER_LOCK = threading.Lock()
_LABELS: Optional[List[str]] = None
_GROUPS: Optional[Dict[str, np.ndarray]] = None


def load_labels(model_path: str = MODEL_PATH) -> List[str]:
    global _LABELS
    if _LABELS is None:
        with zipfile.ZipFile(model_path) as zf:
            _LABELS = zf.read("yamnet_label_list.txt").decode("utf-8").splitlines()
    return _LABELS


def _indices(labels: Sequence[str], names: set) -> np.ndarray:
    missing = names.difference(labels)
    if missing:
        raise ValueError(f"YAMNet labels missing: {sorted(missing)}")
    return np.array(sorted(labels.index(n) for n in names), dtype=np.int64)


def class_groups() -> Dict[str, np.ndarray]:
    global _GROUPS
    if _GROUPS is None:
        labels = load_labels()
        music = np.arange(labels.index(MUSIC_SUBTREE[0]), labels.index(MUSIC_SUBTREE[1]) + 1)
        _GROUPS = {
            "speech": _indices(labels, SPEECH_CLASSES),
            "shout": _indices(labels, SHOUT_CLASSES),
            "crowd": _indices(labels, CROWD_CLASSES),
            "vocal_non_speech": _indices(labels, VOCAL_NON_SPEECH_CLASSES),
            "media": np.union1d(_indices(labels, MEDIA_CLASSES), music),
            "distress": _indices(labels, DISTRESS_CLASSES),
        }
    return _GROUPS


def _get_yamnet_model():
    """Load the YAMNet TFLite interpreter once; None if unavailable."""
    global _YAMNET_MODEL, _YAMNET_LOAD_ATTEMPTED
    if _YAMNET_MODEL is not None:
        return _YAMNET_MODEL
    with _LOAD_LOCK:
        if _YAMNET_MODEL is not None or _YAMNET_LOAD_ATTEMPTED:
            return _YAMNET_MODEL
        _YAMNET_LOAD_ATTEMPTED = True
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(MODEL_PATH)
            try:
                from ai_edge_litert.interpreter import Interpreter  # lightweight runtime, if installed
            except Exception:
                import tensorflow as tf

                Interpreter = tf.lite.Interpreter
            interpreter = Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            class_groups()
            _YAMNET_MODEL = interpreter
            logger.info("YAMNET_READY | labels=%d", len(load_labels()))
        except Exception as exc:
            logger.error("YAMNET_UNAVAILABLE | using speech-gate fallback | reason=%s", exc)
    return _YAMNET_MODEL


def status() -> Dict[str, Any]:
    return {"backend": "yamnet" if _get_yamnet_model() is not None else "speech_gate_fallback"}


def _frames(waveform: np.ndarray, hop: int = FRAME_SAMPLES // 2) -> List[np.ndarray]:
    w = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if w.shape[0] <= FRAME_SAMPLES:
        pad = FRAME_SAMPLES - w.shape[0]
        return [np.pad(w, (0, pad), mode="reflect") if 0 < pad < w.shape[0] else np.pad(w, (0, pad))]
    starts = list(range(0, w.shape[0] - FRAME_SAMPLES + 1, hop))
    if starts[-1] != w.shape[0] - FRAME_SAMPLES:
        starts.append(w.shape[0] - FRAME_SAMPLES)
    return [w[s:s + FRAME_SAMPLES] for s in starts]


def _scores(waveform: np.ndarray) -> Optional[np.ndarray]:
    """Mean 521-class scores over ~1 s frames, or None if YAMNet is unavailable."""
    model = _get_yamnet_model()
    if model is None:
        return None
    inp = model.get_input_details()[0]["index"]
    out = model.get_output_details()[0]["index"]
    rows = []
    with _INFER_LOCK:
        for frame in _frames(waveform):
            model.set_tensor(inp, frame.astype(np.float32))
            model.invoke()
            rows.append(np.asarray(model.get_tensor(out), dtype=np.float32).reshape(-1))
    return np.mean(np.stack(rows), axis=0)


def decide_from_scores(scores: np.ndarray) -> Dict[str, Any]:
    """Map mean class scores to a SafeEar category (pure; unit-testable)."""
    groups = class_groups()
    labels = load_labels()
    g = {name: float(scores[idx].max()) for name, idx in groups.items()}
    human = max(g["speech"], g["shout"])
    distress_idx = groups["distress"][int(np.argmax(scores[groups["distress"]]))]

    if human >= HUMAN_SPEECH_CONFIDENCE_THRESHOLD and human >= g["media"]:
        if g["crowd"] >= GROUP_THRESHOLD and g["crowd"] > human:
            category, confidence = "crowd", g["crowd"]
        else:
            category, confidence = "human_speech", human
    elif g["media"] >= GROUP_THRESHOLD:
        category, confidence = "reject", g["media"]
    elif g["vocal_non_speech"] >= GROUP_THRESHOLD:
        category, confidence = "vocal_non_speech", g["vocal_non_speech"]
    elif g["crowd"] >= GROUP_THRESHOLD:
        category, confidence = "crowd", g["crowd"]
    else:
        # Not confidently speech: confidence is the speech evidence, not the top
        # score of an arbitrary class (loud music used to pass as "uncertain 0.9").
        category, confidence = "uncertain", human

    top_idx = int(np.argmax(scores))
    return {
        "category": category,
        "confidence": float(confidence),
        "top_class": labels[top_idx],
        "top_score": float(scores[top_idx]),
        "all_scores": {f"group_{k}": round(v, 4) for k, v in g.items()},
        "distress_score": g["distress"],
        "distress_class": labels[distress_idx],
        "method": "yamnet",
    }


def _fallback_classify(waveform: np.ndarray) -> Dict[str, Any]:
    """Used only when YAMNet cannot be loaded: spectral speech gate."""
    from app.utils.speech_gate import assess_speech_likeness

    speech_ok, _ = assess_speech_likeness(waveform, 16000)
    return {
        "category": "human_speech" if speech_ok else "reject",
        "confidence": 0.75 if speech_ok else 0.3,
        "top_class": "speech" if speech_ok else "unknown",
        "top_score": 0.0,
        "all_scores": {},
        "distress_score": 0.0,
        "distress_class": None,
        "method": "fallback_gate",
    }


def classify_audio_event_windowed(waveform: np.ndarray, sr: int = 16000, **_: Any) -> Dict[str, Any]:
    """Classify a window (any length) by averaging YAMNet scores over ~1 s frames."""
    w = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if w.size == 0:
        return {"category": "reject", "confidence": 0.0, "top_class": "unknown", "top_score": 0.0,
                "all_scores": {}, "distress_score": 0.0, "distress_class": None, "method": "empty"}
    try:
        scores = _scores(w)
    except Exception as exc:
        logger.warning("YAMNET_INFERENCE_FAILED | reason=%s", exc)
        scores = None
    if scores is None:
        return _fallback_classify(w)
    result = decide_from_scores(scores)
    result["num_frames"] = len(_frames(w))
    return result


def classify_audio_event(waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    return classify_audio_event_windowed(waveform, sr)


def distress_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    score = float(result.get("distress_score", 0.0) or 0.0)
    return {
        "distress": score >= DISTRESS_SCORE_THRESHOLD,
        "distress_score": score,
        "distress_class": result.get("distress_class"),
    }


def classify_distress_event(waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    return distress_from_result(classify_audio_event_windowed(waveform, sr))
