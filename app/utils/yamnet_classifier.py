"""YAMNet audio event classifier (Tier 2 of detection pipeline)."""

import logging
import numpy as np
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

YAMNET_HUMAN_CLASSES = {
    "speech", "conversation", "whispering", "shout", "child speech", "talk",
    "male speech", "female speech", "man speaking", "woman speaking"
}

YAMNET_VOCAL_NON_SPEECH = {
    "laughter", "crying", "cough", "sneeze", "throat clearing", "choking",
}

YAMNET_REJECT = {
    "music", "television", "radio", "dog", "barking", "cat", "meow",
    "door slam", "door knock", "static", "white noise", "tone"
}

# Change 7: Distress classes — screaming, crying, aggression.
# These map to the simplified yamnet_classes list in _score_to_decision.
# "shout" and "baby_cry" are already tracked; "crying" is in vocal_non_speech
# above but also counted here so both paths can fire independently.
YAMNET_DISTRESS_CLASSES: set = {
    "shout",
    "baby_cry",
    "crying",
    "screaming",    # may appear in full YAMNet class list as a score index
    "whimpering",
    "aggressive",
    "child_speech",  # included so high-energy child speech can contribute to distress scoring
}

HUMAN_SPEECH_CONFIDENCE_THRESHOLD = 0.60
DISTRESS_SCORE_THRESHOLD = float(__import__("os").environ.get("SAFEEAR_DISTRESS_SCORE_THRESHOLD", "0.25"))


_YAMNET_MODEL = None
_YAMNET_LOAD_ATTEMPTED = False


def _get_yamnet_model():
    """Load YAMNet TFLite model (lazy singleton)."""
    global _YAMNET_MODEL, _YAMNET_LOAD_ATTEMPTED
    if _YAMNET_MODEL is None and not _YAMNET_LOAD_ATTEMPTED:
        _YAMNET_LOAD_ATTEMPTED = True
        try:
            import tensorflow as tf
            model_path = os.path.join("app", "models", "yamnet.tflite")
            if not os.path.exists(model_path):
                logger.warning("[yamnet] Model not found at %s, using fallback.", model_path)
                return None
            _YAMNET_MODEL = tf.lite.Interpreter(model_path=model_path)
            _YAMNET_MODEL.allocate_tensors()
        except Exception as e:
            logger.warning("[yamnet] Failed to load TFLite model: %s", str(e))
            return None
    return _YAMNET_MODEL


def classify_audio_event(waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """
    Classify audio using YAMNet. Returns:
    {
        "category": "human_speech" | "vocal_non_speech" | "reject" | "uncertain",
        "confidence": float (0-1),
        "top_class": str,
        "all_scores": dict of class -> score
    }
    """
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)

    if len(waveform) < 15600:
        waveform = np.pad(waveform, (0, 15600 - len(waveform)), mode="reflect")

    model = _get_yamnet_model()
    if model is None:
        return _fallback_classify(waveform)

    try:
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        input_shape = tuple(int(x) for x in input_details[0].get("shape", [15600]))
        # Typical YAMNet-Lite input is 15600 samples. Adapt length to model expectation.
        if len(input_shape) >= 1 and input_shape[-1] > 0:
            expected_len = int(input_shape[-1])
        else:
            expected_len = 15600

        if waveform.shape[0] < expected_len:
            in_wave = np.pad(waveform, (0, expected_len - waveform.shape[0]), mode="reflect")
        elif waveform.shape[0] > expected_len:
            # Center-crop to expected duration to avoid set_tensor dimension mismatch.
            start = (waveform.shape[0] - expected_len) // 2
            in_wave = waveform[start : start + expected_len]
        else:
            in_wave = waveform

        in_wave = np.asarray(in_wave, dtype=np.float32)

        if len(input_shape) == 2 and input_shape[0] == 1:
            input_tensor = in_wave.reshape(1, -1)
        elif len(input_shape) == 1:
            input_tensor = in_wave
        else:
            # Fallback for uncommon model signatures.
            input_tensor = in_wave.reshape(input_shape)

        model.set_tensor(input_details[0]["index"], input_tensor)
        model.invoke()

        scores = model.get_tensor(output_details[0]["index"])
        scores = np.mean(scores, axis=0).astype(np.float32)

        return _score_to_decision(scores)
    except Exception as e:
        logger.warning("[yamnet] Inference failed: %s", str(e))
        return _fallback_classify(waveform)


def classify_audio_event_windowed(
    waveform: np.ndarray,
    sr: int = 16000,
    frame_sec: float = 0.96,
    hop_sec: float = 0.48,
) -> Dict[str, Any]:
    """Aggregate YAMNet decisions over overlapping near-native frames."""
    w = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if w.size == 0:
        return {
            "category": "reject",
            "confidence": 0.0,
            "top_class": "unknown",
            "all_scores": {"unknown": 0.0},
            "method": "yamnet_windowed",
            "num_frames": 0,
        }

    frame_len = max(1, int(frame_sec * sr))
    hop_len = max(1, int(hop_sec * sr))

    frames = []
    if w.shape[0] <= frame_len:
        frames = [w]
    else:
        for start in range(0, max(1, w.shape[0] - frame_len + 1), hop_len):
            end = start + frame_len
            if end > w.shape[0]:
                break
            frames.append(w[start:end])
        if not frames:
            frames = [w[-frame_len:]]

    decisions = [classify_audio_event(f, sr=sr) for f in frames]

    category_votes: Dict[str, int] = {}
    category_conf: Dict[str, list] = {}
    for d in decisions:
        c = str(d.get("category", "uncertain"))
        conf = float(d.get("confidence", 0.0))
        category_votes[c] = category_votes.get(c, 0) + 1
        category_conf.setdefault(c, []).append(conf)

    def _cat_key(cat: str):
        confs = category_conf.get(cat, [0.0])
        return (category_votes.get(cat, 0), float(np.mean(confs)))

    best_category = max(category_votes.keys(), key=_cat_key)
    best_conf = float(np.median(category_conf.get(best_category, [0.0])))

    top_class_votes: Dict[str, int] = {}
    for d in decisions:
        c = str(d.get("top_class", "unknown"))
        top_class_votes[c] = top_class_votes.get(c, 0) + 1
    best_top_class = max(top_class_votes.keys(), key=lambda k: top_class_votes[k])

    return {
        "category": best_category,
        "confidence": best_conf,
        "top_class": best_top_class,
        "all_scores": {
            f"cat_{cat}": float(np.mean(confs))
            for cat, confs in category_conf.items()
        },
        "method": "yamnet_windowed",
        "num_frames": len(frames),
        "frame_sec": frame_sec,
        "hop_sec": hop_sec,
    }


def _fallback_classify(waveform: np.ndarray) -> Dict[str, Any]:
    """Fallback when YAMNet is unavailable. Use speech gate + energy."""
    from app.utils.speech_gate import assess_speech_likeness

    speech_ok, _ = assess_speech_likeness(waveform, 16000)
    if speech_ok:
        return {
            "category": "human_speech",
            "confidence": 0.75,
            "top_class": "speech",
            "all_scores": {"speech": 0.75},
            "method": "fallback_gate"
        }
    return {
        "category": "reject",
        "confidence": 0.3,
        "top_class": "unknown",
        "all_scores": {"unknown": 0.3},
        "method": "fallback_gate"
    }


def _score_to_decision(scores: np.ndarray) -> Dict[str, Any]:
    """Map YAMNet scores to SafeEar categories."""
    yamnet_classes = [
        "speech", "male_speech", "female_speech", "child_speech", "conversation",
        "whispering", "shout", "baby_cry", "laughter", "cough", "sneeze", "door",
        "music", "television", "telephone", "dog", "cat", "birds"
    ]

    top_idx = np.argmax(scores)
    top_score = float(scores[top_idx])
    top_class = yamnet_classes[top_idx] if top_idx < len(yamnet_classes) else "unknown"

    human_score = sum(
        float(scores[i]) for i, c in enumerate(yamnet_classes)
        if c.lower() in YAMNET_HUMAN_CLASSES and i < len(scores)
    )
    vocal_non_speech_score = sum(
        float(scores[i]) for i, c in enumerate(yamnet_classes)
        if c.lower() in YAMNET_VOCAL_NON_SPEECH and i < len(scores)
    )
    reject_score = sum(
        float(scores[i]) for i, c in enumerate(yamnet_classes)
        if c.lower() in YAMNET_REJECT and i < len(scores)
    )

    category = "uncertain"
    confidence = 0.0

    if human_score >= HUMAN_SPEECH_CONFIDENCE_THRESHOLD:
        category = "human_speech"
        confidence = human_score
    elif vocal_non_speech_score > 0.5:
        category = "vocal_non_speech"
        confidence = vocal_non_speech_score
    elif reject_score > 0.6:
        category = "reject"
        confidence = reject_score
    else:
        category = "uncertain"
        confidence = top_score

    return {
        "category": category,
        "confidence": float(confidence),
        "top_class": top_class,
        "all_scores": {yamnet_classes[i] if i < len(yamnet_classes) else f"class_{i}": float(scores[i]) for i in range(len(scores))},
        "method": "yamnet"
    }


def classify_distress_event(waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """Distress detection using YAMNet class scores (Change 7).

    Runs windowed classification and sums scores for YAMNET_DISTRESS_CLASSES.
    Returns a dict with:
        distress        : bool    — True when distress_score >= DISTRESS_SCORE_THRESHOLD
        distress_score  : float   — aggregate distress evidence in [0, 1]
        distress_class  : str     — the distress class with the highest individual score

    This function is designed to run in parallel with the speaker embedding path.
    It adds no model loading overhead because it reuses the same YAMNet singleton.
    """
    # yamnet_classes must match the order used in _score_to_decision
    yamnet_classes = [
        "speech", "male_speech", "female_speech", "child_speech", "conversation",
        "whispering", "shout", "baby_cry", "laughter", "cough", "sneeze", "door",
        "music", "television", "telephone", "dog", "cat", "birds"
    ]
    distress_normalised = {c.lower().replace(" ", "_") for c in YAMNET_DISTRESS_CLASSES}

    try:
        result = classify_audio_event_windowed(waveform, sr=sr)
        all_scores = result.get("all_scores", {})

        # Sum scores that correspond to distress classes.
        distress_score = 0.0
        best_distress_class: str = ""
        best_distress_val: float = 0.0

        for cls, score in all_scores.items():
            # all_scores keys are prefixed with "cat_" from windowed aggregation
            raw_cls = str(cls).replace("cat_", "").lower().replace(" ", "_")
            if raw_cls in distress_normalised:
                distress_score += float(score)
                if float(score) > best_distress_val:
                    best_distress_val = float(score)
                    best_distress_class = raw_cls

        distress_score = min(1.0, distress_score)
        is_distress = distress_score >= DISTRESS_SCORE_THRESHOLD

        if is_distress:
            logger.info(
                "[distress] DISTRESS_DETECTED | score=%.4f | class=%s",
                distress_score, best_distress_class,
            )

        return {
            "distress": is_distress,
            "distress_score": float(distress_score),
            "distress_class": best_distress_class or None,
        }

    except Exception as exc:
        logger.warning("[distress] classify_distress_event failed: %s", str(exc))
        return {
            "distress": False,
            "distress_score": 0.0,
            "distress_class": None,
        }
