"""YAMNet audio event classifier (Tier 2 of detection pipeline)."""

import numpy as np
import os
from typing import Dict, Any, Tuple

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

HUMAN_SPEECH_CONFIDENCE_THRESHOLD = 0.60


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
                print(f"[yamnet] Warning: model not found at {model_path}. Using fallback.")
                return None
            _YAMNET_MODEL = tf.lite.Interpreter(model_path=model_path)
            _YAMNET_MODEL.allocate_tensors()
        except Exception as e:
            print(f"[yamnet] Failed to load TFLite model: {e}")
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
        print(f"[yamnet] Inference failed: {e}")
        return _fallback_classify(waveform)


def _fallback_classify(waveform: np.ndarray) -> Dict[str, Any]:
    """Fallback when YAMNet is unavailable. Use speech gate + energy."""
    from app.utils.speech_gate import assess_speech_likeness

    speech_ok, metrics = assess_speech_likeness(waveform, 16000)
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
