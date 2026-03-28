"""Stage 2: reject obvious non-speech before speaker embedding."""

from typing import Any, Dict, Tuple

import librosa
import numpy as np

from . import config


def assess_speech_likeness(waveform: np.ndarray, sr: int) -> Tuple[bool, Dict[str, Any]]:
    w = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if len(w) < 512:
        return False, {"reason": "too_short", "samples": len(w)}

    rms = float(np.sqrt(np.mean(w**2) + 1e-12))
    if rms < config.SPEECH_GATE_MIN_RMS:
        return False, {"reason": "low_rms", "rms": rms}

    S = np.abs(librosa.stft(w.astype(np.float64), n_fft=512, hop_length=160, win_length=400))
    flat = float(librosa.feature.spectral_flatness(S=S).mean())
    cent = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())

    passed = (
        flat <= config.SPEECH_GATE_MAX_FLATNESS
        and config.SPEECH_GATE_CENTROID_MIN_HZ <= cent <= config.SPEECH_GATE_CENTROID_MAX_HZ
    )

    detail: Dict[str, Any] = {
        "rms": rms,
        "spectral_flatness": flat,
        "spectral_centroid_hz": cent,
    }
    if not passed:
        detail["reason"] = "not_speech_like"
    return passed, detail
