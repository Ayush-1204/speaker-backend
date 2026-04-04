"""Audio preprocessing helpers used across detect/enroll/verify pipelines."""

import numpy as np

from . import config


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Normalize loudness with conservative gain limits to stabilize embeddings."""
    w = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if w.size == 0 or not config.ENABLE_VOLUME_NORMALIZATION:
        return w

    # Remove DC offset before RMS/peak checks.
    w = w - np.float32(np.mean(w))

    rms = float(np.sqrt(np.mean(w**2) + 1e-12))
    if rms < config.VOLUME_MIN_RMS:
        return w

    max_gain = float(10.0 ** (config.VOLUME_MAX_GAIN_DB / 20.0))
    target_gain = float(config.VOLUME_TARGET_RMS / max(rms, 1e-12))
    gain = float(np.clip(target_gain, 1.0 / max_gain, max_gain))
    w = w * np.float32(gain)

    peak = float(np.max(np.abs(w)) + 1e-12)
    if peak > config.VOLUME_PEAK_LIMIT:
        w = w * np.float32(config.VOLUME_PEAK_LIMIT / peak)

    return w.astype(np.float32, copy=False)
