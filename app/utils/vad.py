# app/utils/vad.py
"""
Voice activity detection: Silero VAD (installed `silero-vad` package), with an
energy-based fallback that is used only if Silero cannot be loaded.

Earlier versions loaded Silero through torch.hub and read
`utils.get_speech_timestamps` from a tuple, which always raised; the error was
swallowed and every window silently used the energy fallback (which marks
noise, hum and music as speech).  `status()` now reports which backend is live
so a degraded VAD is visible in logs and /health.
"""

import logging
import os
import threading
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None

logger = logging.getLogger(__name__)

_SILERO_MODEL = None
_SILERO_GET_TS = None
_SILERO_LOAD_ATTEMPTED = False
_SILERO_ERROR: Optional[str] = None
_LOAD_LOCK = threading.Lock()
# The Silero model keeps internal state between frames, so calls are serialized.
_INFER_LOCK = threading.Lock()


def _silero_kwargs() -> dict:
    """Optional Silero overrides from env.  Unset keeps Silero's own defaults."""
    kwargs: dict = {}
    if os.environ.get("SAFEEAR_VAD_THRESHOLD"):
        kwargs["threshold"] = float(os.environ["SAFEEAR_VAD_THRESHOLD"])
    if os.environ.get("SAFEEAR_VAD_MIN_SILENCE_MS"):
        kwargs["min_silence_duration_ms"] = int(os.environ["SAFEEAR_VAD_MIN_SILENCE_MS"])
    if os.environ.get("SAFEEAR_VAD_MIN_SPEECH_MS"):
        kwargs["min_speech_duration_ms"] = int(os.environ["SAFEEAR_VAD_MIN_SPEECH_MS"])
    return kwargs


# torch's own default (physical cores), captured before silero_vad can lower it.
_DEFAULT_TORCH_THREADS = torch.get_num_threads() if torch is not None else 1


def configured_torch_threads() -> int:
    """SAFEEAR_TORCH_THREADS if set.  On a container host set it to the number of
    CPUs the plan actually provides; the default can see more cores than the quota."""
    return max(1, int(os.environ.get("SAFEEAR_TORCH_THREADS", str(_DEFAULT_TORCH_THREADS))))


def restore_torch_threads() -> None:
    """Importing silero_vad calls torch.set_num_threads(1) globally, which made the
    speaker-embedding model ~3x slower.  Re-apply the intended thread count."""
    if torch is not None:
        torch.set_num_threads(configured_torch_threads())


def _get_silero():
    """Load Silero once; return (model, get_speech_timestamps) or (None, None)."""
    global _SILERO_MODEL, _SILERO_GET_TS, _SILERO_LOAD_ATTEMPTED, _SILERO_ERROR
    if _SILERO_MODEL is not None:
        return _SILERO_MODEL, _SILERO_GET_TS
    with _LOAD_LOCK:
        if _SILERO_MODEL is not None or _SILERO_LOAD_ATTEMPTED:
            return _SILERO_MODEL, _SILERO_GET_TS
        _SILERO_LOAD_ATTEMPTED = True
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            model = load_silero_vad()
            restore_torch_threads()
            _SILERO_MODEL, _SILERO_GET_TS = model, get_speech_timestamps
            logger.info("VAD_READY | backend=silero | torch_threads=%d", configured_torch_threads())
        except Exception as exc:
            _SILERO_ERROR = str(exc)
            restore_torch_threads()
            logger.error("VAD_DEGRADED | backend=energy_fallback | reason=%s", exc)
    return _SILERO_MODEL, _SILERO_GET_TS


def status() -> dict:
    model, _ = _get_silero()
    return {"backend": "silero" if model is not None else "energy_fallback", "error": _SILERO_ERROR}


def _silero_segments(audio: np.ndarray, sr: int) -> Optional[List[Tuple[int, int]]]:
    """Silero segments, or None if Silero is unavailable.  [] means no speech."""
    model, get_ts = _get_silero()
    if model is None or torch is None:
        return None
    audio_t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    with _INFER_LOCK, torch.inference_mode():
        stamps = get_ts(audio_t, model, sampling_rate=sr, **_silero_kwargs())
    return [(int(t["start"]), int(t["end"])) for t in stamps]


def _energy_segments(audio: np.ndarray, sr: int, frame_ms: int = 30, threshold_ratio: float = 0.5) -> List[Tuple[int, int]]:
    """Pseudo speech segments in samples (fallback only; cannot tell speech from noise)."""
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0 or len(audio) == 0:
        return [(0, len(audio))] if len(audio) else []

    n_frames = int(np.ceil(len(audio) / frame_len))
    pad_len = n_frames * frame_len - len(audio)
    audio_p = np.concatenate([audio, np.zeros(pad_len, dtype=audio.dtype)]) if pad_len > 0 else audio

    energy = np.mean(np.abs(audio_p.reshape(n_frames, frame_len)), axis=1)
    if np.all(energy == 0):
        return []

    mask = energy > max(1e-6, energy.mean() * threshold_ratio)
    segments: List[Tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        start, end = i * frame_len, min(len(audio), j * frame_len)
        if end > start:
            segments.append((start, end))
        i = j
    return segments


def get_speech_segments(audio: np.ndarray, sr: int = 16000, use_silero: bool = True) -> List[Tuple[int, int]]:
    """Disjoint voiced segments as half-open (start_sample, end_sample).  [] = no speech."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    # SAFEEAR_VAD_BACKEND=energy reproduces the pre-fix behaviour, for A/B comparison only.
    if use_silero and os.environ.get("SAFEEAR_VAD_BACKEND", "silero").strip().lower() != "energy":
        try:
            segments = _silero_segments(audio, sr)
            if segments is not None:
                return segments
        except Exception as exc:
            logger.warning("VAD_SILERO_CALL_FAILED | reason=%s", exc)
    return _energy_segments(audio, sr)


def apply_vad(audio: np.ndarray, sr: int = 16000, use_silero: bool = True) -> np.ndarray:
    """Trim leading/trailing non-speech; returns the input unchanged when no speech is found."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    segments = get_speech_segments(audio, sr, use_silero=use_silero)
    if not segments:
        return audio
    start, end = max(0, segments[0][0]), min(len(audio), segments[-1][1])
    return audio[start:end] if end > start else audio
