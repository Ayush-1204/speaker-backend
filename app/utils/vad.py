# app/utils/vad.py
"""
Silero-based VAD with energy fallback.
Returns a trimmed numpy float32 waveform (speech only) or original audio if VAD fails.
"""

from typing import List, Tuple

import numpy as np
import torch


_SILERO_MODEL = None
_SILERO_UTILS = None
_SILERO_LOAD_ATTEMPTED = False


def _get_silero():
    """Load Silero once; return (model, utils) or (None, None) on failure."""
    global _SILERO_MODEL, _SILERO_UTILS, _SILERO_LOAD_ATTEMPTED
    if _SILERO_MODEL is not None and _SILERO_UTILS is not None:
        return _SILERO_MODEL, _SILERO_UTILS
    if _SILERO_LOAD_ATTEMPTED:
        return None, None

    _SILERO_LOAD_ATTEMPTED = True
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        _SILERO_MODEL = model
        _SILERO_UTILS = utils
        return _SILERO_MODEL, _SILERO_UTILS
    except Exception:
        return None, None


def _energy_segments(audio: np.ndarray, sr: int, frame_ms: int = 30, threshold_ratio: float = 0.5) -> List[dict]:
    """Return pseudo speech segments [{start, end}, ...] in samples (energy fallback)."""
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0:
        return [{"start": 0, "end": len(audio)}]

    n_frames = int(np.ceil(len(audio) / frame_len))
    pad_len = n_frames * frame_len - len(audio)
    if pad_len > 0:
        audio_p = np.concatenate([audio, np.zeros(pad_len, dtype=audio.dtype)])
    else:
        audio_p = audio

    frames = audio_p.reshape(n_frames, frame_len)
    # Use absolute-energy per frame; signed mean is unstable around zero.
    energy = np.mean(np.abs(frames), axis=1)
    if np.all(energy == 0):
        return [{"start": 0, "end": len(audio)}]

    thresh = max(1e-6, energy.mean() * threshold_ratio)
    mask = energy > thresh
    if not mask.any():
        return [{"start": 0, "end": len(audio)}]

    segments = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        start = i * frame_len
        end = min(len(audio), j * frame_len)
        if end > start:
            segments.append({"start": int(start), "end": int(end)})
        i = j
    return segments if segments else [{"start": 0, "end": len(audio)}]


def _energy_vad_trim(audio: np.ndarray, sr: int, frame_ms: int = 30, threshold_ratio: float = 0.5):
    """Simple energy-based VAD fallback"""
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0:
        return audio

    # pad to full frames
    n_frames = int(np.ceil(len(audio) / frame_len))
    pad_len = n_frames * frame_len - len(audio)
    if pad_len > 0:
        audio_p = np.concatenate([audio, np.zeros(pad_len, dtype=audio.dtype)])
    else:
        audio_p = audio

    frames = audio_p.reshape(n_frames, frame_len)
    # Use absolute-energy per frame; signed mean is unstable around zero.
    energy = np.mean(np.abs(frames), axis=1)
    if np.all(energy == 0):
        return audio

    thresh = max(1e-6, energy.mean() * threshold_ratio)
    mask = energy > thresh
    if not mask.any():
        return audio

    first = np.argmax(mask)
    last = len(mask) - 1 - np.argmax(mask[::-1])
    start_sample = first * frame_len
    end_sample = min(len(audio), (last + 1) * frame_len)
    return audio[start_sample:end_sample]


def get_speech_segments(audio: np.ndarray, sr: int = 16000, use_silero: bool = True) -> List[Tuple[int, int]]:
    """
    Return disjoint voiced segments as (start_sample, end_sample) in half-open style [start, end).
    """
    audio = np.asarray(audio, dtype=np.float32)

    if use_silero:
        try:
            model, utils = _get_silero()
            if model is not None and utils is not None:
                get_speech_ts = utils.get_speech_timestamps
                audio_t = torch.tensor(audio, dtype=torch.float32)
                speech_timestamps = get_speech_ts(audio_t, model, sampling_rate=sr)
                if speech_timestamps:
                    return [(int(t["start"]), int(t["end"])) for t in speech_timestamps]
        except Exception:
            pass

    segs = _energy_segments(audio, sr)
    return [(int(s["start"]), int(s["end"])) for s in segs]


def apply_vad(audio: np.ndarray, sr: int = 16000, use_silero: bool = True):
    """
    Trim silence from audio using Silero VAD when available, otherwise energy-based fallback.

    Args:
        audio: 1-D float32 numpy waveform (range roughly [-1,1])
        sr: sample rate (expected 16000)
        use_silero: attempt silero torch.hub model if True

    Returns:
        trimmed audio (numpy float32)
    """

    # ensure numpy array float32
    audio = np.asarray(audio, dtype=np.float32)

    if sr != 16000:
        # Silero models are trained for 16k, but we don't resample here.
        # Caller should resample before calling apply_vad.
        pass

    if use_silero:
        try:
            model, utils = _get_silero()
            if model is None or utils is None:
                return _energy_vad_trim(audio, sr)

            get_speech_ts = utils.get_speech_timestamps
            # model expects torch tensor on cpu
            audio_t = torch.tensor(audio, dtype=torch.float32)
            speech_timestamps = get_speech_ts(audio_t, model, sampling_rate=sr)

            if not speech_timestamps:
                return _energy_vad_trim(audio, sr)

            # merge contiguous segments — keep from first start to last end
            start = speech_timestamps[0]["start"]
            end = speech_timestamps[-1]["end"]
            # ensure ints and within bounds
            start = max(0, int(start))
            end = min(len(audio), int(end))
            if start >= end:
                return _energy_vad_trim(audio, sr)
            return audio[start:end]
        except Exception:
            # any failure -> fallback
            return _energy_vad_trim(audio, sr)
    else:
        return _energy_vad_trim(audio, sr)
