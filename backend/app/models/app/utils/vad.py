# backend/app/utils/vad.py
"""
Silero-based VAD with energy fallback.
Returns a trimmed numpy float32 waveform (speech only) or original audio if VAD fails.
"""

import numpy as np
import torch

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
    energy = frames.mean(axis=1)
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
            # load silero from github via torch.hub (will cache)
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False
            )
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
