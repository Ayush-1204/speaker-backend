import io
import os
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa


def decode_audio_chunk(raw: bytes) -> Tuple[np.ndarray, int, str]:
    """Decode audio bytes into a numpy waveform and sample rate.

    Tries soundfile, then librosa, then raw PCM16LE fallback.
    Returns: (array(float32), sample_rate, decode_method)
    """
    # 1) Preferred path: decode containerized audio (wav/ogg/webm/etc.) with soundfile.
    try:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        return np.asarray(arr), int(sr), "soundfile"
    except Exception:
        pass

    # 2) Fallback path: librosa may decode formats unavailable to soundfile.
    try:
        arr, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return np.asarray(arr, dtype=np.float32), int(sr), "librosa"
    except Exception:
        pass

    # 3) Last resort: treat payload as raw PCM16LE mono.
    if len(raw) >= 2 and (len(raw) % 2 == 0):
        raw_sr = int(os.environ.get("SAFEEAR_RAW_PCM_SR", "16000"))
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if arr.size > 0:
            return arr, raw_sr, "raw_pcm16le"

    raise ValueError("unsupported_audio_chunk_format")
