# app/utils/feature_extractor.py
"""
ECAPA embedding extractor (SpeechBrain) with VAD.
- Loads audio with soundfile + librosa resampling
- Applies VAD (silero fallback already implemented in vad.py)
- Pads too-short audio so ECAPA never crashes
- Returns a 1-D L2-normalized numpy.float32 vector of shape (192,)
"""

import os
import numpy as np
import soundfile as sf
import librosa
import torch
from huggingface_hub import snapshot_download
from speechbrain.inference import EncoderClassifier

from .vad import apply_vad

import torchaudio
torchaudio.set_audio_backend("soundfile")

# Device and model dir
DEVICE = "cpu"
MODEL_DIR = os.path.join("pretrained_models", "ecapa")

# Ensure model is present (snapshot download will cache)
if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id="speechbrain/spkrec-ecapa-voxceleb",
        local_dir=MODEL_DIR,
        ignore_patterns=["*.ckpt"]
    )

# Load ECAPA encoder (force CPU)
ECAPA = EncoderClassifier.from_hparams(
    source=MODEL_DIR,
    run_opts={"device": DEVICE},
)

# Minimum length ECAPA needs (1 sec @ 16kHz)
MIN_SAMPLES = 16000


def _load_audio(path: str, target_sr: int = 16000):
    """Load audio file, convert to mono, resample."""
    data, sr = sf.read(path, dtype="float32")

    if data.ndim == 2:
        data = data.mean(axis=1)

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data.astype(np.float32), target_sr


def get_ecapa_embedding_from_file(path: str):
    """Full embedding pipeline."""

    # 1️⃣ LOAD
    waveform, sr = _load_audio(path, target_sr=16000)
    print("\n🔍 DEBUG → Loaded WAV:", path, "Length:", len(waveform), "SR:", sr)

    # 2️⃣ VAD
    try:
        waveform_vad = apply_vad(waveform, sr=sr, use_silero=True)
    except Exception:
        waveform_vad = apply_vad(waveform, sr=sr, use_silero=False)

    print("🔍 DEBUG → After VAD length:", len(waveform_vad))

    # If VAD removed too much audio → restore original
    if len(waveform_vad) < 160:
        print("⚠️ DEBUG → VAD too small; using original audio")
        waveform_vad = waveform

    # 3️⃣ **CRITICAL FIX — PAD SHORT AUDIO**
    if len(waveform_vad) < MIN_SAMPLES:
        missing = MIN_SAMPLES - len(waveform_vad)
        waveform_vad = np.pad(waveform_vad, (0, missing), mode="reflect")
        print(f"🟦 DEBUG → Audio padded to {len(waveform_vad)} samples")

    # 4️⃣ Torch tensor [batch, time]
    tensor = torch.tensor(waveform_vad, dtype=torch.float32).unsqueeze(0)
    print("🔍 DEBUG → Tensor shape before ECAPA:", tensor.shape)

    # 5️⃣ ECAPA forward
    with torch.no_grad():
        emb_tensor = ECAPA.encode_batch(tensor)

    # 6️⃣ Convert to numpy
    emb = emb_tensor.squeeze().cpu().numpy().astype(np.float32)
    emb = np.asarray(emb).squeeze()

    # 7️⃣ Flatten if needed
    if emb.ndim != 1:
        emb = emb.reshape(-1)

    # 8️⃣ Fix wrong shapes
    if emb.shape[0] != 192:
        print("⚠️ DEBUG → Unexpected embedding shape:", emb.shape)
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        emb = emb.flatten()

    # 9️⃣ Normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    # 🔟 Final validation
    if emb.ndim != 1 or emb.shape[0] != 192:
        raise RuntimeError(f"❌ Invalid embedding shape after processing: {emb.shape}")

    print("✅ DEBUG → Final embedding shape:", emb.shape)

    return emb

