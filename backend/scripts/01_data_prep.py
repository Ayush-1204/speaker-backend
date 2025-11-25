#!/usr/bin/env python3
"""
01_data_prep.py
Standardize raw audio to 16kHz mono, trim silence, save under data/processed_16k/<speaker>/*.wav
Run: python3 01_data_prep.py
"""
from pathlib import Path
import librosa, soundfile as sf
import numpy as np

ROOT = Path.cwd()  # run this from backend project root
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed_16k"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_resample_16k(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y.astype("float32")

def trim_silence(y, top_db=30):
    try:
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return yt
    except Exception:
        return y

def process_all():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw folder not found: {RAW_DIR}")
    for sp in sorted(RAW_DIR.iterdir()):
        if not sp.is_dir(): continue
        out_sp = OUT_DIR / sp.name
        out_sp.mkdir(parents=True, exist_ok=True)
        for wav in sp.glob("*.wav"):
            try:
                y = load_resample_16k(str(wav))
                y = trim_silence(y)
                sf.write(out_sp / wav.name, y, 16000)
            except Exception as e:
                print("Failed:", wav, e)

if __name__ == "__main__":
    print("Processing raw audio -> processed_16k (16kHz mono, trimmed)...")
    process_all()
    print("Done.")
