#!/usr/bin/env python3
"""
02_embeddings_yamnet.py
Extract YAMNet (1024-d) embeddings for all files in data/processed_16k and save:
data/features_yamnet/X_yamnet.npy, y_yamnet.npy, label2idx.json
Run: python3 02_embeddings_yamnet.py
"""
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import json
from tqdm import tqdm

ROOT = Path.cwd()
PROCESSED = ROOT / "data" / "processed_16k"
FEATURES = ROOT / "data" / "features_yamnet"
FEATURES.mkdir(parents=True, exist_ok=True)

# Load YAMNet from TF-Hub
print("Loading YAMNet from TF-Hub...")
YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded.")

def load_wav_16k(path):
    y, sr = sf.read(str(path), dtype='float32')
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y.astype("float32")

def yamnet_embedding_from_wav(path):
    waveform = load_wav_16k(path)
    scores, embeddings, _ = YAMNET(waveform)
    emb = embeddings.numpy().mean(axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype("float32")

def main():
    X = []
    y = []
    label2idx = {}
    idx = 0
    for sp in sorted(PROCESSED.iterdir()):
        if not sp.is_dir(): continue
        label2idx[sp.name] = idx
        print("Processing speaker:", sp.name)
        for wav in tqdm(sorted(sp.glob("*.wav"))):
            try:
                emb = yamnet_embedding_from_wav(str(wav))
                X.append(emb)
                y.append(idx)
            except Exception as e:
                print("Error on", wav, e)
        idx += 1
    X = np.stack(X) if X else np.zeros((0,1024), dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    np.save(FEATURES / "X_yamnet.npy", X)
    np.save(FEATURES / "y_yamnet.npy", y)
    with open(FEATURES / "label2idx.json", "w") as f:
        json.dump(label2idx, f, indent=2)
    print("Saved features:", (FEATURES / "X_yamnet.npy"), X.shape, y.shape)
    print("label2idx:", label2idx)

if __name__ == "__main__":
    main()
