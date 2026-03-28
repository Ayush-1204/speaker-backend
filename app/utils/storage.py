import os
import numpy as np

BASE_DIR = os.path.join("app", "data", "familiar_embeddings")
EXPECTED_DIM = 192


def ensure_dir():
    os.makedirs(BASE_DIR, exist_ok=True)


def save_embedding(name: str, emb: np.ndarray):
    ensure_dir()
    user_dir = os.path.join(BASE_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    emb = np.asarray(emb, dtype=np.float32).squeeze()
    if emb.ndim != 1:
        raise ValueError(f"Embedding must be 1-D array, got shape {emb.shape}")

    count = len([f for f in os.listdir(user_dir) if f.endswith(".npy")])
    filename = os.path.join(user_dir, f"{name}_{count+1}.npy")
    np.save(filename, emb)


def load_user_embeddings(name: str):
    user_dir = os.path.join(BASE_DIR, name)
    if not os.path.exists(user_dir):
        return []

    embeddings = []
    for file in sorted(os.listdir(user_dir)):
        if not file.endswith(".npy"):
            continue
        path = os.path.join(user_dir, file)
        try:
            arr = np.load(path)
        except Exception:
            print(f"[storage] Warning: failed to load {path}")
            continue

        arr = np.asarray(arr, dtype=np.float32).squeeze()

        if arr.ndim != 1:
            print(f"[storage] Warning: skipping embedding with wrong ndim {arr.shape} at {path}")
            continue

        if arr.shape[0] != EXPECTED_DIM:
            print(f"[storage] Warning: skipping embedding with wrong length {arr.shape} at {path}")
            continue

        embeddings.append(arr)

    return embeddings


def load_all_embeddings():
    ensure_dir()
    users = {}
    for user in sorted(os.listdir(BASE_DIR)):
        path = os.path.join(BASE_DIR, user)
        if os.path.isdir(path):
            embs = load_user_embeddings(user)
            if embs:
                users[user] = embs
    return users
