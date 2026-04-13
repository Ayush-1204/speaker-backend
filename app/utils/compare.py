import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


THRESHOLD = 0.75


def match_embedding(query_emb, user_embeddings):
    if len(user_embeddings) == 0:
        return 0.0

    sims = [cosine_similarity(query_emb, e) for e in user_embeddings]
    return max(sims)
