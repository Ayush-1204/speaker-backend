# app/utils/compare.py
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

THRESHOLD = 0.75   

def match_embedding(query_emb, user_embeddings):
    """
    Returns (max_similarity)
    """
    if len(user_embeddings) == 0:
        return 0.0

    sims = [cosine_similarity(query_emb, e) for e in user_embeddings]
    return max(sims)
