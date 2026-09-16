import logging
import os
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

_COHORT_PATH = os.path.join(
    os.environ.get('SAFEEAR_DATA_ROOT', os.path.join('app', 'data')),
    'cohort', 'cohort_embeddings.npy')
_cohort = None
_cohort_load_attempted = False

def load_cohort_embeddings():
    global _cohort, _cohort_load_attempted
    if _cohort_load_attempted:
        return _cohort
    _cohort_load_attempted = True
    try:
        arr = np.load(_COHORT_PATH).astype(np.float32)
        if arr.ndim != 2:
            logger.error('COHORT_LOAD_FAILED')
            return None
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        _cohort = arr / norms
        logger.info('COHORT_LOADED n=%d dim=%d', _cohort.shape[0], _cohort.shape[1])
    except FileNotFoundError:
        logger.warning('COHORT_FILE_MISSING: ZT-norm disabled')
        _cohort = None
    except Exception as exc:
        logger.error('COHORT_LOAD_FAILED reason=%s', exc)
        _cohort = None
    return _cohort

def compute_cohort_stats(speaker_emb, cohort=None):
    if cohort is None:
        cohort = load_cohort_embeddings()
    if cohort is None:
        return None, None
    vec = np.asarray(speaker_emb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-10:
        return None, None
    vec = vec / norm
    scores = cohort @ vec
    return float(np.mean(scores)), float(np.std(scores))

def zt_norm(raw_score, cohort_mean, cohort_std):
    if cohort_std < 1e-6:
        return raw_score
    return (raw_score - cohort_mean) / cohort_std
