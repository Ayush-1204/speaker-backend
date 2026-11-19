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


# --------------------------------------------------------------------------- #
# Adaptive symmetric score normalisation (AS-norm), calibrated to cosine scale
# --------------------------------------------------------------------------- #
#
#   z = 0.5 * ((s - mu_e) / sd_e + (s - mu_q) / sd_q)
#
# mu_e, sd_e: mean / std of the top-N cohort scores of the enrolled speaker;
# mu_q, sd_q: the same for the query.  The cohort adapts per comparison, so a
# language whose impostor scores sit higher (similar-sounding speakers) is
# shifted back.  The z-score is mapped onto the cosine scale with
# y = scale * z + offset (fitted on development speakers), so the decision
# thresholds and the rest of the decision engine keep their meaning.

class AsNorm:
    def __init__(self, cohort: np.ndarray, top_n: int = 100, scale: float = 1.0, offset: float = 0.0):
        arr = np.asarray(cohort, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        self.cohort = arr / np.where(norms < 1e-10, 1.0, norms)
        self.top_n = int(max(2, min(top_n, len(self.cohort))))
        self.scale = float(scale)
        self.offset = float(offset)

    def _top_stats(self, scores: np.ndarray) -> Tuple[float, float]:
        top = np.partition(scores, len(scores) - self.top_n)[-self.top_n:]
        return float(np.mean(top)), float(max(np.std(top), 1e-4))

    def enroll_stats(self, refs: np.ndarray) -> Tuple[float, float]:
        """Cohort statistics of an enrolled speaker (rows of `refs` are unit-norm voiceprints).
        Each cohort entry is scored like a one-piece query: best match over the voiceprints."""
        return self._top_stats((self.cohort @ np.asarray(refs, dtype=np.float32).T).max(axis=1))

    def query_stats(self, queries: np.ndarray) -> Tuple[float, float]:
        """Cohort statistics of a query (rows are unit-norm pieces), scored with the
        production combination 0.7 * median + 0.3 * mean(top-2) over pieces."""
        s = np.asarray(queries, dtype=np.float32) @ self.cohort.T  # pieces x cohort
        if s.shape[0] == 1:
            return self._top_stats(s[0])
        top2 = np.sort(s, axis=0)[-min(2, s.shape[0]):]
        return self._top_stats(0.7 * np.median(s, axis=0) + 0.3 * top2.mean(axis=0))

    def normalize(self, raw: float, enroll: Tuple[float, float], query: Tuple[float, float]) -> float:
        z = 0.5 * ((raw - enroll[0]) / enroll[1] + (raw - query[0]) / query[1])
        return self.scale * z + self.offset

    def save(self, folder: str) -> None:
        import json
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "cohort.npy"), self.cohort)
        with open(os.path.join(folder, "asnorm.json"), "w", encoding="utf-8") as fh:
            json.dump({"top_n": self.top_n, "scale": self.scale, "offset": self.offset,
                       "cohort_size": int(len(self.cohort))}, fh, indent=2)

    @classmethod
    def load(cls, folder: str) -> "AsNorm":
        import json
        with open(os.path.join(folder, "asnorm.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cls(np.load(os.path.join(folder, "cohort.npy")), cfg["top_n"], cfg["scale"], cfg["offset"])


_ASNORM_DIR = os.environ.get(
    "SAFEEAR_ASNORM_DIR",
    os.path.join(os.environ.get('SAFEEAR_DATA_ROOT', os.path.join('app', 'data')), 'asnorm'))
_asnorm: Optional[AsNorm] = None
_asnorm_loaded = False


def asnorm_enabled_by_env() -> bool:
    return os.environ.get("SAFEEAR_SCORE_NORM", "off").strip().lower() == "asnorm"


def get_asnorm() -> Optional[AsNorm]:
    """The active AS-norm, or None (raw cosine scoring).  Enabled with
    SAFEEAR_SCORE_NORM=asnorm and a folder holding cohort.npy + asnorm.json."""
    global _asnorm, _asnorm_loaded
    if not _asnorm_loaded:
        _asnorm_loaded = True
        if asnorm_enabled_by_env():
            try:
                _asnorm = AsNorm.load(_ASNORM_DIR)
                logger.info("ASNORM_LOADED dir=%s cohort=%d top_n=%d", _ASNORM_DIR, len(_asnorm.cohort), _asnorm.top_n)
            except Exception as exc:  # fall back to raw scores, never stop scoring
                logger.error("ASNORM_LOAD_FAILED dir=%s reason=%s: using raw cosine scores", _ASNORM_DIR, exc)
                _asnorm = None
    return _asnorm


def set_asnorm(norm: Optional[AsNorm]) -> None:
    """Install (or remove, with None) the active AS-norm, e.g. for evaluation and tests."""
    global _asnorm, _asnorm_loaded
    _asnorm, _asnorm_loaded = norm, True
