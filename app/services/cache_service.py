import logging
import numpy as np
import os
import time
from typing import List

from app.utils import feature_extractor as feature_extractor_module
from app.database.session import SessionLocal
from app.database.models import EnrolledSpeaker

logger = logging.getLogger(__name__)
_CACHE_WARMUP_SUMMARY = {
    "parents_loaded": 0,
    "cached_parents": 0,
    "speakers_loaded": 0,
    "embeddings_loaded": 0,
    "failed_files": [],
    "parents_with_zero_embeddings": [],
}


def warmup_embeddings_to_redis(redis_client) -> None:
    """Warm up embeddings into a redis client if provided. Keeps a summary in module state."""
    global _CACHE_WARMUP_SUMMARY
    failed_files: List[str] = []
    parents_with_zero_embeddings: List[str] = []
    parents_loaded = 0
    cached_parents = 0
    speakers_loaded = 0
    embeddings_loaded = 0

    client = redis_client
    if client is None:
        _CACHE_WARMUP_SUMMARY = {
            "parents_loaded": 0,
            "cached_parents": 0,
            "speakers_loaded": 0,
            "embeddings_loaded": 0,
            "failed_files": ["redis_unavailable"],
            "parents_with_zero_embeddings": [],
        }
        logger.info("cache_warmup_complete: redis_unavailable")
        return

    with SessionLocal() as db:
        parents = db.query(EnrolledSpeaker).all()
        parents_loaded = len(parents)

        # iterate parents and store embeddings into redis
        for parent in parents:
            parent_id = str(parent.id)
            parent_embedding_count = 0

            # simplified: look in feature_extractor directories similar to previous logic
            speaker_dir = os.path.join(os.environ.get("SAFEEAR_DATA_ROOT", "app/data"), "tenants", parent_id, "embeddings")
            if not os.path.isdir(speaker_dir):
                failed_files.append(f"{speaker_dir}:missing_dir")
                logger.warning("CACHE_WARMUP_FILE_SKIPPED | file=%s | reason=missing_dir", speaker_dir)
                continue

            for speaker_id in os.listdir(speaker_dir):
                sp_dir = os.path.join(speaker_dir, speaker_id)
                if not os.path.isdir(sp_dir):
                    continue
                speaker_embeddings = 0
                for fname in sorted(os.listdir(sp_dir)):
                    if not fname.endswith(".npy"):
                        continue
                    fpath = os.path.join(sp_dir, fname)
                    try:
                        arr = np.asarray(np.load(fpath), dtype=np.float32).reshape(-1)
                    except Exception as exc:
                        failed_files.append(f"{fpath}:load_failed")
                        logger.warning("CACHE_WARMUP_FILE_SKIPPED | file=%s | reason=load_failed | error=%s", fpath, str(exc))
                        continue

                    if arr.shape[0] != feature_extractor_module.config.EMBEDDING_DIM:
                        failed_files.append(f"{fpath}:invalid_dim_{arr.shape[0]}")
                        logger.warning("CACHE_WARMUP_FILE_SKIPPED | file=%s | reason=invalid_dim | dim=%s", fpath, arr.shape[0])
                        continue

                    key = f"safeear:emb:{parent_id}:{speaker_id}:{fname}"
                    client.set(key, arr.astype(np.float32).tobytes())
                    embeddings_loaded += 1
                    speaker_embeddings += 1

                if speaker_embeddings > 0:
                    speakers_loaded += 1
                    parent_embedding_count += speaker_embeddings

            if parent_embedding_count == 0:
                parents_with_zero_embeddings.append(parent_id)
            if parent_embedding_count > 0:
                cached_parents += 1

    _CACHE_WARMUP_SUMMARY = {
        "parents_loaded": parents_loaded,
        "cached_parents": cached_parents,
        "speakers_loaded": speakers_loaded,
        "embeddings_loaded": embeddings_loaded,
        "failed_files": failed_files,
        "parents_with_zero_embeddings": parents_with_zero_embeddings,
    }
    logger.info("cache_warmup_complete: %s", _CACHE_WARMUP_SUMMARY)


_UNAVAILABLE_UNTIL = 0.0


def get_redis_client():
    try:
        import redis as _redis
    except Exception:
        return None

    redis_url = (
        (os.environ.get("REDIS_URL") or "").strip()
        or (os.environ.get("SAFEEAR_REDIS_URL") or "").strip()
        or "redis://localhost:6379/0"
    )
    global _UNAVAILABLE_UNTIL
    if time.monotonic() < _UNAVAILABLE_UNTIL:
        return None
    try:
        client = _redis.Redis.from_url(redis_url, decode_responses=False, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        return client
    except Exception as exc:
        # Skip Redis for a while instead of paying the connect timeout on every call.
        _UNAVAILABLE_UNTIL = time.monotonic() + float(os.environ.get("SAFEEAR_REDIS_RETRY_SEC", "60"))
        logger.warning("CACHE_WARMUP_SKIPPED | reason=redis_unavailable | error=%s", str(exc))
        return None
