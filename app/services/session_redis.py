import os
import json
import threading
import time
import logging
from typing import Optional, Dict, Any
import base64
import numpy as np

try:
    import redis
except Exception:
    redis = None

logger = logging.getLogger(__name__)


def _resolve_redis_url() -> str:
    return (
        (os.environ.get("REDIS_URL") or "").strip()
        or (os.environ.get("SAFEEAR_REDIS_URL") or "").strip()
        or "redis://localhost:6379/0"
    )


_UNAVAILABLE_UNTIL = 0.0
_RETRY_AFTER_SEC = float(os.environ.get("SAFEEAR_REDIS_RETRY_SEC", "60"))
_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def get_redis_client():
    """Shared Redis client (redis-py pools connections and is thread-safe), or None.

    Created and pinged once; after a failure Redis is skipped for
    SAFEEAR_REDIS_RETRY_SEC instead of paying the connect timeout on every call."""
    global _CLIENT
    if redis is None or time.monotonic() < _UNAVAILABLE_UNTIL:
        return None
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        try:
            url = _resolve_redis_url()
            client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
            client.ping()
            _CLIENT = client
            return client
        except Exception as exc:
            mark_redis_unavailable(exc)
            return None


def mark_redis_unavailable(exc: Exception) -> None:
    global _UNAVAILABLE_UNTIL, _CLIENT
    _CLIENT = None
    _UNAVAILABLE_UNTIL = time.monotonic() + _RETRY_AFTER_SEC
    logger.warning("SESSION_REDIS_UNAVAILABLE | retry_in=%ss | reason=%s", int(_RETRY_AFTER_SEC), str(exc))


def _key(parent_id: str, device_id: str) -> str:
    return f"safeear:session:{parent_id}:{device_id}"


def save_session_metadata(parent_id: str, device_id: str, payload: Dict[str, Any]) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        # One round trip: value and expiry together.
        client.set(_key(parent_id, device_id), json.dumps(payload),
                   ex=int(os.environ.get("SAFEEAR_SESSION_TTL_SEC", "900")))
    except Exception as exc:
        mark_redis_unavailable(exc)


def load_session_metadata(parent_id: str, device_id: str) -> Optional[Dict[str, Any]]:
    client = get_redis_client()
    if client is None:
        return None
    try:
        k = _key(parent_id, device_id)
        raw = client.get(k)
        if not raw:
            return None
        return json.loads(raw)  # type: ignore
    except Exception as exc:
        logger.debug("SESSION_REDIS_LOAD_FAILED: %s", str(exc))
        return None


def delete_session_metadata(parent_id: str, device_id: str) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        client.delete(_key(parent_id, device_id))
    except Exception:
        pass


def _buf_key(parent_id: str, device_id: str, name: str) -> str:
    return f"safeear:session:{parent_id}:{device_id}:buf:{name}"


def save_session_buffer(parent_id: str, device_id: str, name: str, arr: Optional[np.ndarray]) -> None:
    """Store a float32 numpy array as base64-encoded bytes in Redis."""
    client = get_redis_client()
    if client is None:
        return
    try:
        k = _buf_key(parent_id, device_id, name)
        if arr is None:
            client.delete(k)
            return
        raw = np.asarray(arr, dtype=np.float32).tobytes()
        client.set(k, base64.b64encode(raw).decode("ascii"))
        client.expire(k, int(os.environ.get("SAFEEAR_SESSION_TTL_SEC", "300")))
    except Exception as exc:
        logger.debug("SESSION_REDIS_SAVE_BUFFER_FAILED: %s", str(exc))


def load_session_buffer(parent_id: str, device_id: str, name: str) -> Optional[np.ndarray]:
    client = get_redis_client()
    if client is None:
        return None
    try:
        k = _buf_key(parent_id, device_id, name)
        raw = client.get(k)
        if not raw:
            return None
        data = base64.b64decode(raw)  # type: ignore
        arr = np.frombuffer(data, dtype=np.float32)
        return arr
    except Exception as exc:
        logger.debug("SESSION_REDIS_LOAD_BUFFER_FAILED: %s", str(exc))
        return None
