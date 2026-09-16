import logging
import json
from typing import Optional, Dict, Any

from .session_redis import get_redis_client, save_session_metadata, load_session_metadata, delete_session_metadata
from .session_redis import save_session_buffer, load_session_buffer

logger = logging.getLogger(__name__)


def persist_session(parent_id: str, device_id: str, metadata: Dict[str, Any]) -> None:
    try:
        save_session_metadata(parent_id, device_id, metadata)
    except Exception as exc:
        logger.debug("persist_session_failed: %s", str(exc))


def load_session(parent_id: str, device_id: str) -> Optional[Dict[str, Any]]:
    try:
        return load_session_metadata(parent_id, device_id)
    except Exception as exc:
        logger.debug("load_session_failed: %s", str(exc))
        return None


def delete_session(parent_id: str, device_id: str) -> None:
    try:
        delete_session_metadata(parent_id, device_id)
    except Exception as exc:
        logger.debug("delete_session_failed: %s", str(exc))


def incr_field(parent_id: str, device_id: str, field: str, amount: int = 1) -> Optional[int]:
    client = get_redis_client()
    if client is None:
        return None
    k = f"safeear:session:{parent_id}:{device_id}"
    max_retries = 10
    try:
        # Use JSON stored value; do a transaction to update field atomically
        with client.pipeline() as pipe:
            for _ in range(max_retries):
                try:
                    pipe.watch(k)
                    raw = pipe.get(k)
                    if not raw:
                        # initialize
                        val = {field: amount}
                    else:
                        obj = json.loads(raw)  # type: ignore
                        obj[field] = int(obj.get(field, 0)) + amount
                        val = obj
                    pipe.multi()
                    pipe.set(k, json.dumps(val))
                    pipe.execute()
                    return int(val[field])
                except Exception:
                    pipe.reset()
                    continue
    except Exception as exc:
        logger.debug("incr_field_failed: %s", str(exc))
    return None


def save_buffer(parent_id: str, device_id: str, name: str, arr) -> None:
    try:
        save_session_buffer(parent_id, device_id, name, arr)
    except Exception as exc:
        logger.debug("save_buffer_failed: %s", str(exc))


def load_buffer(parent_id: str, device_id: str, name: str):
    try:
        return load_session_buffer(parent_id, device_id, name)
    except Exception as exc:
        logger.debug("load_buffer_failed: %s", str(exc))
        return None

