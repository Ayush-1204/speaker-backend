import asyncio
import threading
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

_WS_LOOP = None
_WS_STATE_LOCK = threading.Lock()
_WS_CLIENTS_BY_PARENT: Dict[str, Dict[str, asyncio.Queue]] = {}
_WS_SEQ_BY_PARENT: Dict[str, int] = {}
_WS_PUBLISHED_COUNTS: Dict[str, int] = {
    "alert_created": 0,
    "device_status_changed": 0,
    "monitoring_changed": 0,
}
_WS_LAST_QUEUE_LAG_MS: int = 0
_DEVICE_STATUS_EVENT_LAST_SENT: Dict[str, int] = {}


def _ws_set_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _WS_LOOP
    with _WS_STATE_LOCK:
        _WS_LOOP = loop


def _ws_next_seq(parent_id: str) -> int:
    with _WS_STATE_LOCK:
        next_seq = _WS_SEQ_BY_PARENT.get(parent_id, 0) + 1
        _WS_SEQ_BY_PARENT[parent_id] = next_seq
        return next_seq


async def _ws_register_client(parent_id: str, client_id: str, queue: asyncio.Queue) -> None:
    with _WS_STATE_LOCK:
        parent_clients = _WS_CLIENTS_BY_PARENT.setdefault(parent_id, {})
        parent_clients[client_id] = queue
        active_count = sum(len(v) for v in _WS_CLIENTS_BY_PARENT.values())
    logger.info("WS_CONNECT | parent_id=%s | client_id=%s | active_clients=%s", parent_id, client_id, active_count)


async def _ws_unregister_client(parent_id: str, client_id: str) -> None:
    with _WS_STATE_LOCK:
        parent_clients = _WS_CLIENTS_BY_PARENT.get(parent_id)
        if parent_clients and client_id in parent_clients:
            del parent_clients[client_id]
            if not parent_clients:
                _WS_CLIENTS_BY_PARENT.pop(parent_id, None)
        active_count = sum(len(v) for v in _WS_CLIENTS_BY_PARENT.values())
    logger.info("WS_DISCONNECT | parent_id=%s | client_id=%s | active_clients=%s", parent_id, client_id, active_count)


async def _ws_publish_async(parent_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    global _WS_LAST_QUEUE_LAG_MS
    seq = _ws_next_seq(parent_id)
    event = {
        "type": event_type,
        "payload": payload,
        "timestamp_ms": int(time.time() * 1000),
        "seq": seq,
    }
    queued_at_ms = int(time.time() * 1000)

    with _WS_STATE_LOCK:
        parent_clients = dict(_WS_CLIENTS_BY_PARENT.get(parent_id, {}))
        if event_type in _WS_PUBLISHED_COUNTS:
            _WS_PUBLISHED_COUNTS[event_type] += 1

    delivered = 0
    for _client_id, queue in parent_clients.items():
        try:
            if queue.full():
                try:
                    queue.get_nowait()
                except Exception:
                    pass
            queue.put_nowait({"event": event, "queued_at_ms": queued_at_ms})
            delivered += 1
        except Exception:
            continue

    _WS_LAST_QUEUE_LAG_MS = 0
    logger.info(
        "WS_EVENT_PUBLISHED | parent_id=%s | type=%s | seq=%s | delivered=%s",
        parent_id,
        event_type,
        seq,
        delivered,
    )


def publish_parent_event(parent_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    loop = _WS_LOOP
    if loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(_ws_publish_async(parent_id, event_type, payload), loop)
    except Exception as exc:
        logger.warning("WS_EVENT_PUBLISH_FAILED | parent_id=%s | type=%s | reason=%s", parent_id, event_type, str(exc))


def _should_publish_device_status(parent_id: str, device_id: str, min_interval_ms: int = 800) -> bool:
    key = f"{parent_id}:{device_id}"
    now = int(time.time() * 1000)
    with _WS_STATE_LOCK:
        last = _DEVICE_STATUS_EVENT_LAST_SENT.get(key)
        if last is not None and (now - last) < min_interval_ms:
            return False
        _DEVICE_STATUS_EVENT_LAST_SENT[key] = now
        return True
