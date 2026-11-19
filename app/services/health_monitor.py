"""Tell the parent when protection is silently degraded.

Two conditions are reported to the parent's phone (push + websocket event):
  * a monitored device has sent no audio for SAFEEAR_DEVICE_SILENT_SEC while
    monitoring is on (battery dead, app killed, no network, mic blocked);
  * the server cannot score voices (speaker model unavailable).
Each condition is reported once per episode and at most once per throttle window.
"""

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.database.models import Device, DeviceRole

logger = logging.getLogger(__name__)

DEVICE_SILENT_SEC = int(os.environ.get("SAFEEAR_DEVICE_SILENT_SEC", "300"))
# Devices silent for longer than this are treated as retired, not as an outage.
DEVICE_STALE_SEC = int(os.environ.get("SAFEEAR_DEVICE_STALE_SEC", str(24 * 3600)))
DEGRADED_THROTTLE_SEC = int(os.environ.get("SAFEEAR_DEGRADED_THROTTLE_SEC", "3600"))

_LOCK = threading.Lock()
# device_id -> last_activity value already reported (one push per silence episode)
_REPORTED_SILENCE: Dict[str, datetime] = {}
# (parent_id, reason) -> monotonic time of last push
_LAST_DEGRADED_PUSH: Dict[Tuple[str, str], float] = {}


def _naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def find_newly_silent_devices(db: Session, now: Optional[datetime] = None) -> List[Device]:
    """Monitored child devices that went quiet and have not been reported yet."""
    now = _naive_utc(now or datetime.now(timezone.utc))
    silent_before = now - timedelta(seconds=DEVICE_SILENT_SEC)
    stale_before = now - timedelta(seconds=DEVICE_STALE_SEC)
    rows = (
        db.query(Device)
        .filter(
            Device.role == DeviceRole.child_device,
            Device.monitoring_enabled.is_(True),
            Device.last_activity_at.isnot(None),
        )
        .all()
    )
    out: List[Device] = []
    with _LOCK:
        for row in rows:
            last = _naive_utc(row.last_activity_at)
            key = str(row.id)
            if last is None or last > silent_before:
                _REPORTED_SILENCE.pop(key, None)  # active again: re-arm
                continue
            if last < stale_before or _REPORTED_SILENCE.get(key) == last:
                continue
            _REPORTED_SILENCE[key] = last
            out.append(row)
    return out


def _parent_push_token(db: Session, parent_id: str) -> Optional[str]:
    row = (
        db.query(Device)
        .filter(
            Device.parent_id == uuid.UUID(str(parent_id)),
            Device.role == DeviceRole.parent_device,
            Device.device_token.isnot(None),
        )
        .order_by(Device.updated_at.desc())
        .first()
    )
    return row.device_token if row else None


def notify_parent(db: Session, parent_id: str, reason: str, title: str, body: str, extra: Optional[dict] = None) -> bool:
    """Push + websocket event, throttled per (parent, reason)."""
    key = (str(parent_id), reason)
    now = time.monotonic()
    with _LOCK:
        last = _LAST_DEGRADED_PUSH.get(key)
        if last is not None and now - last < DEGRADED_THROTTLE_SEC:
            return False
        _LAST_DEGRADED_PUSH[key] = now

    payload = {"type": "protection_degraded", "reason": reason, "title": title, "body": body, **(extra or {})}
    try:
        from app.services.websocket_manager import publish_parent_event

        publish_parent_event(str(parent_id), "protection_degraded", payload)
    except Exception as exc:
        logger.warning("DEGRADED_WS_PUBLISH_FAILED | parent_id=%s | reason=%s", parent_id, exc)

    token = _parent_push_token(db, parent_id)
    sent = False
    if token:
        from app.utils.notification_worker import send_status_push

        sent = send_status_push(token, title, body, payload)
    logger.warning("PROTECTION_DEGRADED | parent_id=%s | reason=%s | push_sent=%s", parent_id, reason, sent)
    return sent


def report_silent_devices(db: Session) -> int:
    count = 0
    for device in find_newly_silent_devices(db):
        name = device.device_name or "Your child's device"
        minutes = max(1, DEVICE_SILENT_SEC // 60)
        notify_parent(
            db,
            str(device.parent_id),
            reason=f"device_silent:{device.id}",
            title="SafeEar: monitoring interrupted",
            body=f"{name} has not sent audio for {minutes}+ minutes while monitoring is on. "
                 "Check that it is charged, connected and the app is running.",
            extra={"device_id": str(device.id)},
        )
        count += 1
    return count


def report_scoring_unavailable(db: Session, parent_id: str) -> None:
    notify_parent(
        db,
        parent_id,
        reason="scoring_unavailable",
        title="SafeEar: protection degraded",
        body="The server cannot identify voices right now, so stranger alerts are paused. "
             "It will retry automatically.",
    )


def reset_state() -> None:
    with _LOCK:
        _REPORTED_SILENCE.clear()
        _LAST_DEGRADED_PUSH.clear()
