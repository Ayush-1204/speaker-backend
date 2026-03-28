"""
Parent alert queue + mode flags.
get_alerts returns JSON array items: { timestamp, location, audio_url } per SpeakerApp ServerAlert.
since = last client timestamp (ms); returns items with timestamp > since.
"""

import os
import shutil
import threading
import time
import uuid
from typing import Any, Dict, List

from .location_store import maps_url_from_last_location

_lock = threading.Lock()
_alerts: List[Dict[str, Any]] = []
_parent_armed = False
_child_armed = False
_MAX = 500

ALERT_AUDIO_SUBDIR = os.path.join("app", "data", "alert_audio")


def ensure_alert_audio_dir() -> str:
    os.makedirs(ALERT_AUDIO_SUBDIR, exist_ok=True)
    return ALERT_AUDIO_SUBDIR


def set_parent_mode_armed(armed: bool) -> None:
    global _parent_armed
    with _lock:
        _parent_armed = armed


def set_child_mode_armed(armed: bool) -> None:
    global _child_armed
    with _lock:
        _child_armed = armed


def get_mode_flags() -> Dict[str, bool]:
    with _lock:
        return {"parent_mode": _parent_armed, "child_mode": _child_armed}


def append_stranger_alert_from_wav(temp_audio_path: str) -> None:
    """Copy clip to static dir and queue for parent poll (only if parent mode armed)."""
    global _alerts
    with _lock:
        if not _parent_armed:
            return
        ensure_alert_audio_dir()
        fname = f"{uuid.uuid4().hex}.wav"
        dest = os.path.join(ALERT_AUDIO_SUBDIR, fname)
        try:
            shutil.copy2(temp_audio_path, dest)
        except OSError:
            return
        ts = int(time.time() * 1000)
        row = {
            "timestamp": ts,
            "location": maps_url_from_last_location(),
            "audio_url": f"/alert_audio/{fname}",
        }
        _alerts.append(row)
        while len(_alerts) > _MAX:
            _alerts.pop(0)


def get_server_alerts_since(since_ms: int) -> List[Dict[str, Any]]:
    """SpeakerApp expects a raw JSON array of ServerAlert."""
    with _lock:
        return [
            {"timestamp": int(a["timestamp"]), "location": str(a["location"]), "audio_url": str(a["audio_url"])}
            for a in _alerts
            if int(a["timestamp"]) > int(since_ms)
        ]
