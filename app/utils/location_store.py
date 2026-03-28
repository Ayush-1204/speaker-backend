"""Last device location (in-memory; resets on process restart)."""

import threading
import time
from typing import Any, Dict, Optional

_lock = threading.Lock()
_last: Optional[Dict[str, Any]] = None


def save_location(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _last
    with _lock:
        row = {**payload, "server_ts": int(time.time() * 1000)}
        _last = row
        return dict(row)


def get_last_location() -> Optional[Dict[str, Any]]:
    with _lock:
        return dict(_last) if _last else None


def maps_url_from_last_location() -> str:
    """Google Maps link for SpeakerApp ServerAlert.location (URL string)."""
    loc = get_last_location()
    if not loc:
        return "https://www.google.com/maps"
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    if lat is None or lon is None:
        return "https://www.google.com/maps"
    return f"https://www.google.com/maps?q={lat},{lon}"
