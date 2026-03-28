"""Exclusive parent/child device locks (matches SpeakerApp HomeScreen + LockUtils)."""

import threading
from typing import List, Optional

_lock = threading.Lock()
_parent_device: Optional[str] = None
_child_device: Optional[str] = None


def try_acquire_parent(device_id: str) -> bool:
    with _lock:
        global _parent_device
        if _parent_device is None or _parent_device == device_id:
            _parent_device = device_id
            return True
        return False


def try_acquire_child(device_id: str) -> bool:
    with _lock:
        global _child_device
        if _child_device is None or _child_device == device_id:
            _child_device = device_id
            return True
        return False


def release_device(device_id: str) -> List[str]:
    """Clear locks held by this device. Returns roles released: 'parent' and/or 'child'."""
    released: List[str] = []
    with _lock:
        global _parent_device, _child_device
        if _parent_device == device_id:
            _parent_device = None
            released.append("parent")
        if _child_device == device_id:
            _child_device = None
            released.append("child")
    return released
