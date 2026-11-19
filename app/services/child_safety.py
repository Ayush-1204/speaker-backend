"""Child-device features: code pairing, parent-nearby pause, SOS alerts.

Database logic only; the HTTP endpoints live in app.main.
"""

import hashlib
import hmac
import logging
import math
import os
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.database.models import Alert, Device, DeviceRole, HouseholdConfig, PairingRequest
from app.utils import prd_services_db as svc
from app.utils.audio_preprocess import normalize_waveform

logger = logging.getLogger(__name__)

PAIRING_TTL_SEC = int(os.environ.get("SAFEEAR_PAIRING_TTL_SEC", "300"))
# Parent-nearby rule: both fixes must be this fresh and this accurate to pause detection.
PROXIMITY_MAX_FIX_AGE_SEC = float(os.environ.get("SAFEEAR_PROXIMITY_MAX_FIX_AGE_SEC", "120"))
PROXIMITY_MAX_ACCURACY_M = float(os.environ.get("SAFEEAR_PROXIMITY_MAX_ACCURACY_M", "30"))
PROXIMITY_CACHE_MS = int(os.environ.get("SAFEEAR_PROXIMITY_CACHE_MS", "15000"))
PROXIMITY_RADIUS_MIN_M = 5.0
PROXIMITY_RADIUS_MAX_M = 100.0
SOS_DEDUPE_SEC = int(os.environ.get("SAFEEAR_SOS_DEDUPE_SEC", "30"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #

def _active_code_exists(db: Session, code: str, now: datetime) -> bool:
    return (
        db.query(PairingRequest)
        .filter(
            PairingRequest.code == code,
            PairingRequest.expires_at > now,
            PairingRequest.consumed_at.is_(None),
        )
        .first()
        is not None
    )


def start_pairing(db: Session, installation_id: str, device_name: Optional[str]) -> Dict[str, Any]:
    installation_id = (installation_id or "").strip()
    if not installation_id:
        raise HTTPException(status_code=422, detail="installation_id_required")
    now = _utcnow()
    # Old unclaimed requests are useless once expired; keep the table small.
    db.query(PairingRequest).filter(PairingRequest.expires_at < now - timedelta(days=1)).delete(
        synchronize_session=False
    )
    for _ in range(20):
        code = f"{secrets.randbelow(1_000_000):06d}"
        if not _active_code_exists(db, code, now):
            break
    else:  # pragma: no cover - a million codes would have to be in use
        raise HTTPException(status_code=503, detail="pairing_busy")
    secret = secrets.token_urlsafe(32)
    req = PairingRequest(
        id=uuid.uuid4(),
        code=code,
        secret_hash=_hash_secret(secret),
        installation_id=installation_id,
        device_name=(device_name or "").strip() or None,
        expires_at=now + timedelta(seconds=PAIRING_TTL_SEC),
    )
    db.add(req)
    db.commit()
    logger.info("PAIRING_STARTED | pair_id=%s", req.id)
    return {"pair_id": str(req.id), "pair_secret": secret, "code": code, "expires_in": PAIRING_TTL_SEC}


def claim_pairing(db: Session, parent_id: str, code: str) -> Device:
    code = (code or "").strip().replace(" ", "")
    now = _utcnow()
    req = (
        db.query(PairingRequest)
        .filter(PairingRequest.code == code, PairingRequest.expires_at > now, PairingRequest.consumed_at.is_(None))
        .order_by(PairingRequest.created_at.desc())
        .first()
    )
    if req is None:
        raise HTTPException(status_code=404, detail="pairing_code_invalid_or_expired")
    if req.claimed_parent_id is not None:
        raise HTTPException(status_code=409, detail="pairing_code_already_used")
    name = req.device_name or f"Watch {req.installation_id[:6]}"
    device = svc.create_device(db, parent_id, name, DeviceRole.child_device, installation_id=req.installation_id)
    req.claimed_parent_id = device.parent_id
    req.device_id = device.id
    db.commit()
    logger.info("PAIRING_CLAIMED | pair_id=%s | parent_id=%s | device_id=%s", req.id, parent_id, device.id)
    return device


def pairing_status(db: Session, pair_id: str, pair_secret: str) -> Tuple[str, Optional[Device]]:
    """Returns ("pending", None) or ("claimed", device); claimed is returned only once."""
    try:
        pid = uuid.UUID(str(pair_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="pairing_not_found")
    req = db.query(PairingRequest).filter(PairingRequest.id == pid).first()
    if req is None or not hmac.compare_digest(req.secret_hash, _hash_secret(pair_secret or "")):
        raise HTTPException(status_code=404, detail="pairing_not_found")
    if req.consumed_at is not None:
        raise HTTPException(status_code=410, detail="pairing_already_completed")
    if req.claimed_parent_id is None:
        if _naive_utc(req.expires_at) <= _utcnow():
            raise HTTPException(status_code=410, detail="pairing_code_expired")
        return "pending", None
    device = db.query(Device).filter(Device.id == req.device_id).first()
    if device is None:
        raise HTTPException(status_code=410, detail="pairing_device_removed")
    req.consumed_at = _utcnow()
    db.commit()
    return "claimed", device


# --------------------------------------------------------------------------- #
# Household settings and parent location
# --------------------------------------------------------------------------- #

def get_or_create_config(db: Session, parent_id: str) -> HouseholdConfig:
    parent_uuid = uuid.UUID(str(parent_id))
    cfg = db.query(HouseholdConfig).filter(HouseholdConfig.parent_id == parent_uuid).first()
    if cfg is None:
        cfg = HouseholdConfig(id=uuid.uuid4(), parent_id=parent_uuid)
        db.add(cfg)
        db.commit()
        db.refresh(cfg)
    return cfg


def config_payload(cfg: Optional[HouseholdConfig]) -> Dict[str, Any]:
    return {
        "parent_proximity_enabled": bool(cfg.parent_proximity_enabled) if cfg else False,
        "parent_proximity_radius_m": float(cfg.parent_proximity_radius_m or 10.0) if cfg else 10.0,
    }


def read_config(db: Session, parent_id: str) -> Dict[str, Any]:
    cfg = db.query(HouseholdConfig).filter(HouseholdConfig.parent_id == uuid.UUID(str(parent_id))).first()
    return config_payload(cfg)


def update_config(db: Session, parent_id: str, enabled: Optional[bool], radius_m: Optional[float]) -> Dict[str, Any]:
    cfg = get_or_create_config(db, parent_id)
    if enabled is not None:
        cfg.parent_proximity_enabled = bool(enabled)
    if radius_m is not None:
        if not (PROXIMITY_RADIUS_MIN_M <= float(radius_m) <= PROXIMITY_RADIUS_MAX_M):
            raise HTTPException(status_code=422, detail="parent_proximity_radius_out_of_range")
        cfg.parent_proximity_radius_m = float(radius_m)
    db.commit()
    db.refresh(cfg)
    logger.info("HOUSEHOLD_CONFIG_UPDATED | parent_id=%s | %s", parent_id, config_payload(cfg))
    return config_payload(cfg)


def _fix_time(location_ts_ms: Optional[int]) -> datetime:
    now = _utcnow()
    if location_ts_ms is None:
        return now
    ts = datetime.fromtimestamp(location_ts_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)
    # A device clock ahead of the server must not make a fix look fresh for longer.
    return min(ts, now)


def store_fix(
    db: Session,
    device: Device,
    lat: float,
    lon: float,
    accuracy_m: Optional[float],
    location_ts_ms: Optional[int] = None,
) -> Device:
    device.last_location_lat = float(lat)
    device.last_location_lon = float(lon)
    device.last_location_accuracy_m = float(accuracy_m) if accuracy_m is not None else None
    device.last_location_ts = _fix_time(location_ts_ms)
    db.commit()
    db.refresh(device)
    return device


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _usable_fix(device: Optional[Device], now: datetime) -> bool:
    if device is None or device.last_location_lat is None or device.last_location_lon is None:
        return False
    ts = _naive_utc(device.last_location_ts)
    if ts is None or (now - ts).total_seconds() > PROXIMITY_MAX_FIX_AGE_SEC:
        return False
    acc = device.last_location_accuracy_m
    return acc is not None and acc <= PROXIMITY_MAX_ACCURACY_M


def parent_nearby(db: Session, parent_id: str, child: Device) -> Dict[str, Any]:
    """Is the child device within the configured radius of a parent phone?

    Fail-safe: anything missing, stale or inaccurate means "not nearby", so
    detection keeps running.
    """
    parent_uuid = uuid.UUID(str(parent_id))
    cfg = db.query(HouseholdConfig).filter(HouseholdConfig.parent_id == parent_uuid).first()
    if cfg is None or not cfg.parent_proximity_enabled:
        return {"near": False, "reason": "rule_off"}
    now = _utcnow()
    if not _usable_fix(child, now):
        return {"near": False, "reason": "child_fix_unusable"}
    radius = float(cfg.parent_proximity_radius_m or 10.0)
    parents = (
        db.query(Device)
        .filter(
            Device.parent_id == parent_uuid,
            Device.role == DeviceRole.parent_device,
            Device.last_location_ts.isnot(None),
        )
        .all()
    )
    best = None
    for p in parents:
        if not _usable_fix(p, now):
            continue
        d = haversine_m(child.last_location_lat, child.last_location_lon, p.last_location_lat, p.last_location_lon)
        if best is None or d < best:
            best = d
    if best is None:
        return {"near": False, "reason": "parent_fix_unusable"}
    return {"near": best <= radius, "reason": "distance", "distance_m": round(best, 1), "radius_m": radius}


def check_parent_nearby_cached(db: Session, parent_id: str, child: Device) -> bool:
    """parent_nearby() evaluated at most once per PROXIMITY_CACHE_MS per device stream."""
    session = svc.get_or_create_session(parent_id, str(child.id))
    now = svc.now_ms()
    if now - session.proximity_checked_ms < PROXIMITY_CACHE_MS:
        return session.proximity_near
    result = parent_nearby(db, parent_id, child)
    was_near = session.proximity_near
    session.proximity_near = bool(result["near"])
    session.proximity_checked_ms = now
    if session.proximity_near and not was_near:
        # Audio before and after the pause must not be stitched into one streak.
        svc.reset_stream(session)
        logger.info("PARENT_NEARBY_PAUSE | device_id=%s | %s", child.id, result)
    elif was_near and not session.proximity_near:
        logger.info("PARENT_NEARBY_RESUME | device_id=%s | %s", child.id, result)
    return session.proximity_near


# --------------------------------------------------------------------------- #
# SOS
# --------------------------------------------------------------------------- #

def recent_sos(db: Session, device: Device) -> Optional[Alert]:
    cutoff = _utcnow() - timedelta(seconds=SOS_DEDUPE_SEC)
    row = (
        db.query(Alert)
        .filter(Alert.device_id == device.id, Alert.alert_type == "sos")
        .order_by(Alert.timestamp.desc())
        .first()
    )
    if row is not None and _naive_utc(row.timestamp) >= cutoff:
        return row
    return None


def save_sos_clip(parent_id: str, alert_id: str, waveform: np.ndarray, sr: int) -> str:
    adir = os.path.join(svc.TENANTS_ROOT, str(parent_id), "alerts", str(alert_id))
    os.makedirs(adir, exist_ok=True)
    path = os.path.join(adir, "clip.wav")
    tmp = f"{path}.tmp"
    wave = normalize_waveform(np.asarray(waveform, dtype=np.float32).reshape(-1))
    sf.write(tmp, wave, sr, format="WAV")
    os.replace(tmp, path)
    return path


def parent_push_token(db: Session, parent_uuid: uuid.UUID) -> Optional[str]:
    row = (
        db.query(Device)
        .filter(
            Device.parent_id == parent_uuid,
            Device.role == DeviceRole.parent_device,
            Device.device_token.isnot(None),
        )
        .order_by(Device.updated_at.desc())
        .first()
    )
    return row.device_token if row else None


def run_in_background(fn, *args, **kwargs) -> None:
    threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
