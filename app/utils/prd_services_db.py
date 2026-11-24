"""PRD services with SQLAlchemy database backing and device role support."""

import logging
import os
import secrets
import shutil
import threading
import time
import uuid
import json
import hashlib
import hmac
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import (
    Parent,
    Device,
    DeviceRole,
    EnrolledSpeaker,
    SpeakerRole,
    Alert as AlertModel,
    AuthCredential,
    AuthProvider,
    HouseholdConfig,
)
from app.utils import config
from app.utils.audio_preprocess import normalize_waveform
from app.utils.feature_extractor import embed_waveform_chunks
from app.utils.speech_gate import assess_speech_likeness
from app.utils.vad import get_speech_segments
from app.services.session_manager import persist_session, delete_session, load_session
from app.utils.yamnet_classifier import classify_audio_event_windowed, distress_from_result

DATA_ROOT = (os.environ.get("SAFEEAR_DATA_ROOT", os.path.join("app", "data")) or "").strip() or os.path.join("app", "data")
TENANTS_ROOT = os.path.join(DATA_ROOT, "tenants")

JWT_SECRET = os.environ.get("SAFEEAR_JWT_SECRET", "change-me-in-production")
JWT_EXPIRE_SEC = int(os.environ.get("SAFEEAR_JWT_EXPIRE_SEC", "604800"))

WINDOW_SEC = float(os.environ.get("SAFEEAR_WINDOW_SEC", "1.5"))
HOP_SEC = float(os.environ.get("SAFEEAR_HOP_SEC", "0.25"))
ALERT_RING_SEC = float(os.environ.get("SAFEEAR_ALERT_RING_SEC", "10.0"))
STRANGER_PREROLL_SEC = float(os.environ.get("SAFEEAR_STRANGER_PREROLL_SEC", "5.0"))
# Audio kept recording after an alert fires and appended to its clip, so the parent
# hears the stranger keep talking, not only the seconds that triggered the alert.
ALERT_POST_ROLL_SEC = float(os.environ.get("SAFEEAR_ALERT_POST_ROLL_SEC", "5.0"))
_REDIMNET_HUB_REPO = os.environ.get("REDIMNET_HUB_REPO", "PalabraAI/redimnet2")
_REDIMNET_HUB_ENTRY = os.environ.get("REDIMNET_HUB_ENTRY", "redimnet2")
_IS_REDIMNET2 = ("redimnet2" in _REDIMNET_HUB_REPO.lower()) or ("redimnet2" in _REDIMNET_HUB_ENTRY.lower())

_DEFAULT_T_HIGH = "0.47" if _IS_REDIMNET2 else "0.72"
# 0.39 (was 0.33) with a streak of 3: stricter stranger detection, 2026-11-24.
# Replay of the eval sessions (T_H 0.47, T_L 0.39, K 3): 92.9 % of strangers detected,
# 3 false alerts in 5.6 family hours (results/P8_false_alert_rates.csv).
_DEFAULT_T_LOW = "0.39" if _IS_REDIMNET2 else "0.60"
_DEFAULT_CONFIRM_WINDOWS = "3"

T_HIGH = float(os.environ.get("SAFEEAR_T_HIGH", _DEFAULT_T_HIGH))
T_LOW = float(os.environ.get("SAFEEAR_T_LOW", _DEFAULT_T_LOW))
STRANGER_CONFIRM_COUNT = int(os.environ.get("SAFEEAR_CONFIRM_WINDOWS", _DEFAULT_CONFIRM_WINDOWS))
DEBOUNCE_SEC = int(os.environ.get("SAFEEAR_DEBOUNCE_SEC", "60"))
ENABLE_DISTRESS_DETECTION = os.environ.get("SAFEEAR_ENABLE_DISTRESS_ALERTS", "false").strip().lower() in {"1", "true", "yes", "on"}
# "concat" (default): one embedding of the window's voiced speech, matching enrollment.
# "segments": legacy per-VAD-segment embeddings, kept only for A/B comparison.
INFERENCE_EMBED_MODE = os.environ.get("SAFEEAR_INFERENCE_EMBED", "concat").strip().lower()

logger = logging.getLogger(__name__)
DEVICE_ONLINE_TTL_SEC = int(os.environ.get("SAFEEAR_DEVICE_ONLINE_TTL_SEC", "120"))

_LOCK = threading.Lock()
_SESSIONS: Dict[str, "SessionState"] = {}
_INFER_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("SAFEEAR_INFER_WORKERS", "2")))

# Track empty chunks per device to detect audio capture failures
_EMPTY_CHUNK_COUNTS: Dict[str, Tuple[int, float]] = {}  # device_id -> (count, last_reset_time)
_EMPTY_CHUNK_THRESHOLD = int(os.environ.get("SAFEEAR_EMPTY_CHUNK_THRESHOLD", "5"))
_EMPTY_CHUNK_RESET_SEC = int(os.environ.get("SAFEEAR_EMPTY_CHUNK_RESET_SEC", "10"))


def _normalized_embedding(vec: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return None
    return (arr / norm).astype(np.float32)


def _compute_query_centroid(query_embs: List[np.ndarray]) -> Optional[np.ndarray]:
    normalized_chunks: List[np.ndarray] = []
    for emb in query_embs or []:
        normed = _normalized_embedding(emb)
        if normed is not None and normed.shape[0] == config.EMBEDDING_DIM:
            normalized_chunks.append(normed)

    if not normalized_chunks:
        return None

    centroid = np.mean(np.stack(normalized_chunks, axis=0), axis=0)
    return _normalized_embedding(centroid)


def _as_uuid(value: Any, field_name: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid_{field_name}") from exc


def _ensure_parent_dirs(parent_id: Union[str, uuid.UUID]) -> Dict[str, str]:
    base = os.path.join(TENANTS_ROOT, str(parent_id))
    embeddings = os.path.join(base, "embeddings")
    alerts = os.path.join(base, "alerts")
    os.makedirs(embeddings, exist_ok=True)
    os.makedirs(alerts, exist_ok=True)
    return {"base": base, "embeddings": embeddings, "alerts": alerts}


def now_ms() -> int:
    return int(time.time() * 1000)


def track_empty_chunk(device_id: str) -> Tuple[int, bool]:
    """Track consecutive empty chunks per device to detect audio capture failures.
    
    Args:
        device_id: The device sending empty chunks
    
    Returns:
        Tuple of (count, exceeded_threshold)
        - count: Current consecutive empty chunk count
        - exceeded_threshold: True if count >= SAFEEAR_EMPTY_CHUNK_THRESHOLD
    
    Resets count after SAFEEAR_EMPTY_CHUNK_RESET_SEC (default 10s) of inactivity.
    Helps diagnose audio capture failures on child device.
    """
    now_sec = time.time()
    with _LOCK:
        if device_id not in _EMPTY_CHUNK_COUNTS:
            _EMPTY_CHUNK_COUNTS[device_id] = (1, now_sec)
            return 1, False
        
        count, last_reset = _EMPTY_CHUNK_COUNTS[device_id]
        if now_sec - last_reset > _EMPTY_CHUNK_RESET_SEC:
            # Reset counter after timeout
            _EMPTY_CHUNK_COUNTS[device_id] = (1, now_sec)
            return 1, False
        
        count += 1
        _EMPTY_CHUNK_COUNTS[device_id] = (count, now_sec)
        return count, count >= _EMPTY_CHUNK_THRESHOLD


def _perf_metrics(waveform_len: int, sr: int, started_at: float) -> Dict[str, float]:
    elapsed_sec = max(0.0, time.perf_counter() - started_at)
    audio_sec = max(1e-6, float(waveform_len) / float(max(1, sr)))
    return {
        "processing_ms": round(elapsed_sec * 1000.0, 1),
        "audio_ms": round(audio_sec * 1000.0, 1),
        "rtf": round(elapsed_sec / audio_sec, 4),
    }


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64d(data: str) -> bytes:
    pad = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + pad)


def make_jwt(parent_id: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(parent_id),
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRE_SEC,
        "jti": secrets.token_urlsafe(8),
    }
    h = _b64(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(JWT_SECRET.encode("utf-8"), f"{h}.{p}".encode("ascii"), hashlib.sha256).digest()
    s = _b64(sig)
    return f"{h}.{p}.{s}"


def parse_jwt(token: str) -> Dict[str, Any]:
    try:
        h, p, s = token.split(".")
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid_token_format") from exc
    expected = _b64(hmac.new(JWT_SECRET.encode("utf-8"), f"{h}.{p}".encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, s):
        raise HTTPException(status_code=401, detail="invalid_token_signature")
    payload = json.loads(_b64d(p).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="token_expired")
    return payload


def upsert_parent(db: Session, google_sub: str, email: Optional[str], display_name: Optional[str]) -> Parent:
    parent = db.query(Parent).filter(Parent.google_sub == google_sub).first()
    if parent is None:
        parent = Parent(
            id=uuid.uuid4(),
            google_sub=google_sub,
            email=email,
            display_name=display_name
        )
        db.add(parent)
    else:
        parent.email = email or parent.email
        parent.display_name = display_name or parent.display_name
        parent.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(parent)
    _ensure_parent_dirs(str(parent.id))
    return parent


def _normalize_email(email: str) -> str:
    v = (email or "").strip().lower()
    if not v or "@" not in v:
        raise HTTPException(status_code=422, detail="invalid_email")
    return v


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if password is None or len(password) < 8:
        raise HTTPException(status_code=422, detail="password_too_short")
    if salt is None:
        salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return f"pbkdf2_sha256$120000${base64.b64encode(salt).decode('ascii')}${base64.b64encode(derived).decode('ascii')}"


def _verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algo, rounds_s, salt_b64, digest_b64 = encoded_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        rounds = int(rounds_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
    except Exception:
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(actual, expected)


def register_parent_with_email(
    db: Session, email: str, password: str, display_name: Optional[str] = None
) -> Parent:
    norm_email = _normalize_email(email)
    existing_cred = db.query(AuthCredential).filter(AuthCredential.email == norm_email).first()
    if existing_cred is not None:
        raise HTTPException(status_code=409, detail="email_already_registered")

    parent = Parent(
        id=uuid.uuid4(),
        # Keep existing schema intact by using deterministic non-google subject for email users.
        google_sub=f"email:{norm_email}",
        email=norm_email,
        display_name=display_name,
    )
    db.add(parent)
    db.flush()

    cred = AuthCredential(
        id=uuid.uuid4(),
        parent_id=parent.id,
        provider=AuthProvider.email_password,
        email=norm_email,
        password_hash=_hash_password(password),
    )
    db.add(cred)
    db.commit()
    db.refresh(parent)
    _ensure_parent_dirs(str(parent.id))
    return parent


_DUMMY_PASSWORD_HASH: Optional[str] = None


def login_parent_with_email(db: Session, email: str, password: str) -> Parent:
    global _DUMMY_PASSWORD_HASH
    norm_email = _normalize_email(email)
    cred = db.query(AuthCredential).filter(AuthCredential.email == norm_email).first()
    if cred is None:
        # Spend the same hashing time as a real check so response timing does
        # not reveal which addresses are registered.
        if _DUMMY_PASSWORD_HASH is None:
            _DUMMY_PASSWORD_HASH = _hash_password(secrets.token_urlsafe(16))
        _verify_password(password, _DUMMY_PASSWORD_HASH)
        raise HTTPException(status_code=401, detail="invalid_email_or_password")
    if not _verify_password(password, str(getattr(cred, "password_hash"))):
        raise HTTPException(status_code=401, detail="invalid_email_or_password")

    parent = db.query(Parent).filter(Parent.id == cred.parent_id).first()
    if parent is None:
        raise HTTPException(status_code=401, detail="parent_not_found")
    return parent


def create_password_reset_token(db: Session, email: str, ttl_sec: int = 3600) -> Optional[str]:
    norm_email = _normalize_email(email)
    cred = db.query(AuthCredential).filter(AuthCredential.email == norm_email).first()
    if cred is None:
        # Do not reveal user existence.
        return None

    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    cred.reset_token_hash = token_hash
    cred.reset_token_expires_at = datetime.fromtimestamp(time.time() + ttl_sec, timezone.utc)
    cred.updated_at = datetime.now(timezone.utc)
    db.commit()
    return raw


def reset_password_with_token(db: Session, token: str, new_password: str) -> str:
    """Set a new password; returns the parent id so callers can end old sessions."""
    token_hash = hashlib.sha256((token or "").encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    cred = (
        db.query(AuthCredential)
        .filter(
            AuthCredential.reset_token_hash == token_hash,
            AuthCredential.reset_token_expires_at.isnot(None),
            AuthCredential.reset_token_expires_at >= now,
        )
        .first()
    )
    if cred is None:
        raise HTTPException(status_code=400, detail="invalid_or_expired_reset_token")

    cred.password_hash = _hash_password(new_password)
    cred.reset_token_hash = None
    cred.reset_token_expires_at = None
    cred.updated_at = datetime.now(timezone.utc)
    db.commit()
    return str(cred.parent_id)


def get_parent(db: Session, parent_id: str) -> Parent:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    parent = db.query(Parent).filter(Parent.id == parent_uuid).first()
    if not parent:
        raise HTTPException(status_code=401, detail="parent_not_found")
    return parent


def _release_fcm_token_elsewhere(db: Session, token: str, keep_device_id: Optional[uuid.UUID]) -> None:
    """FCM tokens identify one app install.  If another row (in any family) still
    holds this token, clear it there so pushes for that row stop reaching this
    phone.  The other row itself is never moved or modified otherwise."""
    q = db.query(Device).filter(Device.device_token == token)
    if keep_device_id is not None:
        q = q.filter(Device.id != keep_device_id)
    for other in q.all():
        other.device_token = None
        other.updated_at = datetime.now(timezone.utc)
    db.flush()


def _apply_role_change(existing: Device, parent_uuid: uuid.UUID, previous_role: DeviceRole, role: DeviceRole) -> None:
    if previous_role != role and previous_role == DeviceRole.child_device and role == DeviceRole.parent_device:
        existing.monitoring_enabled = False
        try:
            stop_session(str(parent_uuid), str(existing.id))
        except Exception:
            logger.warning("DEVICE_ROLE_SWITCH_SESSION_STOP_FAILED | device_id=%s", str(existing.id))
        _send_monitoring_command_to_device(existing, False)
    elif previous_role != role and role == DeviceRole.child_device:
        existing.monitoring_enabled = True
        _send_monitoring_command_to_device(existing, True)


def create_device(
    db: Session,
    parent_id: str,
    device_name: Optional[str],
    role: DeviceRole,
    installation_id: Optional[str] = None,
    device_token: Optional[str] = None,
) -> Device:
    """Register or re-register a device for this parent (upsert by stable identifier).

    Only rows belonging to this parent are ever matched.  Every registration bumps
    access_token_version, which revokes device tokens issued earlier.
    """
    parent_uuid = _as_uuid(parent_id, "parent_id")
    normalized_name = (device_name or "").strip()
    normalized_installation_id = (installation_id or "").strip()
    if not normalized_installation_id and not normalized_name:
        raise HTTPException(status_code=422, detail="stable_device_identifier_required")

    token = (device_token or "").strip() or None

    existing = None
    if normalized_installation_id:
        existing = (
            db.query(Device)
            .filter(Device.parent_id == parent_uuid, Device.installation_id == normalized_installation_id)
            .first()
        )
    if existing is None and normalized_name:
        existing = (
            db.query(Device)
            .filter(
                Device.parent_id == parent_uuid,
                func.lower(Device.device_name) == normalized_name.lower(),
            )
            .first()
        )
    if existing is None and token is not None:
        existing = (
            db.query(Device)
            .filter(Device.parent_id == parent_uuid, Device.device_token == token)
            .first()
        )

    if existing is not None:
        previous_role = existing.role
        if normalized_installation_id:
            existing.installation_id = normalized_installation_id
        if normalized_name:
            existing.device_name = normalized_name
        existing.role = role
        if token is not None and existing.device_token != token:
            _release_fcm_token_elsewhere(db, token, existing.id)
            existing.device_token = token
        _apply_role_change(existing, parent_uuid, previous_role, role)
        existing.access_token_version = int(existing.access_token_version or 0) + 1
        existing.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing)
        return existing

    if token is not None:
        _release_fcm_token_elsewhere(db, token, None)

    device = Device(
        id=uuid.uuid4(),
        parent_id=parent_uuid,
        installation_id=normalized_installation_id or None,
        device_name=normalized_name or normalized_installation_id,
        role=role,
        device_token=token,
        access_token_version=1,
    )
    db.add(device)
    db.commit()
    db.refresh(device)
    return device


def update_device_token(db: Session, parent_id: str, device_id: str, device_token: Optional[str]) -> Device:
    device = get_device(db, parent_id, device_id)
    token = (device_token or "").strip() or None

    if token == device.device_token:
        return device

    try:
        now = datetime.now(timezone.utc)
        if token is not None:
            (
                db.query(Device)
                .filter(Device.device_token == token, Device.id != device.id)
                .update(
                    {
                        Device.device_token: None,
                        Device.updated_at: now,
                    },
                    synchronize_session=False,
                )
            )

        (
            db.query(Device)
            .filter(Device.id == device.id)
            .update(
                {
                    Device.device_token: token,
                    Device.updated_at: now,
                },
                synchronize_session=False,
            )
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="device_token_conflict") from exc

    db.refresh(device)
    return device


def list_devices(db: Session, parent_id: str, since_ms: Optional[int] = None) -> List[Device]:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    query = db.query(Device).filter(Device.parent_id == parent_uuid)
    if since_ms is not None:
        since_dt = datetime.fromtimestamp(float(since_ms) / 1000.0, timezone.utc)
        query = query.filter(Device.updated_at >= since_dt)
    return query.order_by(Device.created_at.desc()).all()


def _send_monitoring_command_to_device(device: Device, monitoring_enabled: bool) -> bool:
    token = (device.device_token or "").strip()
    if not token:
        return False

    try:
        # Same Firebase initialisation as alert pushes (including the local
        # credentials-file fallback), so whichever runs first works for both.
        from app.utils.notification_worker import _init_firebase

        messaging = _init_firebase()
        if messaging is None:
            return False

        message = messaging.Message(
            data={
                "type": "monitoring_control",
                "action": "start" if monitoring_enabled else "stop",
                "monitoring_enabled": "true" if monitoring_enabled else "false",
                "device_id": str(device.id),
            },
            android=messaging.AndroidConfig(priority="high"),
            token=token,
        )
        messaging.send(message, dry_run=False)
        return True
    except Exception as exc:
        logger.warning(
            "MONITORING_COMMAND_SEND_FAILED | device_id=%s | reason=%s",
            str(device.id),
            str(exc),
        )
        return False


def set_device_monitoring(
    db: Session,
    parent_id: str,
    device_id: str,
    monitoring_enabled: bool,
) -> Tuple[Device, bool]:
    device = get_device(db, parent_id, device_id)
    device.monitoring_enabled = monitoring_enabled
    device.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(device)

    if not monitoring_enabled:
        try:
            stop_session(parent_id, device_id)
        except Exception:
            logger.warning("DEVICE_SESSION_STOP_FAILED | device_id=%s", device_id)

    command_sent = False
    if device.role == DeviceRole.child_device:
        command_sent = _send_monitoring_command_to_device(device, monitoring_enabled)
    return device, command_sent


def update_device_heartbeat(
    db: Session,
    parent_id: str,
    device_id: str,
    battery_percent: Optional[int],
    is_online: Optional[bool],
    monitoring_enabled: Optional[bool],
) -> Device:
    device = get_device(db, parent_id, device_id)

    if battery_percent is not None:
        battery = int(max(0, min(100, battery_percent)))
        device.battery_percent = battery
    if is_online is not None:
        device.is_online = is_online
    if monitoring_enabled is not None:
        device.monitoring_enabled = monitoring_enabled

    device.last_heartbeat_at = datetime.now(timezone.utc)
    device.last_activity_at = datetime.now(timezone.utc)  # Mark activity on heartbeat
    device.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(device)
    return device


def update_device_battery(
    db: Session,
    parent_id: str,
    device_id: str,
    battery_percent: Optional[int],
    device: Optional[Device] = None,
) -> Device:
    """Persist battery telemetry when provided from chunk/location/ack flows."""
    device = device if device is not None else get_device(db, parent_id, device_id)
    if battery_percent is None:
        return device

    battery = int(max(0, min(100, battery_percent)))
    if device.battery_percent == battery:
        return device

    device.battery_percent = battery
    device.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(device)
    return device


def get_effective_online(device: Device) -> bool:
    """Compute online status from freshness window (last_seen semantics).

    A device is online when recent activity was observed within TTL.
    Activity includes chunk upload, location update, heartbeat, and monitoring ack.
    """
    # Check most recent activity timestamp (could be from chunk, location, or heartbeat)
    latest_activity = None
    if device.last_activity_at:
        latest_activity = device.last_activity_at
    if device.last_heartbeat_at:
        if latest_activity is None or device.last_heartbeat_at > latest_activity:
            latest_activity = device.last_heartbeat_at
    
    if not latest_activity:
        return False
    
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    latest_activity = latest_activity.replace(tzinfo=None)
    age_sec = (now_utc - latest_activity).total_seconds()
    return age_sec <= float(DEVICE_ONLINE_TTL_SEC)


def get_device(db: Session, parent_id: str, device_id: str) -> Device:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    device_uuid = _as_uuid(device_id, "device_id")
    device = db.query(Device).filter(Device.id == device_uuid, Device.parent_id == parent_uuid).first()
    if not device:
        raise HTTPException(status_code=404, detail="device_not_found")
    return device


ACTIVITY_WRITE_INTERVAL_SEC = float(os.environ.get("SAFEEAR_ACTIVITY_WRITE_INTERVAL_SEC", "15"))


def update_device_activity(db: Session, parent_id: str, device_id: str, device: Optional[Device] = None) -> Device:
    """Mark the device as active (last_activity_at), used for "online" status and
    the silence watchdog.  Called for audio chunks, location updates and heartbeats.

    Streaming devices send a chunk about every second; one write per
    ACTIVITY_WRITE_INTERVAL_SEC is enough for the 120 s online window and the
    5-minute silence watchdog.
    """
    device = device if device is not None else get_device(db, parent_id, device_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    last = device.last_activity_at
    if last is not None and last.tzinfo is not None:
        last = last.astimezone(timezone.utc).replace(tzinfo=None)
    if last is None or (now - last).total_seconds() > ACTIVITY_WRITE_INTERVAL_SEC:
        device.last_activity_at = now
        db.commit()
        db.refresh(device)
    return device


def update_device_location(db: Session, parent_id: str, device_id: str, lat: Optional[float], lon: Optional[float]) -> Device:
    device = get_device(db, parent_id, device_id)
    if lat is not None:
        device.last_location_lat = lat
    if lon is not None:
        device.last_location_lon = lon
    device.last_location_ts = datetime.now(timezone.utc)
    device.last_activity_at = datetime.now(timezone.utc)  # Mark activity on location update
    db.commit()
    db.refresh(device)
    return device


def create_speaker(db: Session, parent_id: str, display_name: str) -> EnrolledSpeaker:
    parent_uuid = _as_uuid(parent_id, "parent_id")

    normalized_name = (display_name or "").strip()
    if not normalized_name:
        raise HTTPException(status_code=422, detail="display_name_required")

    # Reuse an existing speaker entry for the same name (case-insensitive).
    matches = (
        db.query(EnrolledSpeaker)
        .filter(
            EnrolledSpeaker.parent_id == parent_uuid,
            func.lower(EnrolledSpeaker.display_name) == normalized_name.lower(),
        )
        .order_by(EnrolledSpeaker.sample_count.desc(), EnrolledSpeaker.updated_at.desc())
        .all()
    )

    if matches:
        # Prefer rows that still have real embeddings on disk over stale metadata-only rows.
        primary = max(
            matches,
            key=lambda sp: (
                _count_speaker_embedding_files(parent_id, str(sp.id)),
                int(sp.sample_count or 0),  # type: ignore
                sp.updated_at,
            ),
        )
        primary_dir = _speaker_dir(parent_id, str(primary.id))
        os.makedirs(primary_dir, exist_ok=True)

        # Merge duplicate rows/files into the primary row so UI shows a single entry.
        for dup in [sp for sp in matches if sp.id != primary.id]:
            dup_dir = _speaker_dir(parent_id, str(dup.id))
            moved = 0
            if os.path.isdir(dup_dir):
                existing_files = [f for f in os.listdir(primary_dir) if f.endswith(".npy")]
                next_idx = len(existing_files) + 1
                for fname in sorted(os.listdir(dup_dir)):
                    if not fname.endswith(".npy"):
                        continue
                    src = os.path.join(dup_dir, fname)
                    dst = os.path.join(primary_dir, f"emb_{next_idx}.npy")
                    next_idx += 1
                    try:
                        shutil.move(src, dst)
                        moved += 1
                    except Exception:
                        continue
                try:
                    shutil.rmtree(dup_dir)
                except OSError:
                    pass

            primary.sample_count = int(primary.sample_count or 0) + int(max(moved, int(dup.sample_count or 0)))  # type: ignore
            db.delete(dup)

        actual_count, _ = _sync_speaker_sample_count(primary, parent_id)
        primary.sample_count = actual_count
        primary.display_name = normalized_name
        primary.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(primary)
        return primary

    speaker = EnrolledSpeaker(
        id=uuid.uuid4(),
        parent_id=parent_uuid,
        display_name=normalized_name,
        sample_count=0
    )
    db.add(speaker)
    db.commit()
    db.refresh(speaker)
    os.makedirs(_speaker_dir(parent_id, str(speaker.id)), exist_ok=True)
    return speaker


def _speaker_dir(parent_id: Union[str, uuid.UUID], speaker_id: Union[str, uuid.UUID]) -> str:
    dirs = _ensure_parent_dirs(parent_id)
    return os.path.join(dirs["embeddings"], str(speaker_id))


def get_speaker_embedding_dir(parent_id: str, speaker_id: str) -> str:
    return _speaker_dir(parent_id, speaker_id)


def _count_speaker_embedding_files(parent_id: Union[str, uuid.UUID], speaker_id: Union[str, uuid.UUID]) -> int:
    sdir = _speaker_dir(parent_id, speaker_id)
    if not os.path.isdir(sdir):
        return 0
    return len([f for f in os.listdir(sdir) if f.endswith(".npy")])


def _sync_speaker_sample_count(speaker: EnrolledSpeaker, parent_id: str) -> Tuple[int, bool]:
    actual_count = _count_speaker_embedding_files(parent_id, str(speaker.id))
    sc = speaker.sample_count
    changed = (int(str(sc)) if sc is not None else 0) != actual_count
    if changed:
        speaker.sample_count = actual_count
        speaker.updated_at = datetime.now(timezone.utc)
    return actual_count, changed


def _speaker_avatar_candidates(parent_id: Union[str, uuid.UUID], speaker_id: Union[str, uuid.UUID]) -> List[str]:
    sdir = _speaker_dir(parent_id, speaker_id)
    return [
        os.path.join(sdir, "profile.jpg"),
        os.path.join(sdir, "profile.jpeg"),
        os.path.join(sdir, "profile.png"),
    ]


def get_speaker_avatar_path(parent_id: str, speaker_id: str) -> Optional[str]:
    for path in _speaker_avatar_candidates(parent_id, speaker_id):
        if os.path.isfile(path):
            return path
    return None


def save_speaker_avatar(
    db: Session,
    parent_id: str,
    speaker_id: str,
    image_bytes: bytes,
    content_type: str,
) -> str:
    speaker = get_speaker(db, parent_id, speaker_id)
    sdir = _speaker_dir(parent_id, str(speaker.id))
    os.makedirs(sdir, exist_ok=True)

    normalized_type = (content_type or "").lower().strip()
    ext = ".png" if normalized_type == "image/png" else ".jpg"
    target = os.path.join(sdir, f"profile{ext}")

    # Ensure only one avatar file exists for stable storage and no orphaned images.
    for existing in _speaker_avatar_candidates(parent_id, str(speaker.id)):
        if os.path.isfile(existing):
            try:
                os.remove(existing)
            except OSError:
                pass

    with open(target, "wb") as f:
        f.write(image_bytes)

    speaker.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(speaker)
    return target


def list_speakers(db: Session, parent_id: str) -> List[EnrolledSpeaker]:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    rows = (
        db.query(EnrolledSpeaker)
        .filter(EnrolledSpeaker.parent_id == parent_uuid)
        .order_by(EnrolledSpeaker.created_at.desc())
        .all()
    )

    valid_rows: List[EnrolledSpeaker] = []
    dirty = False
    for row in rows:
        actual_count, changed = _sync_speaker_sample_count(row, parent_id)
        if changed:
            dirty = True
        if actual_count > 0:
            valid_rows.append(row)

    if dirty:
        db.commit()

    if rows and not valid_rows:
        logger.warning("SPEAKER_LIST_EMPTY_EMBEDDINGS | parent_id=%s | metadata_rows=%s", parent_id, len(rows))

    return valid_rows


def get_speaker(db: Session, parent_id: str, speaker_id: str) -> EnrolledSpeaker:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    speaker_uuid = _as_uuid(speaker_id, "speaker_id")
    speaker = db.query(EnrolledSpeaker).filter(
        EnrolledSpeaker.id == speaker_uuid,
        EnrolledSpeaker.parent_id == parent_uuid
    ).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="speaker_not_found")
    return speaker


def rename_speaker(db: Session, parent_id: str, speaker_id: str, new_name: str) -> EnrolledSpeaker:
    speaker = get_speaker(db, parent_id, speaker_id)
    speaker.display_name = new_name
    speaker.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(speaker)
    return speaker


def delete_speaker(db: Session, parent_id: str, speaker_id: str) -> None:
    speaker = get_speaker(db, parent_id, speaker_id)
    db.delete(speaker)
    db.commit()
    sdir = _speaker_dir(parent_id, speaker_id)
    if os.path.exists(sdir):
        shutil.rmtree(sdir)
    invalidate_gallery(parent_id)


def save_speaker_embedding(db: Session, parent_id: str, speaker_id: str, emb: np.ndarray) -> str:
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    if arr.shape[0] != config.EMBEDDING_DIM:
        raise HTTPException(status_code=400, detail=f"invalid_embedding_dim_{arr.shape[0]}")
    sdir = _speaker_dir(parent_id, speaker_id)
    os.makedirs(sdir, exist_ok=True)
    existing = [f for f in os.listdir(sdir) if f.endswith(".npy")]
    # Next number after the highest existing one, so a gap can never overwrite a file.
    numbers = [int(f[4:-4]) for f in existing if f.startswith("emb_") and f[4:-4].isdigit()]
    idx = max(numbers, default=0) + 1
    fpath = os.path.join(sdir, f"emb_{idx}.npy")
    while os.path.exists(fpath):
        idx += 1
        fpath = os.path.join(sdir, f"emb_{idx}.npy")
    np.save(fpath, arr)
    invalidate_gallery(parent_id)

    speaker = get_speaker(db, parent_id, speaker_id)
    speaker.sample_count = len(existing) + 1
    speaker.updated_at = datetime.now(timezone.utc)
    db.commit()
    return fpath


# Structured container for per-speaker gallery info used by the scoring path.
# Keyed by speaker_id string; each value carries embeddings plus metadata
# needed for ZT-norm and role-aware decision logic.
_SpeakerGalleryEntry = Dict[str, Any]  # keys: embs, role, t_high_override, cohort_mean, cohort_std


GALLERY_CACHE_TTL_SEC = float(os.environ.get("SAFEEAR_GALLERY_CACHE_TTL_SEC", "600"))
_GALLERY_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_GALLERY_LOCK = threading.Lock()


def _unit_rows(vectors: List[np.ndarray]) -> np.ndarray:
    """Stack vectors as rows scaled to unit length (zero vectors stay zero)."""
    mat = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors], axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.where(norms > 0.0, mat / np.where(norms > 0.0, norms, 1.0), 0.0).astype(np.float32)


def invalidate_gallery(parent_id: str) -> None:
    """Drop the cached voiceprints of a family (call after any enrollment change)."""
    with _GALLERY_LOCK:
        _GALLERY_CACHE.pop(str(parent_id), None)


def load_parent_embeddings(db: Session, parent_id: str) -> Dict[str, "_SpeakerGalleryEntry"]:
    """Enrolled voiceprints of a family, cached in memory.

    The detect path calls this for every scored window; reading the database and
    every .npy file each time was the slowest non-model step.  The cache is
    invalidated on enroll/delete and expires after GALLERY_CACHE_TTL_SEC as a
    safety net for files changed outside the API.  Each entry also carries
    "matrix": the voiceprints stacked as unit-length rows for fast scoring.
    Returns a fresh dict, so callers may remove keys without touching the cache.
    """
    key = str(parent_id)
    now = time.monotonic()
    with _GALLERY_LOCK:
        hit = _GALLERY_CACHE.get(key)
        if hit is not None and now - hit[0] < GALLERY_CACHE_TTL_SEC:
            return dict(hit[1])
    gallery = _load_parent_embeddings_uncached(db, key)
    for entry in gallery.values():
        entry["matrix"] = _unit_rows(entry["embs"])
    with _GALLERY_LOCK:
        _GALLERY_CACHE[key] = (now, gallery)
    return dict(gallery)


def _load_parent_embeddings_uncached(
    db: Session, parent_id: str
) -> Dict[str, "_SpeakerGalleryEntry"]:
    """Load all enrolled speaker embeddings for a parent.

    Returns a dict keyed by speaker_id.  Each value is a dict with:
        embs            : List[np.ndarray]  L2-normalised 192-D vectors
        role            : str              "familiar_adult" | "child_self"
        t_high_override : float | None     per-speaker threshold (child uses lower value)
        cohort_mean     : float | None     ZT-norm pre-computed mean
        cohort_std      : float | None     ZT-norm pre-computed std
    """
    out: Dict[str, Any] = {}
    for sp in list_speakers(db, parent_id):
        sdir = _speaker_dir(parent_id, str(sp.id))
        if not os.path.isdir(sdir):
            continue
        embs: List[np.ndarray] = []
        for fname in sorted(os.listdir(sdir)):
            if not fname.endswith(".npy"):
                continue
            fpath = os.path.join(sdir, fname)
            # One unreadable or wrong-size file must not disable detection for the
            # whole family: skip it and keep the speaker's other references.
            try:
                arr = np.asarray(np.load(fpath, allow_pickle=False), dtype=np.float32).reshape(-1)
            except Exception as exc:
                logger.error("EMBEDDING_LOAD_FAILED | file=%s | reason=%s | skipped", fpath, str(exc))
                continue
            if arr.shape[0] != config.EXPECTED_EMBEDDING_DIM or not np.all(np.isfinite(arr)):
                logger.error(
                    "EMBEDDING_INVALID | file=%s | expected_dim=%s | got=%s | skipped",
                    fpath, config.EXPECTED_EMBEDDING_DIM, arr.shape[0],
                )
                continue
            embs.append(arr)
        if embs:
            out[str(sp.id)] = {
                "embs": embs,
                "role": getattr(sp, "role", "familiar_adult") or "familiar_adult",
                "t_high_override": getattr(sp, "t_high_override", None),
                "cohort_mean": getattr(sp, "cohort_mean", None),
                "cohort_std": getattr(sp, "cohort_std", None),
            }
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def score_against_parent(
    db: Session, parent_id: str, query_embs: List[np.ndarray]
) -> Tuple[float, Optional[str], Optional[str]]:
    candidates = load_parent_embeddings(db, parent_id)
    return _score_against_candidates(candidates, query_embs, parent_id)


def _score_against_candidates(
    candidates: Dict[str, Any],
    query_embs: List[np.ndarray],
    parent_id: Optional[str] = None,
    t_high_global: float = T_HIGH,
    scores_out: Optional[Dict[str, float]] = None,
) -> Tuple[float, Optional[str], Optional[str]]:
    """Score query embeddings against all enrolled speakers.

    Args:
        candidates  : dict from load_parent_embeddings() — each value is a
                      _SpeakerGalleryEntry dict with keys: embs, role,
                      t_high_override, cohort_mean, cohort_std.
        query_embs  : list of L2-normalised query embeddings for this chunk.
        parent_id   : used for logging only.
        t_high_global : household / global T_HIGH used for ZT-norm context.
        scores_out  : if given, filled with {speaker_id: score} for every speaker.

    Returns:
        (best_score, best_speaker_id, best_speaker_role)
        best_score is ZT-normalised when cohort stats are available, otherwise raw.
    """
    best = -1.0
    best_speaker: Optional[str] = None
    best_role: Optional[str] = None
    speaker_scores: Dict[str, float] = {}

    if not query_embs:
        return 0.0, None, None
    queries = _unit_rows(query_embs)

    # Adaptive symmetric normalisation (opt-in, SAFEEAR_SCORE_NORM=asnorm).  Its output
    # is calibrated to the cosine scale, so the usual thresholds apply.
    from app.utils.score_normaliser import get_asnorm
    asnorm = get_asnorm()
    query_stats = asnorm.query_stats(queries) if asnorm is not None else None

    for sid, entry in candidates.items():
        # Support both legacy List[np.ndarray] (no metadata) and new dict format.
        if isinstance(entry, dict):
            embs = entry["embs"]
            refs = entry.get("matrix")
            role = entry.get("role", "familiar_adult")
            cohort_mean = entry.get("cohort_mean")
            cohort_std = entry.get("cohort_std")
        else:
            embs = entry  # backward compat
            refs = None
            role = "familiar_adult"
            cohort_mean = None
            cohort_std = None
        if not len(embs):
            continue
        if refs is None:
            refs = _unit_rows(embs)

        # Cosine of every query chunk against every reference in one product;
        # each chunk keeps its best-matching reference.
        best_per_chunk = (queries @ refs.T).max(axis=1)
        if asnorm is not None:
            per_chunk_best = [float(v) for v in best_per_chunk]  # normalisation needs the full range
        else:
            per_chunk_best = [float(v) for v in best_per_chunk if v >= 0.0]
        if not per_chunk_best:
            continue

        # Robust score: up-weight median, down-weight a single lucky match.
        top_chunks = sorted(per_chunk_best, reverse=True)[: min(2, len(per_chunk_best))]
        raw_score = float((0.7 * np.median(per_chunk_best)) + (0.3 * np.mean(top_chunks)))

        if asnorm is not None:
            cached = entry.get("_asnorm_stats") if isinstance(entry, dict) else None
            if cached is not None and cached[0] is asnorm:
                enroll_stats = cached[1]
            else:
                enroll_stats = asnorm.enroll_stats(refs)
                if isinstance(entry, dict):
                    entry["_asnorm_stats"] = (asnorm, enroll_stats)
            speaker_score = float(asnorm.normalize(raw_score, enroll_stats, query_stats))
        # Cohort z-norm (opt-in).  The result is a z-score, not a cosine, so it
        # must not be clipped or compared against cosine thresholds.
        elif ENABLE_SCORE_NORM and cohort_mean is not None and cohort_std is not None and cohort_std > 1e-6:
            from app.utils.score_normaliser import zt_norm
            speaker_score = float(zt_norm(raw_score, cohort_mean, cohort_std))
            logger.debug(
                "ZT_NORM | speaker_id=%s | raw=%.4f | mean=%.4f | std=%.4f | normed=%.4f",
                sid, raw_score, cohort_mean, cohort_std, speaker_score,
            )
        else:
            speaker_score = raw_score

        speaker_scores[sid] = speaker_score
        if scores_out is not None:
            scores_out[str(sid)] = round(float(speaker_score), 4)
        if speaker_score > best:
            best = speaker_score
            best_speaker = sid
            best_role = role

    if speaker_scores:
        logger.debug(
            "SPEAKER_SCORING | parent_id=%s | scores={%s}",
            parent_id or "unknown",
            ", ".join(f"{sid}:{score:.4f}" for sid, score in speaker_scores.items()),
        )

    if best < 0:
        logger.info("NO_SPEAKER_MATCH | parent_id=%s | score=0.0", parent_id or "unknown")
        return 0.0, None, None

    logger.info(
        "BEST_SPEAKER_MATCHED | parent_id=%s | speaker_id=%s | role=%s | score=%.4f",
        parent_id or "unknown", best_speaker, best_role, best,
    )
    return float(best), best_speaker, best_role


def create_alert(
    db: Session,
    parent_id: str,
    device_id: str,
    audio_clip_path: str,
    confidence_score: float,
    latitude: Optional[float],
    longitude: Optional[float],
    alert_type: str = "unenrolled_contact",
) -> AlertModel:
    """Persist an alert record.

    alert_type: "unenrolled_contact" (default) | "distress"
    """
    parent_uuid = _as_uuid(parent_id, "parent_id")
    device_uuid = _as_uuid(device_id, "device_id")
    alert = AlertModel(
        id=uuid.uuid4(),
        parent_id=parent_uuid,
        device_id=device_uuid,
        timestamp=datetime.now(timezone.utc),
        confidence_score=float(confidence_score),
        audio_clip_path=audio_clip_path,
        latitude=latitude,
        longitude=longitude,
        acknowledged_at=None,
        alert_type=alert_type,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    logger.info(
        "ALERT_CREATED | alert_id=%s | parent_id=%s | device_id=%s | alert_type=%s | confidence_score=%.4f | has_location=%s",
        alert.id, parent_id, device_id, alert_type, confidence_score, latitude is not None,
    )
    return alert


# ---------------------------------------------------------------------------
# Change 4: Confusability detection
# ---------------------------------------------------------------------------

def check_enrollment_confusability(
    db: Session,
    parent_id: str,
    new_speaker_id: str,
    threshold: float = 0.75,
) -> List[Tuple[str, float]]:
    """Return list of (speaker_id, similarity) for enrolled speakers whose
    centroid is within `threshold` cosine distance of the newly enrolled speaker.

    Called after every enrollment or re-enrollment.  Conflicts at ≥ 0.75 cosine
    indicate the enrolled pair will frequently cross-match in inference, which
    suppresses alerts that should fire.

    Returns an empty list when there are fewer than 2 speakers total.
    """
    all_embeddings = load_parent_embeddings(db, parent_id)
    new_entry = all_embeddings.pop(new_speaker_id, None)
    if not new_entry or not other_embeddings_exist(all_embeddings):
        return []

    new_embs: List[np.ndarray] = new_entry["embs"] if isinstance(new_entry, dict) else new_entry
    new_centroid = _compute_query_centroid(new_embs)
    if new_centroid is None:
        return []

    conflicts: List[Tuple[str, float]] = []
    for sid, entry in all_embeddings.items():
        embs: List[np.ndarray] = entry["embs"] if isinstance(entry, dict) else entry
        other_centroid = _compute_query_centroid(embs)
        if other_centroid is None:
            continue
        sim = _cos(new_centroid, other_centroid)
        if sim >= threshold:
            conflicts.append((sid, float(sim)))

    conflicts.sort(key=lambda x: -x[1])
    if conflicts:
        logger.warning(
            "CONFUSABILITY_DETECTED | parent_id=%s | new_speaker=%s | conflicts=%s",
            parent_id,
            new_speaker_id,
            [(sid, f"{s:.3f}") for sid, s in conflicts],
        )
    return conflicts


def other_embeddings_exist(gallery: Dict[str, Any]) -> bool:
    """True if the gallery contains at least one speaker entry."""
    return bool(gallery)


# ---------------------------------------------------------------------------
# Change 5: Per-household calibrated thresholds
# ---------------------------------------------------------------------------

_GLOBAL_T_HIGH = T_HIGH
_GLOBAL_T_LOW = T_LOW

# Per-household thresholds are opt-in.  Calibrating from enrollment clips alone
# (clean phone audio, family-only impostors) does not transfer to the worn
# device, so the field-tuned global thresholds are used unless this is enabled.
USE_HOUSEHOLD_THRESHOLDS = os.environ.get("SAFEEAR_USE_HOUSEHOLD_THRESHOLDS", "false").strip().lower() in {"1", "true", "yes", "on"}
# When false, auto_calibrate_thresholds() only logs what it would have set.
AUTO_CALIBRATE_APPLY = os.environ.get("SAFEEAR_AUTO_CALIBRATE_APPLY", "false").strip().lower() in {"1", "true", "yes", "on"}
# Cohort score normalisation is opt-in; thresholds must be re-derived in the
# z-score domain before enabling it.
ENABLE_SCORE_NORM = os.environ.get("SAFEEAR_ENABLE_SCORE_NORM", "false").strip().lower() in {"1", "true", "yes", "on"}


def get_household_thresholds(db: Session, parent_id: str) -> Tuple[float, float]:
    """Return (T_HIGH, T_LOW) for this household.

    Returns the global env-var thresholds unless SAFEEAR_USE_HOUSEHOLD_THRESHOLDS
    is enabled, in which case a calibrated HouseholdConfig record takes precedence.
    """
    if not USE_HOUSEHOLD_THRESHOLDS:
        return _GLOBAL_T_HIGH, _GLOBAL_T_LOW
    try:
        parent_uuid = _as_uuid(parent_id, "parent_id")
        cfg = db.query(HouseholdConfig).filter(
            HouseholdConfig.parent_id == parent_uuid
        ).first()
        if cfg is not None and cfg.t_high is not None and cfg.t_low is not None:
            return float(getattr(cfg, "t_high")), float(getattr(cfg, "t_low"))
    except Exception as exc:
        logger.warning("HOUSEHOLD_THRESHOLD_LOAD_FAILED | parent_id=%s | reason=%s", parent_id, exc)
    return _GLOBAL_T_HIGH, _GLOBAL_T_LOW


def auto_calibrate_thresholds(db: Session, parent_id: str) -> Optional[Tuple[float, float]]:
    """Compute EER-based T_HIGH / T_LOW from current enrollment gallery.

    Called automatically at the end of compute_and_store_enrollment_quality
    when sample_count >= 5.  Saves result to HouseholdConfig.

    Returns (T_HIGH, T_LOW) or None if calibration was skipped.
    """
    all_embeddings = load_parent_embeddings(db, parent_id)
    if not all_embeddings:
        return None

    # Collect target scores (leave-one-out: each embedding scored against
    # the other embeddings of the same speaker).
    target_scores: List[float] = []
    for sid, entry in all_embeddings.items():
        embs: List[np.ndarray] = entry["embs"] if isinstance(entry, dict) else entry
        if len(embs) < 2:
            continue
        for i, q in enumerate(embs):
            refs = embs[:i] + embs[i + 1:]
            scores = [_cos(q, r) for r in refs]
            if scores:
                target_scores.append(float(np.mean(scores)))

    if len(target_scores) < 3:
        logger.info(
            "AUTO_CALIBRATE_SKIPPED | parent_id=%s | reason=insufficient_target_scores | n=%d",
            parent_id, len(target_scores),
        )
        return None

    # Collect impostor scores by cross-scoring speakers against each other.
    impostor_scores: List[float] = []
    speaker_ids = list(all_embeddings.keys())
    for i, sid_a in enumerate(speaker_ids):
        entry_a = all_embeddings[sid_a]
        embs_a: List[np.ndarray] = entry_a["embs"] if isinstance(entry_a, dict) else entry_a
        for sid_b in speaker_ids[i + 1:]:
            entry_b = all_embeddings[sid_b]
            embs_b: List[np.ndarray] = entry_b["embs"] if isinstance(entry_b, dict) else entry_b
            for q in embs_a[:3]:  # cap to 3 to avoid N² blowup
                scores = [_cos(q, r) for r in embs_b[:3]]
                impostor_scores.extend(scores)

    if len(impostor_scores) < 3:
        # Not enough cross-speaker data; need more enrolled speakers.
        logger.info(
            "AUTO_CALIBRATE_SKIPPED | parent_id=%s | reason=need_more_speakers | impostor_n=%d",
            parent_id, len(impostor_scores),
        )
        return None

    try:
        from app.utils.calibration import find_eer_threshold
        eer_t, eer = find_eer_threshold(target_scores, impostor_scores)
    except Exception:
        # Fallback: use global defaults
        logger.warning("AUTO_CALIBRATE_FAILED | parent_id=%s", parent_id, exc_info=True)
        return None

    # Margin below EER threshold for T_LOW
    margin = float(os.environ.get("SAFEEAR_T_MARGIN", "0.10"))
    min_t_high = 0.45 if _IS_REDIMNET2 else 0.30
    t_high = float(np.clip(eer_t, min_t_high, 0.90))
    t_low = float(np.clip(eer_t - margin, min_t_high - 0.15, t_high - 0.05))

    if not AUTO_CALIBRATE_APPLY:
        logger.info(
            "AUTO_CALIBRATE_SHADOW | parent_id=%s | would_set_t_high=%.4f | would_set_t_low=%.4f | eer=%.4f "
            "| target_n=%d | impostor_n=%d | active_t_high=%.4f | active_t_low=%.4f",
            parent_id, t_high, t_low, eer, len(target_scores), len(impostor_scores),
            _GLOBAL_T_HIGH, _GLOBAL_T_LOW,
        )
        return None

    try:
        parent_uuid = _as_uuid(parent_id, "parent_id")
        cfg = db.query(HouseholdConfig).filter(
            HouseholdConfig.parent_id == parent_uuid
        ).first()
        sample_count = sum(
            len(e["embs"] if isinstance(e, dict) else e)
            for e in all_embeddings.values()
        )
        if cfg is None:
            cfg = HouseholdConfig(
                id=uuid.uuid4(),
                parent_id=parent_uuid,
                t_high=t_high,
                t_low=t_low,
                calibrated_at=datetime.now(timezone.utc),
                calibration_sample_count=sample_count,
            )
            db.add(cfg)
        else:
            cfg.t_high = t_high
            cfg.t_low = t_low
            cfg.calibrated_at = datetime.now(timezone.utc)
            cfg.calibration_sample_count = sample_count
            cfg.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            "AUTO_CALIBRATE_DONE | parent_id=%s | t_high=%.4f | t_low=%.4f | eer=%.4f | samples=%d",
            parent_id, t_high, t_low, eer, sample_count,
        )
    except Exception as exc:
        logger.error("AUTO_CALIBRATE_SAVE_FAILED | parent_id=%s | reason=%s", parent_id, exc)

    return t_high, t_low


def get_household_config_dict(db: Session, parent_id: str) -> Dict[str, Any]:
    """Change 12: Fetch context gates and thresholds."""
    try:
        parent_uuid = uuid.UUID(parent_id)
        cfg = db.query(HouseholdConfig).filter(
            HouseholdConfig.parent_id == parent_uuid
        ).first()
        
        if not cfg:
            return {}
            
        return {
            "t_high": cfg.t_high,
            "t_low": cfg.t_low,
            "quiet_hours_start": cfg.quiet_hours_start,
            "quiet_hours_end": cfg.quiet_hours_end,
            "home_lat": cfg.home_lat,
            "home_lon": cfg.home_lon,
            "home_radius_m": cfg.home_radius_m,
            "party_mode_until": cfg.party_mode_until.isoformat() if cfg.party_mode_until else None,
        }
    except Exception as e:
        logger.error("Failed to fetch household config for parent %s: %s", parent_id, e)
        return {}


def flush_child_drift_scores(db: Session, parent_id: str, speaker_id: str, session_state: 'SessionState') -> Optional[bool]:
    """Change 10: Update the drift score EMA for the child speaker.
    Returns True if a drift warning should be emitted, False otherwise.
    """
    if not session_state.child_drift_scores:
        return None
        
    try:
        from app.database.models import EnrolledSpeaker
        parent_uuid = uuid.UUID(parent_id)
        sp_uuid = uuid.UUID(speaker_id)
        
        sp = db.query(EnrolledSpeaker).filter(
            EnrolledSpeaker.parent_id == parent_uuid,
            EnrolledSpeaker.id == sp_uuid
        ).first()
        
        if not sp:
            session_state.child_drift_scores.clear()
            return None
            
        # Calculate new average of the accumulated scores
        scores = session_state.child_drift_scores
        batch_avg = sum(scores) / len(scores)
        session_state.child_drift_scores.clear()
        
        # Exponential moving average (EMA) update
        if getattr(sp, "drift_score_ema") is None:
            setattr(sp, "drift_score_ema", float(batch_avg))
        else:
            alpha = 0.1  # Update rate
            current_ema = float(getattr(sp, "drift_score_ema"))
            setattr(sp, "drift_score_ema", float((alpha * batch_avg) + ((1.0 - alpha) * current_ema)))
            
        sp.last_drift_check_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        
        # Check if drift is severe enough to warrant re-enrollment
        if sp.quality_score and sp.drift_score_ema is not None and sp.drift_score_ema < (sp.quality_score * 0.80):  # type: ignore
            # Only warn once every 7 days
            if not sp.drift_alert_sent_at or (datetime.now(timezone.utc).replace(tzinfo=None) - sp.drift_alert_sent_at).total_seconds() > 7 * 86400:
                sp.drift_alert_sent_at = datetime.now(timezone.utc).replace(tzinfo=None)
                db.commit()
                return True
                
        return False
    except Exception as e:
        logger.error("Failed to flush child drift scores: %s", e)
        db.rollback()
        return None


def list_alerts(
    db: Session,
    parent_id: str,
    limit: int = 50,
    offset: int = 0,
    since_ms: Optional[int] = None,
) -> List[AlertModel]:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    query = db.query(AlertModel).filter(AlertModel.parent_id == parent_uuid)
    if since_ms is not None:
        since_dt = datetime.fromtimestamp(float(since_ms) / 1000.0, timezone.utc)
        query = query.filter(AlertModel.updated_at >= since_dt)
    return query.order_by(AlertModel.timestamp.desc()).offset(offset).limit(limit).all()


def ack_alert(db: Session, parent_id: str, alert_id: str) -> AlertModel:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    alert_uuid = _as_uuid(alert_id, "alert_id")
    alert = db.query(AlertModel).filter(
        AlertModel.id == alert_uuid,
        AlertModel.parent_id == parent_uuid
    ).first()
    if not alert:
        raise HTTPException(status_code=404, detail="alert_not_found")
    alert.acknowledged_at = datetime.now(timezone.utc)
    alert.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return alert


def get_alert(db: Session, parent_id: str, alert_id: str) -> AlertModel:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    alert_uuid = _as_uuid(alert_id, "alert_id")
    alert = db.query(AlertModel).filter(
        AlertModel.id == alert_uuid,
        AlertModel.parent_id == parent_uuid
    ).first()
    if not alert:
        raise HTTPException(status_code=404, detail="alert_not_found")
    return alert


def _delete_clip_path(path: Optional[str]) -> None:
    """Remove an alert clip together with its alert folder (metadata.json included)."""
    if not path:
        return
    try:
        clip_dir = os.path.dirname(path)
        alerts_root = os.path.dirname(clip_dir)
        if clip_dir and os.path.basename(alerts_root) == "alerts" and os.path.isdir(clip_dir):
            shutil.rmtree(clip_dir, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except OSError:
        # Clip cleanup should never block DB deletion.
        pass


CLIP_RETENTION_DAYS = int(os.environ.get("SAFEEAR_CLIP_RETENTION_DAYS", "30"))


def purge_old_alert_clips(db: Session, retention_days: int = CLIP_RETENTION_DAYS) -> int:
    """Delete alert audio older than the retention period.

    The alert record (time, device, score, location) stays in the history; only
    the recording of the people involved is removed.  0 disables purging.
    """
    if retention_days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=retention_days)
    rows = (
        db.query(AlertModel)
        .filter(AlertModel.audio_clip_path.isnot(None), AlertModel.timestamp < cutoff)
        .all()
    )
    for row in rows:
        _delete_clip_path(str(row.audio_clip_path))
        row.audio_clip_path = None
    if rows:
        db.commit()
    return len(rows)


def delete_parent_account(db: Session, parent_id: str) -> None:
    """Permanently delete a family: recordings, voiceprints, devices, alerts, sessions."""
    from app.database.models import RefreshToken

    parent = get_parent(db, parent_id)
    parent_uuid = parent.id
    for device in list(parent.devices):
        try:
            stop_session(str(parent_uuid), str(device.id))
        except Exception:
            pass
    db.query(RefreshToken).filter(RefreshToken.parent_id == parent_uuid).delete(synchronize_session=False)
    db.query(HouseholdConfig).filter(HouseholdConfig.parent_id == parent_uuid).delete(synchronize_session=False)
    from app.database.models import PairingRequest

    db.query(PairingRequest).filter(PairingRequest.claimed_parent_id == parent_uuid).delete(synchronize_session=False)
    db.delete(parent)  # cascades to devices, speakers, alerts and credentials
    db.commit()
    shutil.rmtree(os.path.join(TENANTS_ROOT, str(parent_uuid)), ignore_errors=True)
    invalidate_gallery(str(parent_uuid))


def delete_alert(db: Session, parent_id: str, alert_id: str) -> None:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    alert_uuid = _as_uuid(alert_id, "alert_id")
    alert = db.query(AlertModel).filter(
        AlertModel.id == alert_uuid,
        AlertModel.parent_id == parent_uuid
    ).first()
    if not alert:
        raise HTTPException(status_code=404, detail="alert_not_found")

    clip_path = alert.audio_clip_path
    db.delete(alert)
    db.commit()
    _delete_clip_path(str(clip_path) if clip_path else None)


def delete_all_alerts(db: Session, parent_id: str) -> int:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    rows = db.query(AlertModel).filter(AlertModel.parent_id == parent_uuid).all()
    if not rows:
        return 0

    clip_paths = [row.audio_clip_path for row in rows]
    deleted_count = len(rows)
    for row in rows:
        db.delete(row)
    db.commit()

    for path in clip_paths:
        _delete_clip_path(str(path) if path else None)

    return deleted_count


@dataclass
class SessionState:
    parent_id: str
    device_id: str
    sr: int = 16000
    inference_buffer: Optional[np.ndarray] = None
    clip_buffer: Optional[np.ndarray] = None
    stranger_segment: Optional[np.ndarray] = None
    stranger_segment_started_ms: int = 0
    stranger_segment_ended_ms: int = 0
    stranger_streak: int = 0
    last_alert_ms: int = 0
    lat: Optional[float] = None
    lon: Optional[float] = None
    last_familiar_ms: int = 0
    last_familiar_speaker_id: Optional[str] = None
    familiar_recovery_streak: int = 0
    active_stranger_embedding: Optional[np.ndarray] = None
    recent_confirmed_stranger_embeddings: List[np.ndarray] = field(default_factory=list)
    # Change 7: consecutive windows with distress_score >= DISTRESS_ALERT_THRESHOLD
    distress_streak: int = 0
    # Chunk stream bookkeeping (client-supplied seq / t_start_ms), used to detect
    # gaps and reordering so the stranger streak only spans contiguous audio.
    stream_id: Optional[str] = None
    last_seq: Optional[int] = None
    last_chunk_end_ms: Optional[int] = None
    last_distress_alert_ms: int = 0
    # Change 10: drift tracking
    child_drift_scores: List[float] = field(default_factory=list)
    # Change 11: weighted streak
    stranger_weight_sum: float = 0.0
    # Change 12: Cached context gate configuration
    household_config: Optional[Dict[str, Any]] = None
    last_config_fetch_ms: int = 0
    # Phase 5: Device obstruction detection
    obstruction_counter: int = 0
    device_obstructed_alert_sent_at: int = 0
    # Parent-nearby rule (app.services.child_safety): cached result per stream.
    proximity_checked_ms: int = 0
    proximity_near: bool = False
    # Alert clip post-roll: the saved clip is rewritten once this much more audio arrived.
    post_roll_clip_path: Optional[str] = None
    post_roll_buffer: Optional[np.ndarray] = None
    post_roll_remaining: int = 0
    last_used_monotonic: float = 0.0

def _session_key(parent_id: str, device_id: str) -> str:
    return f"{parent_id}:{device_id}"


_DEVICE_LOCKS: Dict[str, threading.Lock] = {}


def device_lock(parent_id: str, device_id: str) -> threading.Lock:
    """Per-device lock: chunks from one device are processed strictly one at a time."""
    key = _session_key(parent_id, device_id)
    with _LOCK:
        lock = _DEVICE_LOCKS.get(key)
        if lock is None:
            lock = _DEVICE_LOCKS[key] = threading.Lock()
        return lock


SESSION_IDLE_EVICT_SEC = int(os.environ.get("SAFEEAR_SESSION_IDLE_EVICT_SEC", "600"))
_PERSISTED_FIELDS = (
    "stranger_streak", "stranger_weight_sum", "familiar_recovery_streak", "last_alert_ms",
    "last_familiar_ms", "last_familiar_speaker_id", "stream_id", "last_seq", "last_chunk_end_ms",
    "last_distress_alert_ms",
)


def get_or_create_session(parent_id: str, device_id: str) -> SessionState:
    """In-memory session for a device.  A new session restores its small decision
    state (streak, debounce clock, stream position) from Redis when available, so
    a server restart neither repeats an alert nor forgets a streak in progress."""
    key = _session_key(parent_id, device_id)
    with _LOCK:
        row = _SESSIONS.get(key)
        if row is not None:
            row.last_used_monotonic = time.monotonic()
            return row
    restored = load_session(parent_id, device_id)
    with _LOCK:
        row = _SESSIONS.get(key)
        if row is None:
            row = SessionState(parent_id=parent_id, device_id=device_id)
            for field_name in _PERSISTED_FIELDS:
                if restored and field_name in restored and restored[field_name] is not None:
                    setattr(row, field_name, restored[field_name])
            _SESSIONS[key] = row
        row.last_used_monotonic = time.monotonic()
        return row


def persist_session_state(session: SessionState) -> None:
    """Save the small decision state (a few hundred bytes) after each processed chunk."""
    persist_session(session.parent_id, session.device_id, {f: getattr(session, f) for f in _PERSISTED_FIELDS})


def evict_idle_sessions(max_idle_sec: int = SESSION_IDLE_EVICT_SEC) -> int:
    """Free audio buffers of devices that stopped streaming (state stays in Redis)."""
    cutoff = time.monotonic() - max_idle_sec
    with _LOCK:
        stale = [k for k, v in _SESSIONS.items() if v.last_used_monotonic < cutoff]
        for k in stale:
            del _SESSIONS[k]
            _DEVICE_LOCKS.pop(k, None)
    return len(stale)


def stop_session(parent_id: str, device_id: str) -> None:
    key = _session_key(parent_id, device_id)
    with _LOCK:
        _SESSIONS.pop(key, None)
    try:
        delete_session(parent_id, device_id)
    except Exception:
        pass


def start_stranger_segment(
    session: SessionState,
    waveform: np.ndarray,
    started_ms: int,
    pre_roll_waveform: Optional[np.ndarray] = None,
) -> None:
    """Mark the start of a stranger episode.  Only timestamps are kept: the alert
    clip comes from the rolling clip_buffer, so the audio is not copied here."""
    session.stranger_segment_started_ms = started_ms
    session.stranger_segment_ended_ms = 0


def append_stranger_segment(session: SessionState, waveform: np.ndarray) -> None:
    """Kept for call-site compatibility; the episode audio is not accumulated."""
    return None


def clear_stranger_segment(session: SessionState) -> None:
    session.stranger_segment = None
    session.stranger_segment_started_ms = 0
    session.stranger_segment_ended_ms = 0


def reset_stream(session: SessionState) -> None:
    """Forget audio context after a gap in the stream.  Windows on either side of a
    gap must not be stitched together or count toward the same stranger streak.
    The alert debounce and familiar-grace clocks are kept."""
    session.inference_buffer = None
    session.stranger_streak = 0
    session.stranger_weight_sum = 0.0
    session.familiar_recovery_streak = 0
    clear_stranger_segment(session)


def clear_stranger_identity(session: SessionState) -> None:
    session.active_stranger_embedding = None
    session.recent_confirmed_stranger_embeddings = []


def record_confirmed_stranger_window(session: SessionState, centroid: Optional[np.ndarray]) -> None:
    normed = _normalized_embedding(centroid) if centroid is not None else None
    if normed is None:
        return
    session.recent_confirmed_stranger_embeddings.append(normed)
    if len(session.recent_confirmed_stranger_embeddings) > 3:
        session.recent_confirmed_stranger_embeddings = session.recent_confirmed_stranger_embeddings[-3:]


def set_active_stranger_embedding(session: SessionState) -> Optional[np.ndarray]:
    if not session.recent_confirmed_stranger_embeddings:
        return None
    centroid = np.mean(np.stack(session.recent_confirmed_stranger_embeddings, axis=0), axis=0)
    normed = _normalized_embedding(centroid)
    session.active_stranger_embedding = normed
    return normed


def update_session_location(parent_id: str, device_id: str, lat: Optional[float], lon: Optional[float]) -> None:
    if lat is None or lon is None:
        return  # a location post without coordinates must not erase the last fix
    s = get_or_create_session(parent_id, device_id)
    s.lat = lat
    s.lon = lon


def clear_clip_buffer(session: SessionState) -> None:
    session.clip_buffer = None


def evaluate_window(db: Session, parent_id: str, waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    started_at = time.perf_counter()

    # Change 5: resolve per-household thresholds, fall back to global defaults.
    t_high, t_low = get_household_thresholds(db, parent_id)

    waveform = normalize_waveform(np.asarray(waveform, dtype=np.float32).reshape(-1))
    rms = float(np.sqrt(np.mean(waveform**2) + 1e-12))
    if rms < 0.003:
        perf = _perf_metrics(len(waveform), sr, started_at)
        logger.info(
            "WINDOW_PERF | parent_id=%s | decision=%s | processing_ms=%.1f | audio_ms=%.1f | rtf=%.4f",
            parent_id,
            "rejected_silence",
            perf["processing_ms"],
            perf["audio_ms"],
            perf["rtf"],
        )
        logger.info("TIER1_VAD_REJECTED | parent_id=%s | reason=low_rms | rms=%.6f", parent_id, rms)
        return {
            "tier1_vad": {"passed": False, "voiced_ms": 0.0, "rms": rms},
            "tier2": {"passed": False, "confidence": 0.0, "reason": "low_rms"},
            "tier3": {"passed": False, "score": 0.0, "closest_speaker_id": None, "closest_speaker_role": None},
            "distress": {"distress": False, "distress_score": 0.0, "distress_class": None},
            "decision": "rejected_silence",
            "perf": perf,
        }

    try:
        segments = get_speech_segments(waveform, sr=sr, use_silero=True)
    except Exception:
        segments = get_speech_segments(waveform, sr=sr, use_silero=False)

    voiced_samples = sum(max(0, e - s) for s, e in segments)
    voiced_ms = round((voiced_samples * 1000.0) / sr, 1)
    if voiced_samples <= int(0.5 * sr):
        perf = _perf_metrics(len(waveform), sr, started_at)
        logger.info(
            "WINDOW_PERF | parent_id=%s | decision=%s | processing_ms=%.1f | audio_ms=%.1f | rtf=%.4f",
            parent_id,
            "rejected",
            perf["processing_ms"],
            perf["audio_ms"],
            perf["rtf"],
        )
        logger.info(
            "TIER1_VAD_REJECTED | parent_id=%s | reason=insufficient_voice | voiced_ms=%.1f | threshold_ms=500",
            parent_id, voiced_ms,
        )
        return {
            "tier1_vad": {"passed": False, "voiced_ms": voiced_ms},
            "tier2": {"passed": False, "confidence": 0.0},
            "tier3": {"passed": False, "score": 0.0, "closest_speaker_id": None, "closest_speaker_role": None},
            "distress": {"distress": False, "distress_score": 0.0, "distress_class": None},
            "decision": "rejected",
            "perf": perf,
        }

    logger.info("TIER1_VAD_PASSED | parent_id=%s | voiced_ms=%.1f | rms=%.6f", parent_id, voiced_ms, rms)

    # Tier 2 and Tier 3 both look only at the voiced audio (silence removed),
    # exactly as enrollment does.
    voiced = np.concatenate([waveform[s:e] for s, e in segments]).astype(np.float32, copy=False)
    speech_ok, speech_metrics = assess_speech_likeness(voiced, sr)
    yamnet_result = classify_audio_event_windowed(voiced, sr)
    category = yamnet_result.get("category", "uncertain")
    confidence = float(yamnet_result.get("confidence", 0.0))
    distress_result = distress_from_result(yamnet_result)

    logger.info(
        "TIER2_EVALUATION | parent_id=%s | speech_ok=%s | yamnet_category=%s | yamnet_confidence=%.4f"
        " | top_class=%s | distress_score=%.4f | flatness=%.4f | centroid_hz=%.1f",
        parent_id, speech_ok, category, confidence, yamnet_result.get("top_class"),
        distress_result["distress_score"], speech_metrics.get("spectral_flatness", 0.0),
        speech_metrics.get("spectral_centroid_hz", 0.0),
    )

    # Hard-reject clear non-speech (music/TV/singing, laughing/crying/coughing).
    # Crowd babble and uncertain audio still go to Tier 3: a confident familiar
    # match there is useful, but they never count as stranger evidence.
    hard_reject = (
        not speech_ok
        or category == "reject"
        or (category == "vocal_non_speech" and confidence >= 0.70)
    )

    if hard_reject:
        perf = _perf_metrics(len(waveform), sr, started_at)
        logger.info(
            "WINDOW_PERF | parent_id=%s | decision=%s | processing_ms=%.1f | audio_ms=%.1f | rtf=%.4f",
            parent_id,
            "discard_non_human_or_low_confidence",
            perf["processing_ms"],
            perf["audio_ms"],
            perf["rtf"],
        )
        logger.info(
            "TIER2_REJECTED | parent_id=%s | speech_ok=%s | yamnet_category=%s | yamnet_confidence=%.4f",
            parent_id, speech_ok, category, confidence,
        )
        return {
            "tier1_vad": {"passed": True, "voiced_ms": voiced_ms},
            "tier2": {"passed": False, "confidence": confidence, **yamnet_result, **speech_metrics},
            "tier3": {"passed": False, "score": 0.0, "closest_speaker_id": None, "closest_speaker_role": None},
            "distress": distress_result,
            "decision": "discard_non_human_or_low_confidence",
            "perf": perf,
        }

    logger.info("TIER2_PASSED | parent_id=%s | proceeding_to_tier3_speaker_match", parent_id)

    # Tier 3 runs only for windows that passed Tier 2.
    try:
        parent_candidates = load_parent_embeddings(db, parent_id)
        if not parent_candidates:
            # Nobody is enrolled yet: every voice would score 0 and look like a
            # stranger.  Hold until at least one family voice is enrolled.
            perf = _perf_metrics(len(waveform), sr, started_at)
            logger.info("TIER3_SETUP_REQUIRED | parent_id=%s | reason=no_enrolled_speakers", parent_id)
            return {
                "tier1_vad": {"passed": True, "voiced_ms": voiced_ms},
                "tier2": {"passed": True, "confidence": confidence, **yamnet_result, **speech_metrics},
                "tier3": {"passed": False, "score": 0.0, "closest_speaker_id": None,
                          "closest_speaker_role": None, "error": "no_enrolled_speakers"},
                "distress": distress_result,
                "decision": "setup_required",
                "perf": perf,
            }
        if INFERENCE_EMBED_MODE == "segments":
            # Legacy (pre-fix) behaviour: one embedding per VAD segment.
            query_embs = embed_waveform_chunks(waveform, sr, segments)
        else:
            query_embs = embed_waveform_chunks(voiced, sr, None, drop_short_tails=True)
    except Exception as exc:
        perf = _perf_metrics(len(waveform), sr, started_at)
        logger.error("TIER3_UNAVAILABLE | parent_id=%s | reason=%s", parent_id, str(exc))
        return {
            "tier1_vad": {"passed": True, "voiced_ms": voiced_ms},
            "tier2": {"passed": True, "confidence": confidence, **yamnet_result, **speech_metrics},
            "tier3": {
                "passed": False,
                "score": 0.0,
                "closest_speaker_id": None,
                "closest_speaker_role": None,
                "error": "tier3_unavailable",
            },
            "distress": distress_result,
            "decision": "tier3_unavailable",
            "perf": perf,
        }

    # Change 1: _score_against_candidates now returns (score, speaker_id, speaker_role).
    speaker_scores: Dict[str, float] = {}
    score, closest_speaker_id, closest_speaker_role = _score_against_candidates(
        parent_candidates, query_embs, parent_id, t_high_global=t_high, scores_out=speaker_scores
    )
    query_centroid = _compute_query_centroid(query_embs)

    logger.info(
        "TIER3_SCORED | parent_id=%s | score=%.4f | closest_speaker_id=%s | role=%s | t_high=%.4f | t_low=%.4f",
        parent_id, score, closest_speaker_id, closest_speaker_role, t_high, t_low,
    )
    perf = _perf_metrics(len(waveform), sr, started_at)
    logger.info(
        "WINDOW_PERF | parent_id=%s | decision=%s | processing_ms=%.1f | audio_ms=%.1f | rtf=%.4f",
        parent_id,
        "tier3_scored",
        perf["processing_ms"],
        perf["audio_ms"],
        perf["rtf"],
    )

    return {
        "tier1_vad": {"passed": True, "voiced_ms": voiced_ms},
        "tier2": {"passed": True, "confidence": confidence, **yamnet_result, **speech_metrics},
        "tier3": {
            "passed": True,
            "score": float(score),
            "closest_speaker_id": closest_speaker_id,
            "closest_speaker_role": closest_speaker_role,
            "speaker_scores": speaker_scores,
            "t_high": t_high,
            "t_low": t_low,
        },
        "distress": distress_result,
        "decision": "tier3_scored",
        "perf": perf,
        "_query_centroid": query_centroid,
    }


def append_frame(session: SessionState, frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32).reshape(-1)
    if session.inference_buffer is None:
        session.inference_buffer = frame
    else:
        session.inference_buffer = np.concatenate([session.inference_buffer, frame])

    infer_keep = int(WINDOW_SEC * session.sr)
    if session.inference_buffer.shape[0] > infer_keep:
        session.inference_buffer = session.inference_buffer[-infer_keep:]

    if session.clip_buffer is None:
        session.clip_buffer = frame.copy()
    else:
        session.clip_buffer = np.concatenate([session.clip_buffer, frame])

    clip_keep = int(ALERT_RING_SEC * session.sr)
    if session.clip_buffer.shape[0] > clip_keep:
        session.clip_buffer = session.clip_buffer[-clip_keep:]

    _feed_post_roll(session, frame)
    return session.inference_buffer


def pad_clip_silence(session: SessionState, n_samples: int) -> None:
    """Insert the silence the device did not upload (silent chunks it skipped), so the
    alert clip keeps real timing instead of jumping between speech chunks.
    Only the clip buffers are padded; the analysis buffer is unchanged."""
    n = min(int(n_samples), int(ALERT_RING_SEC * session.sr))
    if n <= 0:
        return
    zeros = np.zeros(n, dtype=np.float32)
    if session.clip_buffer is not None:
        session.clip_buffer = np.concatenate([session.clip_buffer, zeros])[-int(ALERT_RING_SEC * session.sr):]
    _feed_post_roll(session, zeros)


def start_post_roll(session: SessionState, clip_path: str, pre_alert: np.ndarray) -> None:
    if ALERT_POST_ROLL_SEC <= 0 or not clip_path:
        return
    session.post_roll_clip_path = clip_path
    session.post_roll_buffer = np.asarray(pre_alert, dtype=np.float32).reshape(-1).copy()
    session.post_roll_remaining = int(ALERT_POST_ROLL_SEC * session.sr)


def _feed_post_roll(session: SessionState, frame: np.ndarray) -> None:
    if not session.post_roll_clip_path or session.post_roll_buffer is None:
        return
    take = np.asarray(frame, dtype=np.float32).reshape(-1)[: session.post_roll_remaining]
    session.post_roll_buffer = np.concatenate([session.post_roll_buffer, take])
    session.post_roll_remaining -= take.shape[0]
    if session.post_roll_remaining > 0:
        return
    path = session.post_roll_clip_path
    waveform = session.post_roll_buffer
    session.post_roll_clip_path = None
    session.post_roll_buffer = None
    session.post_roll_remaining = 0
    try:
        tmp = f"{path}.post.tmp"
        sf.write(tmp, normalize_waveform(waveform), session.sr, format="WAV")
        os.replace(tmp, path)  # atomic: a parent playing the clip gets old or new, never half
        logger.info("ALERT_CLIP_POST_ROLL_SAVED | path=%s | seconds=%.1f", path, waveform.shape[0] / session.sr)
    except Exception as exc:
        logger.warning("ALERT_CLIP_POST_ROLL_FAILED | path=%s | reason=%s", path, exc)


def get_clip_buffer_waveform(session: SessionState, fallback: np.ndarray) -> np.ndarray:
    if session.clip_buffer is not None and session.clip_buffer.size > 0:
        return np.asarray(session.clip_buffer, dtype=np.float32).reshape(-1)
    return np.asarray(fallback, dtype=np.float32).reshape(-1)


def compute_and_store_enrollment_quality(db: Session, parent_id: str, speaker_id: str) -> Tuple[Optional[float], Optional[str]]:
    speaker = get_speaker(db, parent_id, speaker_id)
    sdir = _speaker_dir(parent_id, str(speaker.id))
    embs: List[np.ndarray] = []
    if os.path.isdir(sdir):
        for fname in sorted(os.listdir(sdir)):
            if not fname.endswith(".npy"):
                continue
            try:
                arr = np.asarray(np.load(os.path.join(sdir, fname)), dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if arr.shape[0] == config.EMBEDDING_DIM:
                embs.append(arr)

    if len(embs) < 2:
        # Consistency needs at least two embeddings; ask for another sample
        # rather than reporting a quality we cannot measure.
        speaker.quality_score = None
        speaker.quality_label = "pending"
        speaker.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(speaker)
        return None, "pending"

    pair_scores = [_cos(a, b) for a, b in combinations(embs, 2)]
    if not pair_scores:
        speaker.quality_score = None
        speaker.quality_label = None
        speaker.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(speaker)
        return None, None

    pair_arr = np.asarray(pair_scores, dtype=np.float32)
    median_similarity = float(np.median(pair_arr))
    lower_quartile = float(np.percentile(pair_arr, 25))
    spread = float(np.std(pair_arr))

    # Robust score favors consistency of most chunk pairs without over-weighting a few easy matches.
    robust_similarity = float((0.65 * median_similarity) + (0.35 * lower_quartile))

    if robust_similarity >= 0.74 and lower_quartile >= 0.62:
        label = "good"
    elif robust_similarity >= 0.52:
        label = "fair"
    elif len(embs) <= 3:
        # Small enrollment sets are too noisy for a strict poor label.
        label = "fair"
    else:
        label = "poor"

    # Guard against gaming the metric with monotone near-identical chunks (e.g., prolonged vowel only).
    if len(embs) >= 4 and median_similarity >= 0.93 and spread <= 0.015 and label == "good":
        label = "fair"

    speaker.quality_score = robust_similarity
    speaker.quality_label = label
    speaker.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(speaker)
    return robust_similarity, label


def should_hop(frame_samples: int, sr: int = 16000) -> bool:
    return frame_samples >= int(HOP_SEC * sr)


def save_alert_clip(parent_id: str, session: SessionState, waveform: np.ndarray) -> str:
    dirs = _ensure_parent_dirs(parent_id)
    alert_id = str(uuid.uuid4())
    adir = os.path.join(dirs["alerts"], alert_id)
    os.makedirs(adir, exist_ok=True)
    clip_path = os.path.join(adir, "clip.wav")
    tmp_clip_path = f"{clip_path}.tmp"
    metadata_path = os.path.join(adir, "metadata.json")
    metadata = {
        "segment_started_ms": int(session.stranger_segment_started_ms) if session.stranger_segment_started_ms else None,
        "segment_ended_ms": int(session.stranger_segment_ended_ms) if session.stranger_segment_ended_ms else None,
        "sample_rate": int(session.sr),
        "samples": int(np.asarray(waveform).reshape(-1).shape[0]),
    }
    try:
        # Raw chunks are no longer gain-normalized on arrival; level the clip for playback.
        waveform = normalize_waveform(np.asarray(waveform, dtype=np.float32).reshape(-1))
        sf.write(tmp_clip_path, waveform, session.sr, format="WAV")
        try:
            os.replace(tmp_clip_path, clip_path)
        except Exception as rename_exc:
            logger.error(
                "ALERT_CLIP_RENAME_FAILED | parent_id=%s | clip_path=%s | tmp_path=%s | reason=%s",
                parent_id,
                clip_path,
                tmp_clip_path,
                str(rename_exc),
            )
            raise
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    except Exception as exc:
        if os.path.exists(tmp_clip_path):
            try:
                os.remove(tmp_clip_path)
            except OSError:
                pass
        logger.error(
            "ALERT_CLIP_WRITE_FAILED | parent_id=%s | clip_path=%s | tmp_path=%s | reason=%s",
            parent_id,
            clip_path,
            tmp_clip_path,
            str(exc),
        )
        for path in (metadata_path, clip_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        try:
            if os.path.isdir(adir) and not os.listdir(adir):
                os.rmdir(adir)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail="failed_to_write_alert_clip") from exc
    return clip_path
