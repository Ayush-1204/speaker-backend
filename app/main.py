# A local .env file is optional: on Render the variables are set in the
# dashboard, and python-dotenv is only a dev dependency.
try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv()

import asyncio
import contextlib
import dataclasses
import logging
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
try:
    import librosa
except Exception:  # pragma: no cover - optional dependency in some environments
    librosa = None
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketState

from app.database.models import Device, DeviceRole
from app.database.session import SessionLocal, get_db, init_db
from app.utils import prd_services_db as svc
from app.utils.audio_preprocess import normalize_waveform
from app.utils.audio_decoder import decode_audio_chunk
from app.utils.verification_pipeline import enrollment_rejection_message, run_enroll_embedding
from app.utils import feature_extractor as feature_extractor_module
from app.utils import yamnet_classifier as yamnet_classifier_module
from app.utils import vad as vad_module
from app.utils import eval_capture
from app.services.detection_service import DetectionService
from app.utils.notification_worker import send_fcm_push
from app.services.websocket_manager import (
    publish_parent_event,
    _should_publish_device_status,
    _ws_set_loop,
    _ws_register_client,
    _ws_unregister_client,
    _WS_STATE_LOCK,
    _WS_CLIENTS_BY_PARENT,
    _WS_PUBLISHED_COUNTS,
    _WS_LAST_QUEUE_LAG_MS,
)
import app.services.websocket_manager as _ws_manager_module
from app.services.decision_engine import DecisionParams, decide
from app.services.alert_service import create_and_escalate_alert
from app.services.cache_service import (
    warmup_embeddings_to_redis as _warmup_embeddings_to_redis_fn,
    get_redis_client as _get_redis_client,
    _CACHE_WARMUP_SUMMARY,
)
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from app.core import security
from app.core.security import auth_rate_limit, enforce_rate_limit
from app.services import auth_tokens
from app.services import health_monitor
from app.services import child_safety

# Configure logging once, before anything logs (LOG_FORMAT=json for JSON lines).
from app.core.logging import configure_logging

configure_logging(json_format=(os.environ.get("LOG_FORMAT") or "").strip().lower() == "json")

# Module logger — must be defined before _lifespan which uses it.
logger = logging.getLogger(__name__)

# Application instance
@contextlib.asynccontextmanager
async def _lifespan(app_instance):
    """FastAPI lifespan: runs startup logic then yields control to the app."""
    security.validate_security_config()
    init_db()
    try:
        with SessionLocal() as _db:
            auth_tokens.import_legacy_refresh_tokens(_db)
    except Exception as exc:
        logger.warning("LEGACY_REFRESH_IMPORT_FAILED | reason=%s", exc)
    # Eagerly warm model singletons so first detect request does not pay load latency.
    try:
        feature_extractor_module._get_redimnet()
        logger.info("MODEL_WARMUP_OK | model=redimnet")
    except Exception as exc:
        logger.warning("MODEL_WARMUP_FAILED | model=redimnet | reason=%s", str(exc))

    try:
        yamnet_model = yamnet_classifier_module._get_yamnet_model()
        if yamnet_model is None:
            logger.warning("MODEL_WARMUP_SKIPPED | model=yamnet | reason=unavailable")
        else:
            logger.info("MODEL_WARMUP_OK | model=yamnet")
    except Exception as exc:
        logger.warning("MODEL_WARMUP_FAILED | model=yamnet | reason=%s", str(exc))

    try:
        silero_model, silero_utils = vad_module._get_silero()
        if silero_model is None or silero_utils is None:
            logger.warning("MODEL_WARMUP_SKIPPED | model=silero_vad | reason=unavailable")
        else:
            logger.info("MODEL_WARMUP_OK | model=silero_vad")
    except Exception as exc:
        logger.warning("MODEL_WARMUP_FAILED | model=silero_vad | reason=%s", str(exc))

    _warmup_embeddings_to_redis()
    # Importing silero_vad lowers torch to 1 thread; restore it after all models load.
    vad_module.restore_torch_threads()
    await asyncio.to_thread(_warmup_inference)
    watchdog = asyncio.create_task(_device_watchdog_loop())
    try:
        yield  # App handles requests here
    finally:
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watchdog


def _warmup_inference() -> None:
    """Run each model once so the first real chunk does not pay one-time costs
    (graph setup, memory allocation, kernel selection)."""
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    probe = (0.01 * rng.standard_normal(24000)).astype(np.float32)
    for name, fn in (
        ("vad", lambda: vad_module.get_speech_segments(probe)),
        ("yamnet", lambda: yamnet_classifier_module.classify_audio_event_windowed(probe)),
        ("redimnet", lambda: feature_extractor_module.embed_waveform_chunks(probe, 16000)),
    ):
        try:
            fn()
        except Exception as exc:
            logger.warning("WARMUP_INFERENCE_FAILED | model=%s | reason=%s", name, exc)
    logger.info("WARMUP_INFERENCE_DONE | ms=%.0f", (time.perf_counter() - started) * 1000)


async def _device_watchdog_loop() -> None:
    """Every minute, tell parents about monitored devices that went silent."""
    interval = int(os.environ.get("SAFEEAR_WATCHDOG_INTERVAL_SEC", "60"))
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(_run_device_watchdog_once)
        except Exception as exc:
            logger.warning("DEVICE_WATCHDOG_FAILED | reason=%s", exc)


_LAST_CLIP_PURGE = 0.0


def _run_device_watchdog_once() -> int:
    global _LAST_CLIP_PURGE
    evicted = svc.evict_idle_sessions()
    if evicted:
        logger.info("SESSIONS_EVICTED | count=%d", evicted)
    with SessionLocal() as db:
        # Hourly: delete alert recordings past the retention period.
        if time.monotonic() - _LAST_CLIP_PURGE > 3600:
            _LAST_CLIP_PURGE = time.monotonic()
            purged = svc.purge_old_alert_clips(db)
            if purged:
                logger.info("ALERT_CLIPS_PURGED | count=%d | retention_days=%d", purged, svc.CLIP_RETENTION_DAYS)
        return health_monitor.report_silent_devices(db)


app = FastAPI(lifespan=_lifespan)

# CORS only matters for browser clients (the mobile app does not need it).
# Production allows only explicitly configured origins; development allows any
# origin but never with credentials.
_cors_origins = [o.strip() for o in (os.environ.get("SAFEEAR_CORS_ORIGINS") or "").split(",") if o.strip()]
if not _cors_origins and not security.is_production():
    _cors_origins = ["*"]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

MAX_CHUNK_BYTES = int(os.environ.get("SAFEEAR_MAX_CHUNK_BYTES", str(512 * 1024)))
MAX_ENROLL_BYTES = int(os.environ.get("SAFEEAR_MAX_ENROLL_BYTES", str(10 * 1024 * 1024)))
DETECT_RATE_LIMIT_PER_SEC = int(os.environ.get("SAFEEAR_DETECT_RATE_LIMIT_PER_SEC", "10"))


async def _read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="upload_too_large")
    return data

now_ms = svc.now_ms

# Route-level threshold aliases kept for compatibility with existing tests and logs.
T_HIGH = svc.T_HIGH
T_LOW = svc.T_LOW
# Decision rules and their field-tuned defaults live in app/services/decision_engine.py
# (DecisionParams).  Change them only with evidence from scripts/eval/replay.py.
DECISION_PARAMS = DecisionParams.from_env()
STRANGER_MIN_TIER2_CONF = DECISION_PARAMS.stranger_min_tier2_conf
STRANGER_UNCERTAIN_CONF_FLOOR = DECISION_PARAMS.stranger_uncertain_conf_floor
FAMILIAR_GRACE_SEC = DECISION_PARAMS.familiar_grace_sec
FAMILIAR_HOLD_FLOOR = DECISION_PARAMS.familiar_hold_floor
FAMILIAR_GRACE_MARGIN = DECISION_PARAMS.familiar_grace_margin
FAMILIAR_GRACE_MIN_CONF = DECISION_PARAMS.familiar_grace_min_conf
FAMILIAR_RESET_MIN_STREAK = DECISION_PARAMS.familiar_reset_min_streak
SOFT_STRANGER_MARGIN = DECISION_PARAMS.soft_stranger_margin
SOFT_STRANGER_MIN_CONF = DECISION_PARAMS.soft_stranger_min_conf
FORCE_ALERT_SCORE = DECISION_PARAMS.force_alert_score
FORCE_ALERT_MIN_STREAK = DECISION_PARAMS.force_alert_min_streak
ALERT_TRIGGER_STREAK = DECISION_PARAMS.alert_trigger_streak
DEBOUNCE_SEC = DECISION_PARAMS.debounce_sec
SPEAKER_CHANGE_THRESHOLD = DECISION_PARAMS.speaker_change_threshold
STRANGER_PREROLL_SEC = float(os.environ.get("SAFEEAR_STRANGER_PREROLL_SEC", "5.0"))
# Opt-in experimental alert paths (off = April behaviour).
ENABLE_DISTRESS_ALERTS = svc.ENABLE_DISTRESS_DETECTION
DISTRESS_ALERT_THRESHOLD = float(os.environ.get("SAFEEAR_DISTRESS_ALERT_THRESHOLD", "0.65"))
DISTRESS_CONFIRM_WINDOWS = int(os.environ.get("SAFEEAR_DISTRESS_CONFIRM_WINDOWS", "2"))
DISTRESS_DEBOUNCE_SEC = int(os.environ.get("SAFEEAR_DISTRESS_DEBOUNCE_SEC", "60"))
CHILD_DRIFT_FLUSH_EVERY = int(os.environ.get("SAFEEAR_CHILD_DRIFT_FLUSH_EVERY", "20"))
# Idempotency cache for recently processed detect chunks (in-memory)
DETECT_CHUNK_IDEMPOTENCY_WINDOW_MS = int(os.environ.get("SAFEEAR_DETECT_CHUNK_IDEMPOTENCY_WINDOW_MS", "120000"))
_RECENT_DETECT_CHUNK_RESULTS: Dict[str, tuple] = {}
_RECENT_DETECT_CHUNK_RESULTS_LOCK = threading.Lock()


# WebSocket configuration constants
WS_CLIENT_QUEUE_MAX = int(os.environ.get("SAFEEAR_WS_CLIENT_QUEUE_MAX", "64"))
WS_PING_INTERVAL_SEC = float(os.environ.get("SAFEEAR_WS_PING_INTERVAL_SEC", "20.0"))


def _warmup_embeddings_to_redis() -> None:
    """Warm speaker embeddings into Redis on startup (best-effort)."""
    client = _get_redis_client()
    try:
        _warmup_embeddings_to_redis_fn(client)
    except Exception as exc:
        logger.warning("CACHE_WARMUP_FAILED | reason=%s", str(exc))


def _find_eer_threshold(familiar_scores: list, stranger_scores: list) -> tuple:
    """Compute the Equal Error Rate (EER) threshold from familiar/stranger score lists.

    Returns (threshold, eer_percent) where threshold minimises |FAR - FRR|.
    """
    all_scores = sorted(set(float(s) for s in familiar_scores + stranger_scores))
    best_threshold = 0.0
    best_diff = float("inf")
    best_eer = 1.0

    for t in all_scores:
        far = sum(1 for s in stranger_scores if float(s) >= t) / max(1, len(stranger_scores))
        frr = sum(1 for s in familiar_scores if float(s) < t) / max(1, len(familiar_scores))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_threshold = t
            best_eer = (far + frr) / 2.0

    return best_threshold, round(best_eer * 100.0, 2)


def _make_detect_chunk_cache_key(parent_id: str, device_id: str, chunk_id: Optional[str]) -> str:
    return f"detect:{parent_id}:{device_id}:{chunk_id}"


def _resolve_redis_url() -> str:
    """Resolve Redis URL using environment variables.

    Preference order: `REDIS_URL` -> `SAFEEAR_REDIS_URL` -> default localhost.
    """
    return os.environ.get("REDIS_URL") or os.environ.get("SAFEEAR_REDIS_URL") or "redis://localhost:6379/0"


def _get_detect_chunk_cached_response(parent_id: str, device_id: str, chunk_id: Optional[str]):
    if not chunk_id:
        return None
    key = _make_detect_chunk_cache_key(parent_id, device_id, chunk_id)
    now = svc.now_ms()
    stale_before = now - DETECT_CHUNK_IDEMPOTENCY_WINDOW_MS

    with _RECENT_DETECT_CHUNK_RESULTS_LOCK:
        # Lightweight in-memory pruning keeps cache bounded over long uptime.
        stale_keys = [k for k, (ts, _resp) in _RECENT_DETECT_CHUNK_RESULTS.items() if ts < stale_before]
        for stale_key in stale_keys:
            _RECENT_DETECT_CHUNK_RESULTS.pop(stale_key, None)

        record = _RECENT_DETECT_CHUNK_RESULTS.get(key)
        if record is None:
            return None
        ts, response = record
        if ts < stale_before:
            _RECENT_DETECT_CHUNK_RESULTS.pop(key, None)
            return None
        return dict(response)


def _remember_detect_chunk_response(
    parent_id: str,
    device_id: str,
    chunk_id: Optional[str],
    response_payload: Dict[str, Any],
) -> None:
    if not chunk_id:
        return
    key = _make_detect_chunk_cache_key(parent_id, device_id, chunk_id)
    with _RECENT_DETECT_CHUNK_RESULTS_LOCK:
        _RECENT_DETECT_CHUNK_RESULTS[key] = (now_ms(), dict(response_payload))


def _log_pipeline_decision(
    stage: str,
    device_id: str,
    parent_id: str,
    result: str,
    score: Optional[float],
    streak: int,
    latency_ms: float,
) -> None:
    logger.info(
        "PIPELINE_DECISION | stage=%s | parent_id=%s | device_id=%s | result=%s | score=%s | streak=%s | latency_ms=%.1f",
        stage,
        parent_id,
        device_id,
        result,
        "none" if score is None else f"{float(score):.4f}",
        streak,
        latency_ms,
    )

DATA_ROOT = (os.environ.get("SAFEEAR_DATA_ROOT", os.path.join("app", "data")) or "").strip() or os.path.join("app", "data")
TEMP_DIR = os.path.join(DATA_ROOT, "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)

AVATAR_MAX_SIZE_BYTES = int(os.environ.get("SAFEEAR_AVATAR_MAX_SIZE_BYTES", str(5 * 1024 * 1024)))
ALLOWED_AVATAR_CONTENT_TYPES = {"image/jpeg", "image/png"}


def _speaker_avatar_url(request: Request, speaker_id: str) -> str:
    return str(request.url_for("get_speaker_avatar", speaker_id=str(speaker_id)))


def _speaker_payload(row, request: Request, parent_id: str) -> Dict[str, Any]:
    avatar_exists = svc.get_speaker_avatar_path(parent_id, row.id) is not None
    return {
        "id": str(row.id),
        "parent_id": str(row.parent_id),
        "display_name": row.display_name,
        "sample_count": row.sample_count,
        "quality_score": row.quality_score,
        "quality_label": row.quality_label,
        # Change 1: speaker role and per-speaker threshold
        "role": getattr(row, "role", "familiar_adult") or "familiar_adult",
        "t_high_override": getattr(row, "t_high_override", None),
        # Change 4: confusability
        "confusability_score": getattr(row, "confusability_score", None),
        "confusability_label": getattr(row, "confusability_label", None),
        "profile_image_url": _speaker_avatar_url(request, row.id) if avatar_exists else None,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


class GoogleAuthRequest(BaseModel):
    id_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


class EmailRegisterRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None


class EmailLoginRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


SPEAKER_ROLES = {"familiar_adult", "child_self"}


class RenameSpeakerRequest(BaseModel):
    display_name: Optional[str] = None
    role: Optional[str] = None


class FlagFamiliarRequest(BaseModel):
    display_name: str
    speaker_id: Optional[str] = None


class DeviceMonitoringPatchRequest(BaseModel):
    monitoring_enabled: bool


class DeviceHeartbeatRequest(BaseModel):
    battery_percent: Optional[int] = None
    is_online: Optional[bool] = None
    monitoring_enabled: Optional[bool] = None


class DeviceMonitoringAckRequest(BaseModel):
    monitoring_enabled: bool
    is_online: Optional[bool] = None
    battery_percent: Optional[int] = None


class TestFireAlertRequest(BaseModel):
    device_id: str
    confidence_score: float = 0.05
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class CalibrateThresholdsRequest(BaseModel):
    familiar_scores: List[float]
    stranger_scores: List[float]


class DetectLocationRequest(BaseModel):
    device_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    battery_percent: Optional[int] = None
    battery: Optional[int] = None
    accuracy_m: Optional[float] = None
    location_ts_ms: Optional[int] = None


def _extract_bearer(auth_header: Optional[str]) -> str:
    if not auth_header:
        raise HTTPException(status_code=401, detail="missing_authorization_header")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="invalid_authorization_header")
    return auth_header.split(" ", 1)[1].strip()


def _decode_audio_chunk(raw: bytes) -> Tuple[np.ndarray, int, str]:
    return decode_audio_chunk(raw)


def get_principal(
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Authenticate a parent session token or a device token.

    Returns {"parent_id", "device_id", "kind"}; device_id is set only for device tokens.
    """
    claims = security.parse_token(_extract_bearer(authorization))
    parent_id = claims.get("sub")
    if not parent_id:
        raise HTTPException(status_code=401, detail="invalid_token_subject")
    kind = claims.get("typ", "access")
    if kind == "device":
        device_id = str(claims.get("did") or "")
        try:
            device = svc.get_device(db, str(parent_id), device_id)
        except HTTPException as exc:
            raise HTTPException(status_code=401, detail="device_not_found") from exc
        if device.role != DeviceRole.child_device or int(device.access_token_version or 0) != int(claims.get("ver", -1)):
            raise HTTPException(status_code=401, detail="device_token_revoked")
        return {"parent_id": str(parent_id), "device_id": device_id, "kind": "device"}
    if kind != "access":
        raise HTTPException(status_code=401, detail="invalid_token_type")
    try:
        svc.get_parent(db, parent_id)
    except HTTPException as exc:
        raise HTTPException(status_code=401, detail="parent_not_found") from exc
    return {"parent_id": str(parent_id), "device_id": None, "kind": "parent"}


def get_current_parent(principal: Dict[str, Any] = Depends(get_principal)) -> Dict[str, Any]:
    """Parent-only routes: alerts, speakers, monitoring control, device management."""
    if principal["kind"] != "parent":
        raise HTTPException(status_code=403, detail="parent_session_required")
    return principal


def _require_device_access(principal: Dict[str, Any], device_id: str) -> None:
    """A device token may only act on its own device."""
    if principal["kind"] == "device" and str(device_id) != str(principal["device_id"]):
        raise HTTPException(status_code=403, detail="device_mismatch")


def _parent_id_from_access_token(token: str) -> str:
    claims = security.parse_token(token)
    parent_id = claims.get("sub")
    if not parent_id:
        raise HTTPException(status_code=401, detail="invalid_token_subject")
    if claims.get("typ", "access") != "access":
        raise HTTPException(status_code=403, detail="parent_session_required")
    return str(parent_id)


def _token_response(parent, refresh_token: str) -> Dict[str, Any]:
    return {
        "access_token": security.make_access_token(str(parent.id)),
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": security.ACCESS_TOKEN_TTL_SEC,
        "parent": {
            "id": str(parent.id),
            "email": parent.email,
            "display_name": parent.display_name,
        },
    }


def _verify_google_like_token(id_token: str) -> Dict[str, Any]:
    # Dev token bypass: local testing only, off by default and never in production.
    if id_token.startswith("dev:"):
        if not security.dev_tokens_allowed():
            raise HTTPException(status_code=401, detail="dev_token_disabled")
        dev_sub = id_token.split(":", 1)[1].strip()
        if not dev_sub:
            raise HTTPException(status_code=401, detail="invalid_dev_token")
        return {"sub": dev_sub, "email": f"{dev_sub}@dev.local", "name": dev_sub}

    # Collect configured Google client IDs.
    raw_client_ids = []
    for key in ("GOOGLE_CLIENT_ID", "GOOGLE_WEB_CLIENT_ID", "GOOGLE_ANDROID_CLIENT_ID", "GOOGLE_CLIENT_IDS"):
        value = os.environ.get(key)
        if value:
            raw_client_ids.append(value)

    allowed_client_ids = []
    for raw in raw_client_ids:
        allowed_client_ids.extend([part.strip() for part in raw.split(",") if part.strip()])

    allowed_client_ids = list(dict.fromkeys(allowed_client_ids))

    if not allowed_client_ids:
        logger.warning("GOOGLE_AUTH_SKIPPED | reason=no_google_client_id_configured")
        raise HTTPException(status_code=401, detail="missing_google_client_id")

    try:
        import importlib

        grequests = importlib.import_module("google.auth.transport.requests")
        gid = importlib.import_module("google.oauth2.id_token")
        last_error = None
        info = None
        for client_id in allowed_client_ids:
            try:
                info = gid.verify_oauth2_token(id_token, grequests.Request(), audience=client_id)
                break
            except Exception as exc:
                last_error = exc
                continue

        if info is None:
            raise HTTPException(status_code=401, detail="invalid_google_id_token") from last_error

        aud = str(info.get("aud", "")).strip()
        if aud not in allowed_client_ids:
            logger.warning(
                "GOOGLE_AUTH_AUDIENCE_MISMATCH | allowed_count=%d",
                len(allowed_client_ids),
            )
            raise HTTPException(status_code=401, detail="invalid_google_audience")
        return {"sub": info.get("sub"), "email": info.get("email"), "name": info.get("name")}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("GOOGLE_AUTH_VERIFY_FAILED | reason=%s", str(exc))
        raise HTTPException(status_code=401, detail="invalid_google_id_token") from exc


def _send_password_reset_email(to_email: str, reset_link: str) -> None:
    sendgrid_api_key = os.environ.get("SENDGRID_API_KEY")
    from_email = os.environ.get("SENDGRID_FROM_EMAIL", "alerts@safeear.app")
    if not sendgrid_api_key:
        logger.info("PASSWORD_RESET_EMAIL_SKIPPED | reason=sendgrid_not_configured")
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject="SafeEar password reset",
            html_content=(
                "<p>You requested a SafeEar password reset.</p>"
                f"<p><a href=\"{reset_link}\">Reset password</a></p>"
                "<p>If you didn't request this, you can ignore this email.</p>"
            ),
        )
        SendGridAPIClient(sendgrid_api_key).send(message)
    except Exception as exc:
        logger.warning("PASSWORD_RESET_EMAIL_FAILED | reason=%s", str(exc))


def _dt_to_epoch_ms(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def _alert_response_payload(row: Any) -> Dict[str, Any]:
    return {
        "id": str(row.id),
        "parent_id": str(row.parent_id),
        "device_id": str(row.device_id),
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "timestamp_ms": _dt_to_epoch_ms(row.timestamp),
        "confidence_score": row.confidence_score,
        "audio_clip_path": row.audio_clip_path,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "lat": row.latitude,
        "lng": row.longitude,
        "alert_type": getattr(row, "alert_type", None) or "unenrolled_contact",
        "acknowledged_at": row.acknowledged_at.isoformat() if row.acknowledged_at else None,
        "acknowledged_at_ms": _dt_to_epoch_ms(row.acknowledged_at),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "created_at_ms": _dt_to_epoch_ms(row.created_at),
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "updated_at_ms": _dt_to_epoch_ms(row.updated_at),
    }

def _device_payload(row: Device) -> Dict[str, Any]:
    """Build device response with real telemetry and location data.
    
    Returns null for unknown battery (not 0).
    Includes location coordinates and activity timestamp for live location feature.
    """
    last_seen = None
    if row.last_activity_at:
        last_seen = row.last_activity_at.isoformat() if hasattr(row.last_activity_at, 'isoformat') else str(row.last_activity_at)
    
    location_updated_at = None
    if row.last_location_ts:
        location_updated_at = row.last_location_ts.isoformat() if hasattr(row.last_location_ts, 'isoformat') else str(row.last_location_ts)
    
    return {
        "id": str(row.id),
        "parent_id": str(row.parent_id),
        "installation_id": row.installation_id,
        "device_name": row.device_name,
        "role": row.role.value,
        "battery_percent": row.battery_percent,  # null if unknown, not 0
        "is_online": svc.get_effective_online(row),
        "monitoring_enabled": bool(row.monitoring_enabled),
        "last_seen_at": last_seen,
        "latitude": row.last_location_lat,
        "longitude": row.last_location_lon,
        "location_updated_at": location_updated_at,
    }


@app.post("/auth/google", dependencies=[Depends(auth_rate_limit)])
def auth_google(body: GoogleAuthRequest, db: Session = Depends(get_db)):
    info = _verify_google_like_token(body.id_token)
    sub = info.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="google_sub_missing")

    parent = svc.upsert_parent(db, str(sub), info.get("email"), info.get("name"))
    return _token_response(parent, auth_tokens.issue_refresh_token(db, str(parent.id)))


@app.post("/auth/register-email", dependencies=[Depends(auth_rate_limit)])
def auth_register_email(body: EmailRegisterRequest, db: Session = Depends(get_db)):
    parent = svc.register_parent_with_email(db, body.email, body.password, body.display_name)
    return _token_response(parent, auth_tokens.issue_refresh_token(db, str(parent.id)))


@app.post("/auth/login-email", dependencies=[Depends(auth_rate_limit)])
def auth_login_email(body: EmailLoginRequest, db: Session = Depends(get_db)):
    parent = svc.login_parent_with_email(db, body.email, body.password)
    return _token_response(parent, auth_tokens.issue_refresh_token(db, str(parent.id)))


@app.post("/auth/forgot-password", dependencies=[Depends(auth_rate_limit)])
def auth_forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    # Per-address limit so the endpoint cannot be used to flood someone's inbox.
    enforce_rate_limit(f"forgot:{(body.email or '').strip().lower()}", 5, 3600.0)
    reset_token = svc.create_password_reset_token(db, body.email, ttl_sec=3600)

    reset_base_url = os.environ.get("SAFEEAR_PASSWORD_RESET_URL")
    if reset_token and reset_base_url:
        reset_link = f"{reset_base_url}?token={reset_token}"
        _send_password_reset_email(body.email, reset_link)

    # Identical response whether or not the address is registered.
    response: Dict[str, Any] = {
        "status": "ok",
        "message": "If this email is registered, a reset link has been issued.",
    }
    if reset_token and not security.is_production() and os.environ.get("SAFEEAR_DEBUG_RETURN_RESET_TOKEN", "false").lower() == "true":
        response["reset_token"] = reset_token
    return response


@app.post("/auth/reset-password", dependencies=[Depends(auth_rate_limit)])
def auth_reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    parent_id = svc.reset_password_with_token(db, body.token, body.new_password)
    if parent_id:
        # A password reset ends every existing session.
        auth_tokens.revoke_all_for_parent(db, str(parent_id))
    return {"status": "ok", "message": "password_updated"}


@app.post("/auth/refresh", dependencies=[Depends(auth_rate_limit)])
def auth_refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    parent_id, refresh_token = auth_tokens.rotate_refresh_token(db, body.refresh_token)
    return {
        "access_token": security.make_access_token(parent_id),
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": security.ACCESS_TOKEN_TTL_SEC,
    }


@app.api_route("/auth/logout", methods=["POST", "DELETE"])
def auth_logout(
    body: Optional[RefreshRequest] = None,
    refresh_token: Optional[str] = None,
    db: Session = Depends(get_db),
):
    # Some mobile clients cannot reliably send a JSON body with DELETE.
    token = (body.refresh_token if body else None) or refresh_token
    if not token:
        raise HTTPException(status_code=422, detail="refresh_token_required")

    auth_tokens.revoke_refresh_token(db, token)
    return {"status": "ok"}


class DeleteAccountRequest(BaseModel):
    confirm: str


@app.delete("/auth/account")
def delete_account(body: DeleteAccountRequest, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    """Permanently delete the account and every recording, voiceprint and device."""
    if body.confirm != "DELETE":
        raise HTTPException(status_code=422, detail='confirm must be "DELETE"')
    svc.delete_parent_account(db, current["parent_id"])
    logger.info("ACCOUNT_DELETED | parent_id=%s", current["parent_id"])
    return {"status": "deleted"}


@app.post("/enroll/speaker")
async def enroll_speaker(
    request: Request,
    display_name: str = Form(...),
    speaker_id: Optional[str] = Form(None),
    # Change 1: accept speaker role for 3-category decision logic
    # familiar_adult | child_self.  Omitted = keep the speaker's current role
    # (new speakers default to familiar_adult), so adding a sample never resets it.
    role: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    upload = file or audio
    if upload is None:
        raise HTTPException(status_code=422, detail="missing_audio_file")

    # Validate role
    if role is not None and role not in SPEAKER_ROLES:
        raise HTTPException(status_code=422, detail=f"invalid_role: must be one of {sorted(SPEAKER_ROLES)}")

    parent_id = current["parent_id"]
    # For updates, ensure the target speaker exists before processing audio.
    speaker = svc.get_speaker(db, parent_id, speaker_id) if speaker_id else None

    data = await _read_upload_limited(upload, MAX_ENROLL_BYTES)
    os.makedirs(TEMP_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as tmp:
        tmp.write(data)
        temp_path = tmp.name

    try:
        embs, stage_info = await run_in_threadpool(run_enroll_embedding, temp_path)
    except ValueError as exc:
        logger.warning(f"ENROLL_SPEAKER_FAILED | parent_id={parent_id} | reason={str(exc)}")
        raise HTTPException(
            status_code=400,
            detail={"code": str(exc), "message": enrollment_rejection_message(str(exc))},
        ) from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    # Only create a new speaker after enrollment audio is validated and embeddings are extracted.
    if speaker is None:
        speaker = svc.create_speaker(db, parent_id, display_name)

    if role is not None and speaker.role != role:
        speaker.role = role
        db.commit()
        db.refresh(speaker)
        svc.invalidate_gallery(parent_id)

    for idx, emb in enumerate(embs):
        emb_arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb_arr.shape[0] != feature_extractor_module.config.EXPECTED_EMBEDDING_DIM:
            logger.error(
                "ENROLL_EMBEDDING_DIM_MISMATCH | parent_id=%s | speaker_id=%s | index=%s | expected=%s | got=%s",
                parent_id,
                str(speaker.id),
                idx,
                feature_extractor_module.config.EXPECTED_EMBEDDING_DIM,
                emb_arr.shape[0],
            )
            raise HTTPException(
                status_code=400,
                detail=f"dim mismatch: expected {feature_extractor_module.config.EXPECTED_EMBEDDING_DIM}, got {emb_arr.shape[0]}",
            )

    for emb in embs:
        svc.save_speaker_embedding(db, parent_id, str(speaker.id), emb)

    # Compute and store speaker quality metrics
    quality_score, quality_label = svc.compute_and_store_enrollment_quality(db, parent_id, str(speaker.id))

    # Trigger auto-calibration when enough samples are available (Change 5)
    try:
        svc.auto_calibrate_thresholds(db, parent_id)
    except Exception as _cal_exc:
        logger.warning("AUTO_CALIBRATE_ERROR | parent_id=%s | reason=%s", parent_id, _cal_exc)

    # Confusability check (Change 4)
    confusability_warnings = []
    try:
        conflicts = svc.check_enrollment_confusability(db, parent_id, str(speaker.id))
        for conflicting_speaker_id, sim in conflicts:
            confusability_warnings.append({
                "conflicting_speaker_id": conflicting_speaker_id,
                "similarity": round(float(sim), 4),
                "warning": (
                    "This speaker's voice is very close to an existing enrolled speaker. "
                    "The system may confuse them during monitoring."
                ),
            })
    except Exception as _conf_exc:
        logger.warning("CONFUSABILITY_CHECK_ERROR | parent_id=%s | reason=%s", parent_id, _conf_exc)

    # Compute samples_needed for this speaker (Change 2)
    current_speaker = svc.get_speaker(db, parent_id, str(speaker.id))
    sample_count = getattr(current_speaker, "sample_count", 0) or 0
    samples_needed = max(0, 5 - sample_count)
    enrollment_complete = samples_needed == 0

    speaker_payload = _speaker_payload(speaker, request, parent_id)
    refreshed_rows = svc.list_speakers(db, parent_id)
    refreshed_items = [_speaker_payload(row, request, parent_id) for row in refreshed_rows]
    response = {
        "status": "enrolled",
        # Current schema.
        "speaker": speaker_payload,
        # Backward compatibility for legacy clients.
        "speakerResponse": speaker_payload,
        "speaker_id": str(speaker.id),
        "display_name": speaker.display_name,
        "samples_saved": len(embs),
        "embedding_dim": int(embs[0].shape[0]),
        "samples_needed": samples_needed,
        "enrollment_complete": enrollment_complete,
        "items": refreshed_items,
        "stages": stage_info,
        "quality_score": quality_score,
        "quality_label": quality_label,
    }
    if quality_label == "poor":
        response["warning"] = "Low enrollment quality. Try recording in a quieter space, speaking naturally, or adding more samples."
    if confusability_warnings:
        response["confusability_warnings"] = confusability_warnings
    return response


@app.get("/enroll/speakers")
def get_enrolled_speakers(request: Request, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    rows = svc.list_speakers(db, current["parent_id"])
    return {
        "items": [_speaker_payload(row, request, current["parent_id"]) for row in rows]
    }


@app.patch("/enroll/speakers/{speaker_id}")
def patch_speaker(
    request: Request,
    speaker_id: str,
    body: RenameSpeakerRequest,
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    if body.role is not None and body.role not in SPEAKER_ROLES:
        raise HTTPException(status_code=422, detail=f"invalid_role: must be one of {sorted(SPEAKER_ROLES)}")
    row = svc.get_speaker(db, current["parent_id"], speaker_id)
    if body.display_name and body.display_name.strip():
        row = svc.rename_speaker(db, current["parent_id"], speaker_id, body.display_name.strip())
    if body.role is not None and row.role != body.role:
        row.role = body.role
        db.commit()
        db.refresh(row)
        svc.invalidate_gallery(current["parent_id"])
    speaker_payload = _speaker_payload(row, request, current["parent_id"])
    return {
        "status": "ok",
        "speaker": speaker_payload,
        "speakerResponse": speaker_payload,
    }


@app.post("/enroll/speakers/{speaker_id}/avatar")
async def upload_speaker_avatar(
    request: Request,
    speaker_id: str,
    profile_image: UploadFile = File(...),
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    content_type = (profile_image.content_type or "").lower().strip()
    if content_type not in ALLOWED_AVATAR_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="unsupported_profile_image_type")

    raw = await profile_image.read(AVATAR_MAX_SIZE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=422, detail="empty_profile_image")
    if len(raw) > AVATAR_MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="profile_image_too_large")

    speaker = svc.get_speaker(db, current["parent_id"], speaker_id)
    svc.save_speaker_avatar(db, current["parent_id"], speaker_id, raw, content_type)
    db.refresh(speaker)

    return _speaker_payload(speaker, request, current["parent_id"])


@app.get("/enroll/speakers/{speaker_id}/avatar", name="get_speaker_avatar")
def get_speaker_avatar(
    speaker_id: str,
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    svc.get_speaker(db, current["parent_id"], speaker_id)
    avatar_path = svc.get_speaker_avatar_path(current["parent_id"], speaker_id)
    if avatar_path is None:
        raise HTTPException(status_code=404, detail="profile_image_not_found")
    media_type = "image/png" if avatar_path.lower().endswith(".png") else "image/jpeg"
    return FileResponse(avatar_path, media_type=media_type)


@app.delete("/enroll/speakers/{speaker_id}")
def remove_speaker(speaker_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    svc.delete_speaker(db, current["parent_id"], speaker_id)
    return {"status": "deleted", "speaker_id": speaker_id}


@app.post("/detect/location")
def detect_location(body: DetectLocationRequest, current=Depends(get_principal), db: Session = Depends(get_db)):
    _require_device_access(current, body.device_id)
    lat = body.latitude if body.latitude is not None else body.lat
    lon = body.longitude if body.longitude is not None else body.lng
    battery_percent = body.battery_percent if body.battery_percent is not None else body.battery
    row = svc.update_device_location(db, current["parent_id"], body.device_id, lat, lon)
    if lat is not None and lon is not None:
        row = child_safety.store_fix(db, row, lat, lon, body.accuracy_m, body.location_ts_ms)
    if battery_percent is not None:
        row = svc.update_device_battery(db, current["parent_id"], body.device_id, battery_percent)
    else:
        logger.warning(
            "DEVICE_BATTERY_MISSING | endpoint=/detect/location | parent_id=%s | device_id=%s",
            current["parent_id"],
            body.device_id,
        )
    svc.update_session_location(current["parent_id"], body.device_id, lat, lon)
    if _should_publish_device_status(current["parent_id"], body.device_id):
        publish_parent_event(current["parent_id"], "device_status_changed", _device_payload(row))
    return {"status": "ok", "device_id": body.device_id, "latitude": lat, "longitude": lon}


STREAM_GAP_MS = int(os.environ.get("SAFEEAR_STREAM_GAP_MS", "3000"))
STALE_CHUNK_MS = int(os.environ.get("SAFEEAR_STALE_CHUNK_MS", "30000"))


def _apply_stream_order(
    parent_id: str,
    device_id: str,
    seq: Optional[int],
    t_start_ms: Optional[int],
    sent_at_ms: Optional[int],
    n_samples: int,
    sr: int,
    stream_id: Optional[str] = None,
    silent_skipped: int = 0,
) -> Optional[Dict[str, Any]]:
    """Enforce chunk ordering for clients that send seq / t_start_ms.

    Returns a response to send instead of processing (dropped chunk), or None to
    continue.  A gap in the stream resets the analysis context so a stranger
    streak never spans audio that was not heard back-to-back.  Clients that send
    neither field keep the previous behaviour.
    """
    if seq is None and t_start_ms is None:
        return None
    # Both times come from the device clock, so their difference is reliable even
    # when the device and server clocks disagree.
    if t_start_ms is not None and sent_at_ms is not None and sent_at_ms - t_start_ms > STALE_CHUNK_MS:
        logger.info("STREAM_STALE_CHUNK_DROPPED | device_id=%s | seq=%s | age_ms=%s", device_id, seq, sent_at_ms - t_start_ms)
        return {"status": "stale_dropped", "seq": seq}

    session = svc.get_or_create_session(parent_id, device_id)
    # A new stream id means the device restarted capture (app restart, new
    # recording session): its seq starts over, so begin from a clean context.
    if stream_id and stream_id != session.stream_id:
        if session.stream_id is not None:
            logger.info("STREAM_RESTART | device_id=%s | stream_id=%s", device_id, stream_id)
            svc.reset_stream(session)
        session.stream_id = stream_id
        session.last_seq = None
        session.last_chunk_end_ms = None

    if seq is not None and session.last_seq is not None and seq <= session.last_seq:
        logger.info("STREAM_OUT_OF_ORDER_DROPPED | device_id=%s | seq=%s | last_seq=%s", device_id, seq, session.last_seq)
        return {"status": "out_of_order_dropped", "seq": seq, "last_seq": session.last_seq}

    # The device may skip uploading chunks it measured as silent (the server would
    # reject them anyway).  Silence was still heard, so it is not a gap; only
    # audio that was lost (dropped, stale, not captured) resets the context.
    silent_skipped = max(0, int(silent_skipped or 0))
    silent_ms = silent_skipped * (n_samples * 1000) // max(1, sr)
    gap = False
    if seq is not None and session.last_seq is not None and seq != session.last_seq + 1:
        gap = True
    if (
        t_start_ms is not None
        and session.last_chunk_end_ms is not None
        and t_start_ms - session.last_chunk_end_ms - silent_ms > STREAM_GAP_MS
    ):
        gap = True
    if gap:
        logger.info(
            "STREAM_GAP_RESET | device_id=%s | seq=%s | last_seq=%s | t_start_ms=%s | last_end_ms=%s",
            device_id, seq, session.last_seq, t_start_ms, session.last_chunk_end_ms,
        )
        svc.reset_stream(session)
    elif t_start_ms is not None and session.last_chunk_end_ms is not None:
        # Silence the device heard but did not upload: keep it in the alert clip.
        pad_ms = int(t_start_ms) - int(session.last_chunk_end_ms)
        if pad_ms > 100:
            svc.pad_clip_silence(session, pad_ms * sr // 1000)

    if seq is not None:
        session.last_seq = seq
    if t_start_ms is not None:
        session.last_chunk_end_ms = int(t_start_ms + (n_samples * 1000) // max(1, sr))
    return None


def _report_scoring_unavailable(parent_id: str) -> None:
    try:
        with SessionLocal() as db:
            health_monitor.report_scoring_unavailable(db, parent_id)
    except Exception as exc:
        logger.warning("SCORING_UNAVAILABLE_REPORT_FAILED | parent_id=%s | reason=%s", parent_id, exc)


@app.post("/detect/chunk")
async def detect_chunk(
    device_id: str = Form(...),
    chunk_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    battery_percent: Optional[int] = Form(None),
    battery: Optional[int] = Form(None),
    seq: Optional[int] = Form(None),
    t_start_ms: Optional[int] = Form(None),
    sent_at_ms: Optional[int] = Form(None),
    stream_id: Optional[str] = Form(None),
    silent_skipped: int = Form(0),
    current=Depends(get_principal),
    db: Session = Depends(get_db),
):
    upload = file or audio
    if upload is None:
        raise HTTPException(status_code=422, detail="missing_audio_chunk")

    _require_device_access(current, device_id)
    enforce_rate_limit(f"detect:{device_id}", DETECT_RATE_LIMIT_PER_SEC, 1.0)
    raw = await _read_upload_limited(upload, MAX_CHUNK_BYTES)
    # Decoding and model inference run on a worker thread so the event loop keeps
    # serving other devices, websockets and heartbeats meanwhile.
    return await run_in_threadpool(
        _detect_chunk_sync, raw, upload, device_id, chunk_id, latitude, longitude,
        battery_percent, battery, seq, t_start_ms, sent_at_ms, stream_id, current, db, silent_skipped,
    )


def _detect_chunk_sync(raw, upload, device_id, chunk_id, latitude, longitude, battery_percent, battery,
                       seq, t_start_ms, sent_at_ms, stream_id, current, db, silent_skipped=0) -> Dict[str, Any]:
    # Chunks from one device are processed one at a time (they share session state).
    with svc.device_lock(current["parent_id"], device_id):
        return _detect_chunk_locked(raw, upload, device_id, chunk_id, latitude, longitude, battery_percent,
                                    battery, seq, t_start_ms, sent_at_ms, stream_id, current, db, silent_skipped)


def _detect_chunk_locked(raw, upload, device_id, chunk_id, latitude, longitude, battery_percent, battery,
                         seq, t_start_ms, sent_at_ms, stream_id, current, db, silent_skipped=0) -> Dict[str, Any]:
    parent_id = current["parent_id"]
    normalized_chunk_id = (chunk_id or "").strip() or None
    if normalized_chunk_id:
        cached = _get_detect_chunk_cached_response(parent_id, device_id, normalized_chunk_id)
        if cached is not None:
            cached["idempotent_replay"] = True
            return cached

    # Ownership is checked before any audio is decoded.
    device = svc.get_device(db, parent_id, device_id)
    if device.role != DeviceRole.child_device:
        raise HTTPException(status_code=403, detail="only_child_devices_can_stream_audio")
    if not device.monitoring_enabled:
        # Monitoring switched off by the parent: drop the audio unprocessed.
        return {"status": "monitoring_disabled", "monitoring_enabled": False}
    if child_safety.check_parent_nearby_cached(db, parent_id, device):
        # Parent-nearby rule: no detection while a parent phone is close by.
        svc.update_device_activity(db, parent_id, device_id, device=device)
        reported = battery_percent if battery_percent is not None else battery
        if reported is not None:
            svc.update_device_battery(db, parent_id, device_id, reported, device=device)
        return {
            "status": "paused_parent_nearby",
            "monitoring_enabled": True,
            "retry_after_ms": child_safety.PROXIMITY_CACHE_MS,
        }

    if not raw:
        try:
            svc.update_device_activity(db, parent_id, device_id)
        except Exception:
            pass
        logger.warning(
            "DETECT_CHUNK_EMPTY | parent_id=%s | device_id=%s | filename=%s | content_type=%s | bytes=0 "
            "| DIAGNOSTIC: Possible audio capture failure on child - check audio permissions and initialization",
            parent_id,
            device_id,
            getattr(upload, "filename", None),
            getattr(upload, "content_type", None),
        )
        empty_count, threshold_exceeded = svc.track_empty_chunk(device_id)
        if threshold_exceeded:
            logger.error(
                "DETECT_CHUNK_FAILURE | parent_id=%s | device_id=%s | empty_chunk_count=%d | threshold=%d "
                "| ACTION: Audio capture failure detected - child device may have permission issues, audio muting, or initialization problem",
                parent_id,
                device_id,
                empty_count,
                svc._EMPTY_CHUNK_THRESHOLD,
            )
        if _should_publish_device_status(parent_id, device_id):
            try:
                row = svc.get_device(db, parent_id, device_id)
                publish_parent_event(parent_id, "device_status_changed", _device_payload(row))
            except Exception:
                pass
        response_payload = {
            "status": "no_hop",
            "reason": "empty_chunk",
            "chunk_samples": 0,
            "consecutive_empty_chunks": empty_count,
        }
        _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, response_payload)
        return response_payload

    try:
        arr, sr, decode_method = _decode_audio_chunk(raw)
    except Exception as exc:
        logger.warning(
            "DETECT_CHUNK_DECODE_FAILED | filename=%s | content_type=%s | bytes=%s | reason=%s",
            getattr(upload, "filename", None),
            getattr(upload, "content_type", None),
            len(raw),
            str(exc),
        )
        raise HTTPException(status_code=400, detail="invalid_audio_chunk") from exc

    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        logger.warning(
            "DETECT_CHUNK_EMPTY_DECODED | filename=%s | content_type=%s | bytes=%s | decode_method=%s",
            getattr(upload, "filename", None),
            getattr(upload, "content_type", None),
            len(raw),
            decode_method,
        )
        response_payload = {"status": "no_hop", "reason": "empty_chunk", "chunk_samples": 0}
        _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, response_payload)
        return response_payload

    if sr != 16000:
        try:
            if librosa is None:
                raise ImportError("librosa is not available")
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000).astype(np.float32)
            sr = 16000
        except Exception as exc:
            logger.warning(
                "DETECT_CHUNK_RESAMPLE_FAILED | filename=%s | content_type=%s | orig_sr=%s | bytes=%s | reason=%s",
                getattr(upload, "filename", None),
                getattr(upload, "content_type", None),
                sr,
                len(raw),
                str(exc),
            )
            raise HTTPException(status_code=400, detail="audio_chunk_must_be_16khz") from exc

    # Loudness is normalized once per 1.5 s analysis window (evaluate_window), not per
    # chunk: per-chunk gain made the level jump inside a window.

    # Mark device as active (for telemetry: is_online, last_seen_at)
    device = svc.update_device_activity(db, parent_id, device_id, device=device)

    reported_battery = battery_percent if battery_percent is not None else battery
    if reported_battery is not None:
        device = svc.update_device_battery(db, parent_id, device_id, reported_battery, device=device)
    else:
        logger.warning(
            "DEVICE_BATTERY_MISSING | endpoint=/detect/chunk | parent_id=%s | device_id=%s",
            parent_id,
            device_id,
        )

    if _should_publish_device_status(parent_id, device_id):
        publish_parent_event(parent_id, "device_status_changed", _device_payload(device))

    logger.info(
        "DETECT_CHUNK_RECEIVED | parent_id=%s | device_id=%s | audio_len=%s | sr=%s | decode_method=%s",
        parent_id,
        device_id,
        len(arr),
        sr,
        decode_method,
    )

    stream_status = _apply_stream_order(
        parent_id, device_id, seq, t_start_ms, sent_at_ms, len(arr), sr, stream_id, silent_skipped,
    )
    if stream_status is not None:
        return stream_status

    svc_result = DetectionService.process_audio_chunk(
        db=db,
        parent_id=parent_id,
        device_id=device_id,
        arr=arr,
        sr=sr,
        latitude=latitude,
        longitude=longitude,
    )

    if svc_result.get("status") != "ok":
        svc_result.pop("session", None)
        _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, svc_result)
        return svc_result

    session = svc_result["session"]
    stage = svc_result["stage"]
    current_stranger_centroid = svc_result.get("current_stranger_centroid")
    score = svc_result.get("score", 0.0)
    closest_speaker_id = svc_result.get("closest_speaker_id")
    # Change 1: extract role for child-self fork in familiar branch below
    closest_speaker_role = str(stage.get("tier3", {}).get("closest_speaker_role") or "familiar_adult")
    # Change 7: extract distress sub-dict from evaluate_window result
    distress_info = stage.get("distress", {"distress": False, "distress_score": 0.0, "distress_class": None})
    # Change 5: use per-household T_HIGH / T_LOW from tier3 result (already resolved by evaluate_window)
    _tier3 = stage.get("tier3", {})
    T_HIGH_eff = float(_tier3.get("t_high", T_HIGH))
    T_LOW_eff = float(_tier3.get("t_low", T_LOW))
    now = now_ms()

    def _capture(decision_value: str, fired: bool) -> None:
        if not eval_capture.is_enabled():
            return
        buf = svc_result.get("inference_buffer")
        n = int(svc_result.get("window_samples") or 0)
        eval_capture.capture_window(
            parent_id=parent_id,
            device_id=device_id,
            now_ms=now,
            stage=stage,
            decision=decision_value,
            alert_fired=fired,
            stranger_streak=session.stranger_streak,
            centroid=current_stranger_centroid,
            window=buf[-n:] if buf is not None and n > 0 else None,
            sr=session.sr,
        )

    # Change 12: Fetch/cache household config for context gates
    if session.last_config_fetch_ms == 0 or (now - session.last_config_fetch_ms) > 300000:
        session.household_config = svc.get_household_config_dict(db, parent_id)
        session.last_config_fetch_ms = now

    decision = "hold"
    alert_fired = False
    alert_id = None
    alert_block_reason = None

    # Log stage results
    tier1_pass = stage.get("tier1_vad", {}).get("passed", False)
    tier2_pass = stage.get("tier2", {}).get("passed", False)
    tier3_pass = stage.get("tier3", {}).get("passed", False)
    tier2_category = str(stage.get("tier2", {}).get("category", "uncertain"))
    tier2_confidence = float(stage.get("tier2", {}).get("confidence", 0.0))
    perf_latency_ms = float(stage.get("perf", {}).get("processing_ms", 0.0))
    logger.info(f"STAGE_RESULT | score={score:.4f} | tier1_vad={tier1_pass} | tier2={tier2_pass} | tier3={tier3_pass} | stage_decision={stage.get('decision', 'unknown')}")
    _log_pipeline_decision(
        stage="vad",
        device_id=device_id,
        parent_id=parent_id,
        result="passed" if tier1_pass else "failed",
        score=None,
        streak=session.stranger_streak,
        latency_ms=perf_latency_ms,
    )
    _log_pipeline_decision(
        stage="tier2",
        device_id=device_id,
        parent_id=parent_id,
        result="passed" if tier2_pass else "failed",
        score=None,
        streak=session.stranger_streak,
        latency_ms=perf_latency_ms,
    )
    _log_pipeline_decision(
        stage="tier3",
        device_id=device_id,
        parent_id=parent_id,
        result="passed" if tier3_pass else "failed",
        score=score,
        streak=session.stranger_streak,
        latency_ms=perf_latency_ms,
    )

    # Phase 5: Device obstruction detection
    if tier1_pass and not tier2_pass:
        session.obstruction_counter += 1
        # 4800 chunks at 4 chunks/sec = 20 minutes of continuous muffled audio
        if session.obstruction_counter >= 4800:
            if (now - session.device_obstructed_alert_sent_at) > 3600000:  # throttle 1 hour
                logger.warning(f"DEVICE_OBSTRUCTION_DETECTED | parent_id={parent_id} | device_id={device_id}")
                create_and_escalate_alert(db, parent_id, device_id, session, score=0.0, alert_type="device_obstructed")
                session.device_obstructed_alert_sent_at = now
    elif tier2_pass:
        session.obstruction_counter = 0

    if not tier3_pass:
        if stage.get("decision") == "tier3_unavailable":
            # The speaker model failed: tell the parent (throttled) instead of
            # silently holding every window.
            logger.error("DETECT_CHUNK_TIER3_UNAVAILABLE | parent_id=%s | device_id=%s", parent_id, device_id)
            threading.Thread(target=_report_scoring_unavailable, args=(parent_id,), daemon=True).start()
        # Rejected windows (silence, music, too little speech) report the stage's
        # reason; "hold_tier3_unavailable" is reserved for a speaker-model failure.
        not_scored_decision = (
            "hold_tier3_unavailable" if stage.get("decision") == "tier3_unavailable"
            else str(stage.get("decision") or "rejected")
        )
        response_payload = {
            "status": "ok",
            "decision": not_scored_decision,
            "stranger_streak": session.stranger_streak,
            "score": score,
            "voiced_ms": (stage.get("tier1_vad") or {}).get("voiced_ms"),
            "thresholds": {"t_high": T_HIGH, "t_low": T_LOW},
            "stage": stage,
            "alert_fired": False,
            "alert_id": None,
        }
        _capture(not_scored_decision, False)
        _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, response_payload)
        return response_payload

    params = dataclasses.replace(DECISION_PARAMS, t_high=T_HIGH_eff, t_low=T_LOW_eff)
    result = decide(
        session,
        score=score,
        closest_speaker_id=closest_speaker_id,
        closest_speaker_role=closest_speaker_role,
        tier2_category=tier2_category,
        tier2_confidence=tier2_confidence,
        current_stranger_centroid=current_stranger_centroid,
        now_ms=now,
        params=params,
    )
    decision = result.decision
    alert_block_reason = result.block_reason
    logger.info(
        "SCORE_DECISION | score=%.4f | t_high=%.4f | t_low=%.4f | decision=%s | streak=%s | weight=%.2f "
        "| tier2_category=%s | tier2_confidence=%.4f | speaker_id=%s | role=%s",
        score, T_HIGH_eff, T_LOW_eff, decision, session.stranger_streak, session.stranger_weight_sum,
        tier2_category, tier2_confidence, closest_speaker_id, closest_speaker_role,
    )

    # Child voice drift tracking, flushed to the DB in batches rather than per window.
    if result.child_self_match and closest_speaker_id:
        session.child_drift_scores.append(score)
        if len(session.child_drift_scores) >= CHILD_DRIFT_FLUSH_EVERY:
            drift_warning = svc.flush_child_drift_scores(db, parent_id, closest_speaker_id, session)
            if drift_warning:
                logger.warning("DRIFT_WARNING | parent_id=%s | child_id=%s | drift threshold breached", parent_id, closest_speaker_id)

    # Capture the full stranger episode instead of only the last rolling window.
    if session.stranger_streak > 0:
        if session.stranger_segment is None:
            pre_roll_samples = int(STRANGER_PREROLL_SEC * session.sr)
            pre_roll_waveform = None
            if session.clip_buffer is not None and session.clip_buffer.shape[0] > len(arr):
                pre_roll_source = session.clip_buffer[:-len(arr)]
                if pre_roll_source.shape[0] > 0:
                    pre_roll_waveform = pre_roll_source[-pre_roll_samples:]
            svc.start_stranger_segment(session, arr, now, pre_roll_waveform=pre_roll_waveform)
            logger.info(
                "STRANGER_SEGMENT_START | parent_id=%s | device_id=%s | started_ms=%s | first_chunk_samples=%s | pre_roll_samples=%s",
                parent_id,
                device_id,
                now,
                int(len(arr)),
                int(pre_roll_waveform.shape[0]) if pre_roll_waveform is not None else 0,
            )
        else:
            svc.append_stranger_segment(session, arr)
    else:
        svc.clear_stranger_segment(session)

    # Identity (unenrolled contact) alert.
    if result.should_alert:
        if result.debounce_reset_similarity is not None:
            logger.info(
                "DEBOUNCE_RESET_SPEAKER_CHANGE | parent_id=%s | device_id=%s | similarity=%.4f | threshold=%.2f",
                parent_id, device_id, result.debounce_reset_similarity, params.speaker_change_threshold,
            )
        session.stranger_segment_ended_ms = now
        aid = create_and_escalate_alert(
            db, parent_id, device_id, session, score,
            current_stranger_centroid=current_stranger_centroid,
        )
        if aid:
            alert_fired = True
            alert_id = aid
            logger.info("IDENTITY_ALERT_FIRED | parent_id=%s | device_id=%s | score=%.4f | alert_id=%s", parent_id, device_id, score, aid)
        else:
            alert_block_reason = "alert_creation_failed"
    elif alert_block_reason:
        logger.info(
            "ALERT_BLOCKED | reason=%s | streak=%s/%s | weight=%.2f | last_alert_ms=%s",
            alert_block_reason, session.stranger_streak, params.alert_trigger_streak,
            session.stranger_weight_sum, session.last_alert_ms,
        )

    # Distress alert path (opt-in, independent of the identity streak and debounce).
    distress_score = float(distress_info.get("distress_score", 0.0))
    distress_class = distress_info.get("distress_class")
    if ENABLE_DISTRESS_ALERTS:
        if distress_score >= DISTRESS_ALERT_THRESHOLD:
            session.distress_streak += 1
        else:
            session.distress_streak = 0
        distress_debounce_elapsed = (now - session.last_distress_alert_ms) >= DISTRESS_DEBOUNCE_SEC * 1000
        if session.distress_streak >= DISTRESS_CONFIRM_WINDOWS and not alert_fired and distress_debounce_elapsed:
            distress_alert_id = create_and_escalate_alert(
                db, parent_id, device_id, session, distress_score,
                alert_type="distress",
            )
            if distress_alert_id:
                alert_fired = True
                alert_id = distress_alert_id
                decision = "distress_alert"
                session.distress_streak = 0
                logger.info(
                    "DISTRESS_ALERT_FIRED | parent_id=%s | device_id=%s | score=%.4f | class=%s | alert_id=%s",
                    parent_id, device_id, distress_score, distress_class, distress_alert_id,
                )

    response_payload = {
        "status": "ok",
        "decision": decision,
        "stranger_streak": session.stranger_streak,
        "score": score,
        # Speech found in the window (debug view on the watch; no speaker identity).
        "voiced_ms": (stage.get("tier1_vad") or {}).get("voiced_ms"),
        "thresholds": {"t_high": T_HIGH_eff, "t_low": T_LOW_eff},
        "distress": {
            "distress": distress_info.get("distress", False),
            "distress_score": distress_score,
            "distress_class": distress_class,
            "distress_streak": getattr(session, "distress_streak", 0),
        },
        "trigger": {
            "required_streak": ALERT_TRIGGER_STREAK,
            "force_alert_score": FORCE_ALERT_SCORE,
            "force_alert_min_streak": FORCE_ALERT_MIN_STREAK,
            "familiar_grace_sec": FAMILIAR_GRACE_SEC,
            "familiar_hold_floor": FAMILIAR_HOLD_FLOOR,
            "familiar_grace_margin": FAMILIAR_GRACE_MARGIN,
            "familiar_grace_min_conf": FAMILIAR_GRACE_MIN_CONF,
            "familiar_reset_min_streak": FAMILIAR_RESET_MIN_STREAK,
            "soft_stranger_margin": SOFT_STRANGER_MARGIN,
            "soft_stranger_min_conf": SOFT_STRANGER_MIN_CONF,
            "stranger_min_tier2_conf": STRANGER_MIN_TIER2_CONF,
            "stranger_uncertain_conf_floor": STRANGER_UNCERTAIN_CONF_FLOOR,
            "debounce_sec": DEBOUNCE_SEC,
            "speaker_change_threshold": SPEAKER_CHANGE_THRESHOLD,
            "block_reason": alert_block_reason,
        },
        "stage": stage,
        "alert_fired": alert_fired,
        "alert_id": alert_id,
    }
    _capture(decision, alert_fired)
    svc.persist_session_state(session)
    if current["kind"] == "device":
        # The monitored device never needs per-speaker scores or thresholds.
        response_payload = {k: v for k, v in response_payload.items() if k not in {"stage", "trigger", "distress"}}
    _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, response_payload)
    return response_payload


@app.delete("/detect/session")
def stop_detect_session(device_id: str, current=Depends(get_principal), db: Session = Depends(get_db)):
    _require_device_access(current, device_id)
    svc.stop_session(current["parent_id"], device_id)
    return {"status": "stopped", "device_id": device_id}


@app.get("/alerts")
def get_alert_history(
    limit: int = 50,
    offset: int = 0,
    since_ms: Optional[int] = None,
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    rows = svc.list_alerts(db, current["parent_id"], limit=max(1, min(limit, 200)), offset=max(0, offset), since_ms=since_ms)
    return {
        "items": [
            _alert_response_payload(row)
            for row in rows
        ]
    }


@app.delete("/alerts/{alert_id}")
def delete_alert(alert_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    svc.delete_alert(db, current["parent_id"], alert_id)
    return {"status": "deleted", "alert_id": alert_id}


@app.delete("/alerts")
def delete_all_alerts(current=Depends(get_current_parent), db: Session = Depends(get_db)):
    deleted_count = svc.delete_all_alerts(db, current["parent_id"])
    return {"status": "cleared", "deleted_count": deleted_count}


@app.post("/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    row = svc.ack_alert(db, current["parent_id"], alert_id)
    alert_payload = _alert_response_payload(row)
    return {
        "status": "acknowledged",
        # Flat fields for clients that deserialize AckAlertResponse at root.
        **alert_payload,
        # Backward-compatible nested payload for older clients.
        "alert": alert_payload,
    }


@app.post("/alerts/test-fire")
def fire_test_alert(body: TestFireAlertRequest, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    if not security.test_endpoints_allowed():
        raise HTTPException(status_code=404, detail="endpoint_not_found")

    parent_id = current["parent_id"]
    # Ensure the device belongs to the current parent.
    svc.get_device(db, parent_id, body.device_id)

    session = svc.get_or_create_session(parent_id, body.device_id)
    if body.latitude is not None:
        session.lat = body.latitude
    if body.longitude is not None:
        session.lon = body.longitude

    # Generate a short synthetic clip so the alert has playable audio.
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(session.sr * duration_sec), endpoint=False, dtype=np.float32)
    waveform = (0.02 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    clip_path = svc.save_alert_clip(parent_id, session, waveform)
    row = svc.create_alert(
        db=db,
        parent_id=parent_id,
        device_id=body.device_id,
        audio_clip_path=clip_path,
        confidence_score=float(body.confidence_score),
        latitude=session.lat,
        longitude=session.lon,
    )
    publish_parent_event(parent_id, "alert_created", _alert_response_payload(row))

    parent = svc.get_parent(db, parent_id)
    parent_device = (
        db.query(Device)
        .filter(
            Device.parent_id == parent.id,
            Device.role == DeviceRole.parent_device,
            Device.device_token.isnot(None),
        )
        .order_by(Device.updated_at.desc())
        .first()
    )
    parent_fcm_token = parent_device.device_token if parent_device else None
    if parent_fcm_token:
        send_fcm_push(
            parent_fcm_token,  # type: ignore
            str(row.id),
            str(row.device_id),
            _dt_to_epoch_ms(row.timestamp) or now_ms(),  # type: ignore
            row.latitude,  # type: ignore
            row.longitude,  # type: ignore
            float(row.confidence_score or 0.0),  # type: ignore
            recipient_role="parent_device",
        )

    return {
        "status": "test_alert_fired",
        "alert_id": str(row.id),
        "device_id": str(row.device_id),
        "confidence_score": row.confidence_score,
        "timestamp_ms": _dt_to_epoch_ms(row.timestamp),  # type: ignore
        "clip_url": f"/alerts/{row.id}/clip",
    }


@app.post("/alerts/{alert_id}/flag-familiar")
def flag_alert_as_familiar(
    alert_id: str,
    request: Request,
    body: FlagFamiliarRequest,
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    parent_id = current["parent_id"]
    alert = svc.get_alert(db, parent_id, alert_id)
    clip_path = alert.audio_clip_path
    if not clip_path or not os.path.exists(clip_path):  # type: ignore
        raise HTTPException(status_code=404, detail="audio_clip_not_found")

    speaker = (
        svc.create_speaker(db, parent_id, body.display_name)
        if not body.speaker_id
        else svc.get_speaker(db, parent_id, body.speaker_id)
    )

    # Change 3: stricter quality gate for alert-clip enrollment (2 s voiced floor is enforced
    # by MIN_VOICED_MS_ENROLL in verification_pipeline; SNR and overlap checks also apply).
    # run_enroll_embedding already applies all gates — no extra call needed here.
    try:
        embs, stage_info = run_enroll_embedding(clip_path)  # type: ignore
    except ValueError as exc:
        # Delete the speaker row if we just created it and enrollment failed.
        if not body.speaker_id:
            try:
                svc.delete_speaker(db, parent_id, str(speaker.id))
            except Exception:
                pass
        logger.warning(
            "ALERT_CLIP_ENROLL_REJECTED | parent_id=%s | alert_id=%s | reason=%s",
            parent_id, alert_id, str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail={"code": str(exc), "message": enrollment_rejection_message(str(exc))},
        ) from exc

    for emb in embs:
        svc.save_speaker_embedding(db, parent_id, str(speaker.id), emb)

    _, quality_label = svc.compute_and_store_enrollment_quality(db, parent_id, str(speaker.id))
    if quality_label == "poor":
        logger.warning(
            "ALERT_CLIP_ENROLL_WARNING | parent_id=%s | alert_id=%s | quality_label=poor "
            "| gallery may be noisy; encourage re-enrollment from a clean recording.",
            parent_id, alert_id,
        )

    # Confusability check (Change 4)
    confusability_warnings = []
    try:
        conflicts = svc.check_enrollment_confusability(db, parent_id, str(speaker.id))
        for conflicting_speaker_id, sim in conflicts:
            confusability_warnings.append({
                "conflicting_speaker_id": conflicting_speaker_id,
                "similarity": round(float(sim), 4),
            })
    except Exception as _conf_exc:
        logger.warning("CONFUSABILITY_CHECK_ERROR | parent_id=%s | reason=%s", parent_id, _conf_exc)

    refreshed_rows = svc.list_speakers(db, parent_id)
    refreshed_items = [_speaker_payload(row, request, parent_id) for row in refreshed_rows]

    resp = {
        "status": "flagged_familiar",
        "source_alert_id": alert_id,
        "speaker_id": str(speaker.id),
        "display_name": speaker.display_name,
        "samples_saved": len(embs),
        "items": refreshed_items,
        "stages": stage_info,
    }
    if quality_label == "poor":
        resp["warning"] = "Alert clip quality is low. Encourage the parent to record a clean enrollment."
    if confusability_warnings:
        resp["confusability_warnings"] = confusability_warnings
    return resp


@app.get("/alerts/{alert_id}/clip")
def get_alert_clip(alert_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    row = svc.get_alert(db, current["parent_id"], alert_id)
    path = row.audio_clip_path
    # Older alerts may exist without persisted media. Return an empty response cleanly.
    if not path:
        return Response(status_code=204)
    if not os.path.exists(path):  # type: ignore
        raise HTTPException(status_code=404, detail="audio_clip_not_found")
    return FileResponse(path, media_type="audio/wav", filename=os.path.basename(path))  # type: ignore


@app.post("/devices")
def create_device(
    device_name: Optional[str] = Form(None),
    installation_id: Optional[str] = Form(None),
    role: str = Form(...),
    device_token: Optional[str] = Form(None),
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    parent_id = current["parent_id"]
    try:
        device_role = DeviceRole(role)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid_device_role")
    
    logger.info(
        "DEVICE_REGISTRATION_START | parent_id=%s | role=%s | device_name=%s | has_fcm_token=%s",
        parent_id,
        role,
        device_name,
        bool(device_token),
    )
    
    device = svc.create_device(db, parent_id, device_name, device_role, installation_id=installation_id, device_token=device_token)
    
    logger.info(
        "DEVICE_REGISTRATION_SUCCESS | device_id=%s | parent_id=%s | role=%s | device_name=%s",
        str(device.id),
        str(device.parent_id),
        device.role.value,
        device.device_name,
    )

    payload: Dict[str, Any] = {
        "status": "created",
        "device": {
            "id": str(device.id),
            "parent_id": str(device.parent_id),
            "installation_id": device.installation_id,
            "device_name": device.device_name,
            "role": device.role.value,
            "device_token": device.device_token
        }
    }
    if device.role == DeviceRole.child_device:
        # The monitored device should keep only this restricted token, not the
        # parent's session: it can stream audio but cannot read alerts, change
        # enrolled voices or switch monitoring off.
        payload["device"]["device_access_token"] = security.make_device_token(
            str(device.parent_id), str(device.id), int(device.access_token_version or 0)
        )
    return payload


@app.post("/devices/upsert")
def upsert_device(
    device_name: Optional[str] = Form(None),
    installation_id: Optional[str] = Form(None),
    role: str = Form(...),
    device_token: Optional[str] = Form(None),
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    return create_device(
        device_name=device_name,
        installation_id=installation_id,
        role=role,
        device_token=device_token,
        current=current,
        db=db,
    )

@app.get("/devices")
def list_devices(
    since_ms: Optional[int] = None,
    current=Depends(get_principal),
    db: Session = Depends(get_db),
):
    parent_id = current["parent_id"]
    if current["kind"] == "device":
        # A device only sees itself (it polls its own monitoring state).
        item = _device_payload(svc.get_device(db, parent_id, current["device_id"]))
        # The child device switches to precise GPS only while the parent-nearby rule is on.
        item["precise_location"] = child_safety.read_config(db, parent_id)["parent_proximity_enabled"]
        return {"items": [item]}
    logger.info("DEVICE_LIST_REQUEST | parent_id=%s", parent_id)
    
    rows = svc.list_devices(db, parent_id, since_ms=since_ms)
    
    logger.info(
        "DEVICE_LIST_RESPONSE | parent_id=%s | count=%d | roles=%s",
        parent_id,
        len(rows),
        ','.join([r.role.value for r in rows]) if rows else "none",
    )
    
    for idx, row in enumerate(rows):
        logger.debug(
            "DEVICE_RESULT[%d] | id=%s | parent_id=%s | role=%s | name=%s",
            idx,
            str(row.id),
            str(row.parent_id),
            row.role.value,
            row.device_name,
        )
    
    return {"items": [_device_payload(row) for row in rows]}

@app.patch("/devices/{device_id}/monitoring")
def patch_device_monitoring(
    device_id: str,
    body: DeviceMonitoringPatchRequest,
    current=Depends(get_principal),
    db: Session = Depends(get_db),
):
    started_at = time.perf_counter()
    if current["kind"] == "device":
        # A child device may switch its own monitoring on, never off.
        _require_device_access(current, device_id)
        if not body.monitoring_enabled:
            raise HTTPException(status_code=403, detail="only_parent_can_stop_monitoring")
    if body.monitoring_enabled and not svc.list_speakers(db, current["parent_id"]):
        # Without an enrolled voice every speaker would be reported as a stranger.
        raise HTTPException(
            status_code=409,
            detail="Enroll at least one family voice before turning monitoring on.",
        )
    row, command_sent = svc.set_device_monitoring(
        db,
        current["parent_id"],
        device_id,
        body.monitoring_enabled,
    )
    publish_parent_event(current["parent_id"], "monitoring_changed", _device_payload(row))
    logger.info(
        "MONITORING_TOGGLE_LATENCY | parent_id=%s | device_id=%s | latency_ms=%.2f",
        current["parent_id"],
        device_id,
        (time.perf_counter() - started_at) * 1000.0,
    )
    response = _device_payload(row)
    response["command_sent"] = command_sent
    return response


class DeviceTokenPatchRequest(BaseModel):
    device_token: Optional[str] = None


@app.patch("/devices/{device_id}/token")
def patch_device_token(
    device_id: str,
    body: DeviceTokenPatchRequest,
    current=Depends(get_principal),
    db: Session = Depends(get_db),
):
    _require_device_access(current, device_id)
    row = svc.update_device_token(db, current["parent_id"], device_id, body.device_token)
    return {"status": "ok", "device": _device_payload(row)}

@app.post("/devices/{device_id}/heartbeat")
def post_device_heartbeat(
    device_id: str,
    body: DeviceHeartbeatRequest,
    current=Depends(get_principal),
    db: Session = Depends(get_db),
):
    _require_device_access(current, device_id)
    if body.battery_percent is None:
        logger.warning(
            "DEVICE_BATTERY_MISSING | endpoint=/devices/{id}/heartbeat | parent_id=%s | device_id=%s",
            current["parent_id"],
            device_id,
        )
    row = svc.update_device_heartbeat(
        db,
        current["parent_id"],
        device_id,
        body.battery_percent,
        body.is_online,
        # Only the parent decides whether monitoring is on.
        body.monitoring_enabled if current["kind"] == "parent" else None,
    )
    if _should_publish_device_status(current["parent_id"], device_id):
        publish_parent_event(current["parent_id"], "device_status_changed", _device_payload(row))
    return {"status": "ok", "device": _device_payload(row)}


@app.post("/devices/{device_id}/monitoring/ack")
def post_monitoring_ack(
    device_id: str,
    body: DeviceMonitoringAckRequest,
    current=Depends(get_principal),
    db: Session = Depends(get_db),
):
    _require_device_access(current, device_id)
    if body.battery_percent is None:
        logger.warning(
            "DEVICE_BATTERY_MISSING | endpoint=/devices/{id}/monitoring/ack | parent_id=%s | device_id=%s",
            current["parent_id"],
            device_id,
        )
    row = svc.update_device_heartbeat(
        db,
        current["parent_id"],
        device_id,
        battery_percent=body.battery_percent,
        is_online=body.is_online,
        monitoring_enabled=body.monitoring_enabled if current["kind"] == "parent" else None,
    )
    if _should_publish_device_status(current["parent_id"], device_id):
        publish_parent_event(current["parent_id"], "device_status_changed", _device_payload(row))
    return {"status": "acknowledged", "device": _device_payload(row)}


# --------------------------------------------------------------------------- #
# Child-device pairing (watch shows a code, the parent enters it on the phone)
# --------------------------------------------------------------------------- #

class PairStartRequest(BaseModel):
    installation_id: str
    device_name: Optional[str] = None


class PairClaimRequest(BaseModel):
    code: str


class PairStatusRequest(BaseModel):
    pair_id: str
    pair_secret: str


@app.post("/pair/start", dependencies=[Depends(auth_rate_limit)])
def pair_start(body: PairStartRequest, db: Session = Depends(get_db)):
    return child_safety.start_pairing(db, body.installation_id, body.device_name)


@app.post("/pair/claim")
def pair_claim(body: PairClaimRequest, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    # Codes are 6 digits: limit guesses per parent account.
    enforce_rate_limit(f"pairclaim:{current['parent_id']}", 10, 60.0)
    device = child_safety.claim_pairing(db, current["parent_id"], body.code)
    payload = _device_payload(device)
    publish_parent_event(current["parent_id"], "device_status_changed", payload)
    return {"status": "paired", "device": payload}


@app.post("/pair/status")
def pair_status(body: PairStatusRequest, request: Request, db: Session = Depends(get_db)):
    enforce_rate_limit(f"pairstatus:{security.client_ip(request)}", 120, 60.0)
    status, device = child_safety.pairing_status(db, body.pair_id, body.pair_secret)
    if status == "pending":
        return {"status": "pending"}
    return {
        "status": "paired",
        "device_id": str(device.id),
        "parent_id": str(device.parent_id),
        "device_name": device.device_name,
        "device_access_token": security.make_device_token(
            str(device.parent_id), str(device.id), int(device.access_token_version or 0)
        ),
    }


# --------------------------------------------------------------------------- #
# Household settings (parent-nearby rule) and parent location
# --------------------------------------------------------------------------- #

class HouseholdConfigPatchRequest(BaseModel):
    parent_proximity_enabled: Optional[bool] = None
    parent_proximity_radius_m: Optional[float] = None


class ParentLocationRequest(BaseModel):
    device_id: str
    latitude: float
    longitude: float
    accuracy_m: Optional[float] = None
    location_ts_ms: Optional[int] = None


@app.get("/household/config")
def get_household_config(current=Depends(get_current_parent), db: Session = Depends(get_db)):
    return child_safety.read_config(db, current["parent_id"])


@app.patch("/household/config")
def patch_household_config(
    body: HouseholdConfigPatchRequest, current=Depends(get_current_parent), db: Session = Depends(get_db)
):
    return child_safety.update_config(
        db, current["parent_id"], body.parent_proximity_enabled, body.parent_proximity_radius_m
    )


@app.post("/parent/location")
def post_parent_location(body: ParentLocationRequest, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    device = svc.get_device(db, current["parent_id"], body.device_id)
    if device.role != DeviceRole.parent_device:
        raise HTTPException(status_code=403, detail="parent_device_required")
    child_safety.store_fix(db, device, body.latitude, body.longitude, body.accuracy_m, body.location_ts_ms)
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# SOS from the child device
# --------------------------------------------------------------------------- #

class SosRequest(BaseModel):
    device_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy_m: Optional[float] = None
    battery_percent: Optional[int] = None


MAX_SOS_CLIP_BYTES = int(os.environ.get("SAFEEAR_MAX_SOS_CLIP_BYTES", str(2 * 1024 * 1024)))


def _push_sos(parent_id: str, alert_id: str, device_id: str, device_name: Optional[str],
              ts_ms: int, lat: Optional[float], lon: Optional[float]) -> None:
    """FCM -> SMS -> email chain for an SOS; runs on a background thread."""
    from app.utils.notification_worker import escalate_alert

    thread_db = SessionLocal()
    try:
        parent = svc.get_parent(thread_db, parent_id)
        token = child_safety.parent_push_token(thread_db, parent.id)
        escalate_alert(
            str(parent.email) if parent.email else None,
            str(parent.phone_number) if parent.phone_number else None,
            token,
            alert_id,
            device_id,
            ts_ms,
            lat,
            lon,
            f"/alerts/{alert_id}/clip",
            1.0,
            recipient_role="parent_device",
            alert_type="sos",
            device_name=device_name,
        )
    except Exception:
        logger.exception("SOS_ESCALATION_FAILED | alert_id=%s", alert_id)
    finally:
        thread_db.close()


@app.post("/sos")
def post_sos(body: SosRequest, current=Depends(get_principal), db: Session = Depends(get_db)):
    """Manual SOS: always delivered, whatever the monitoring state, context gates or
    parent-nearby rule.  A repeat within SOS_DEDUPE_SEC returns the same alert."""
    _require_device_access(current, body.device_id)
    parent_id = current["parent_id"]
    device = svc.get_device(db, parent_id, body.device_id)
    if device.role != DeviceRole.child_device:
        raise HTTPException(status_code=403, detail="sos_requires_child_device")
    if body.latitude is not None and body.longitude is not None:
        device = child_safety.store_fix(db, device, body.latitude, body.longitude, body.accuracy_m)
        svc.update_session_location(parent_id, body.device_id, body.latitude, body.longitude)
    if body.battery_percent is not None:
        device = svc.update_device_battery(db, parent_id, body.device_id, body.battery_percent, device=device)
    existing = child_safety.recent_sos(db, device)
    if existing is not None:
        return {"status": "sent", "deduplicated": True, "alert": _alert_response_payload(existing)}
    lat = body.latitude if body.latitude is not None else device.last_location_lat
    lon = body.longitude if body.longitude is not None else device.last_location_lon
    row = svc.create_alert(db, parent_id, body.device_id, None, 1.0, lat, lon, alert_type="sos")
    payload = _alert_response_payload(row)
    payload["device_name"] = device.device_name
    publish_parent_event(parent_id, "alert_created", payload)
    child_safety.run_in_background(
        _push_sos, parent_id, str(row.id), body.device_id, device.device_name,
        payload["timestamp_ms"] or now_ms(), lat, lon,
    )
    logger.info("SOS_RAISED | parent_id=%s | device_id=%s | alert_id=%s", parent_id, body.device_id, row.id)
    return {"status": "sent", "deduplicated": False, "alert": payload}


@app.post("/sos/{alert_id}/clip")
async def post_sos_clip(
    alert_id: str,
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    current=Depends(get_principal),
    db: Session = Depends(get_db),
):
    upload = file or audio
    if upload is None:
        raise HTTPException(status_code=422, detail="missing_audio")
    row = svc.get_alert(db, current["parent_id"], alert_id)
    if row.alert_type != "sos":
        raise HTTPException(status_code=409, detail="not_an_sos_alert")
    _require_device_access(current, str(row.device_id))
    raw = await _read_upload_limited(upload, MAX_SOS_CLIP_BYTES)
    try:
        arr, sr, _ = _decode_audio_chunk(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_audio") from exc
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if arr.size == 0:
        raise HTTPException(status_code=400, detail="empty_audio")
    row.audio_clip_path = child_safety.save_sos_clip(current["parent_id"], alert_id, arr.reshape(-1), int(sr))
    db.commit()
    db.refresh(row)
    payload = _alert_response_payload(row)
    publish_parent_event(current["parent_id"], "alert_updated", payload)

    def _notify_clip() -> None:
        from app.utils.notification_worker import send_status_push

        thread_db = SessionLocal()
        try:
            token = child_safety.parent_push_token(thread_db, row.parent_id)
            if token:
                send_status_push(
                    token, "SOS audio ready", "Tap to listen to what is happening around your child.",
                    {"type": "sos_audio", "alert_id": alert_id, "alert_type": "sos"},
                )
        finally:
            thread_db.close()

    child_safety.run_in_background(_notify_clip)
    return {"status": "ok", "alert": payload}


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    auth_header = websocket.headers.get("authorization")
    query_token = (websocket.query_params.get("access_token") or "").strip()
    token = ""
    if auth_header:
        if not auth_header.lower().startswith("bearer "):
            await websocket.close(code=1008, reason="invalid_authorization_header")
            return
        token = auth_header.split(" ", 1)[1].strip()
    elif query_token and not security.is_production():
        # Query-string tokens end up in proxy/access logs; development only.
        token = query_token
    else:
        await websocket.close(code=1008, reason="missing_authorization")
        return

    if not token:
        await websocket.close(code=1008, reason="empty_access_token")
        return
    try:
        parent_id = _parent_id_from_access_token(token)
    except Exception:
        await websocket.close(code=1008, reason="invalid_access_token")
        return

    db = SessionLocal()
    try:
        svc.get_parent(db, parent_id)
    except Exception:
        await websocket.close(code=1008, reason="parent_not_found")
        return
    finally:
        db.close()

    await websocket.accept()
    _ws_set_loop(asyncio.get_running_loop())

    client_id = f"{parent_id}:{int(time.time() * 1000)}:{id(websocket)}"
    queue: asyncio.Queue = asyncio.Queue(maxsize=WS_CLIENT_QUEUE_MAX)
    await _ws_register_client(parent_id, client_id, queue)

    receiver_task: Optional[asyncio.Task] = asyncio.create_task(websocket.receive_text())
    queue_task: Optional[asyncio.Task] = None
    try:
        while True:
            if queue_task is None or queue_task.done():
                queue_task = asyncio.create_task(queue.get())

            assert queue_task is not None
            assert receiver_task is not None
            done, _ = await asyncio.wait(
                [queue_task, receiver_task],
                timeout=WS_PING_INTERVAL_SEC,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if receiver_task in done:
                try:
                    await receiver_task
                except asyncio.CancelledError:
                    break
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
                receiver_task = asyncio.create_task(websocket.receive_text())

            if queue_task in done:
                try:
                    queued = queue_task.result()
                except Exception:
                    queued = None
                queue_task = None
                if queued is None:
                    continue

                queued_at_ms = int(queued.get("queued_at_ms", now_ms()))
                event = queued.get("event", {})
                lag_ms = max(0, now_ms() - queued_at_ms)
                _ws_manager_module._WS_LAST_QUEUE_LAG_MS = lag_ms
                try:
                    await websocket.send_json(event)
                except (RuntimeError, WebSocketDisconnect):
                    break
            elif websocket.client_state == WebSocketState.CONNECTED and websocket.application_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": "ping", "timestamp_ms": now_ms()})
                except (RuntimeError, WebSocketDisconnect):
                    break
    except WebSocketDisconnect:
        pass
    finally:
        cleanup: List[asyncio.Task] = []
        for task in (receiver_task, queue_task):
            if task is not None and not task.done():
                task.cancel()
                cleanup.append(task)
        if cleanup:
            await asyncio.gather(*cleanup, return_exceptions=True)
        await _ws_unregister_client(parent_id, client_id)


@app.get("/health")
def health():
    with _WS_STATE_LOCK:
        active_clients = sum(len(v) for v in _WS_CLIENTS_BY_PARENT.values())
        published_counts = dict(_WS_PUBLISHED_COUNTS)
        queue_lag_ms = int(_ws_manager_module._WS_LAST_QUEUE_LAG_MS)
    vad_status = vad_module.status()
    yamnet_status = yamnet_classifier_module.status()
    speaker_model_ok = feature_extractor_module._redimnet is not None
    degraded = vad_status["backend"] != "silero" or yamnet_status["backend"] != "yamnet" or not speaker_model_ok
    return {
        "status": "degraded" if degraded else "ok",
        "models": {
            "vad": vad_status["backend"],
            "audio_events": yamnet_status["backend"],
            "speaker_embedding": "redimnet" if speaker_model_ok else "unavailable",
        },
        "metrics": {
            "active_ws_clients": active_clients,
            "event_queue_lag_ms": queue_lag_ms,
            "events_published": published_counts,
        },
    }


@app.post("/admin/calibrate-thresholds")
def calibrate_thresholds(body: CalibrateThresholdsRequest, current=Depends(get_current_parent)):
    if not security.test_endpoints_allowed():
        raise HTTPException(status_code=404, detail="endpoint_not_found")
    if len(body.familiar_scores or []) + len(body.stranger_scores or []) > 20000:
        raise HTTPException(status_code=413, detail="too_many_scores")
    if not body.familiar_scores or not body.stranger_scores:
        raise HTTPException(status_code=422, detail="familiar_scores_and_stranger_scores_required")

    eer_threshold, eer_percent = _find_eer_threshold(body.familiar_scores, body.stranger_scores)
    far_at_current = sum(1 for s in body.stranger_scores if float(s) >= T_HIGH) / max(1, len(body.stranger_scores))
    frr_at_current = sum(1 for s in body.familiar_scores if float(s) < T_LOW) / max(1, len(body.familiar_scores))
    suggested_t_high = min(1.0, eer_threshold + 0.05)
    suggested_t_low = max(0.0, eer_threshold - 0.05)

    return {
        "current_t_high": float(T_HIGH),
        "current_t_low": float(T_LOW),
        "eer_threshold": float(eer_threshold),
        "eer_percent": float(eer_percent),
        "far_at_current": float(far_at_current),
        "frr_at_current": float(frr_at_current),
        "suggested_t_high": float(suggested_t_high),
        "suggested_t_low": float(suggested_t_low),
    }


@app.get("/health/cache")
def health_cache():
    client = _get_redis_client()
    redis_ping_ms = -1.0
    if client is not None:
        started = time.perf_counter()
        try:
            client.ping()
            redis_ping_ms = round((time.perf_counter() - started) * 1000.0, 2)
        except Exception:
            redis_ping_ms = -1.0

    degraded = bool(_CACHE_WARMUP_SUMMARY.get("parents_with_zero_embeddings")) or redis_ping_ms < 0.0
    return {
        "status": "degraded" if degraded else "ok",
        "cached_parents": int(_CACHE_WARMUP_SUMMARY.get("cached_parents", 0)),
        "cached_speakers": int(_CACHE_WARMUP_SUMMARY.get("speakers_loaded", 0)),
        "redis_ping_ms": redis_ping_ms,
    }


@app.get("/")
def home():
    return {
        "name": "SafeEar Backend",
        "version": "2.4.12",
        "message": "API is running",
    }
