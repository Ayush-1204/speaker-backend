from dotenv import load_dotenv
load_dotenv()

import asyncio
import contextlib
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
from app.utils.verification_pipeline import run_enroll_embedding
from app.utils import feature_extractor as feature_extractor_module
from app.utils import yamnet_classifier as yamnet_classifier_module
from app.utils import vad as vad_module
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
from app.services.decision_engine import _has_speaker_changed
from app.services.cache_service import (
    warmup_embeddings_to_redis as _warmup_embeddings_to_redis_fn,
    get_redis_client as _get_redis_client,
    _CACHE_WARMUP_SUMMARY,
)
from fastapi.middleware.cors import CORSMiddleware

# Module logger — must be defined before _lifespan which uses it.
logger = logging.getLogger(__name__)

# Application instance
@contextlib.asynccontextmanager
async def _lifespan(app_instance):
    """FastAPI lifespan: runs startup logic then yields control to the app."""
    init_db()
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
    yield  # App handles requests here


app = FastAPI(lifespan=_lifespan)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

now_ms = svc.now_ms

# Route-level threshold aliases kept for compatibility with existing tests and logs.
T_HIGH = svc.T_HIGH
T_LOW = svc.T_LOW
STRANGER_MIN_TIER2_CONF = float(os.environ.get("SAFEEAR_STRANGER_MIN_TIER2_CONF", "0.55"))
STRANGER_UNCERTAIN_CONF_FLOOR = float(os.environ.get("SAFEEAR_STRANGER_UNCERTAIN_CONF_FLOOR", "0.25"))
FAMILIAR_GRACE_SEC = int(os.environ.get("SAFEEAR_FAMILIAR_GRACE_SEC", "5"))
FAMILIAR_HOLD_FLOOR = float(os.environ.get("SAFEEAR_FAMILIAR_HOLD_FLOOR", str(T_LOW)))
FAMILIAR_GRACE_MARGIN = float(os.environ.get("SAFEEAR_FAMILIAR_GRACE_MARGIN", "0.05"))
FAMILIAR_GRACE_MIN_CONF = float(os.environ.get("SAFEEAR_FAMILIAR_GRACE_MIN_CONF", "0.75"))
FAMILIAR_RESET_MIN_STREAK = int(os.environ.get("SAFEEAR_FAMILIAR_RESET_MIN_STREAK", "2"))
SOFT_STRANGER_MARGIN = float(os.environ.get("SAFEEAR_SOFT_STRANGER_MARGIN", "0.04"))
SOFT_STRANGER_MIN_CONF = float(os.environ.get("SAFEEAR_SOFT_STRANGER_MIN_CONF", "0.55"))
FORCE_ALERT_SCORE = float(os.environ.get("SAFEEAR_FORCE_ALERT_SCORE", "0.15"))
FORCE_ALERT_MIN_STREAK = int(os.environ.get("SAFEEAR_FORCE_ALERT_MIN_STREAK", "2"))
ALERT_TRIGGER_STREAK = int(os.environ.get("SAFEEAR_ALERT_TRIGGER_STREAK", "3"))
DEBOUNCE_SEC = int(os.environ.get("SAFEEAR_DEBOUNCE_SEC", "60"))
SPEAKER_CHANGE_THRESHOLD = float(os.environ.get("SAFEEAR_SPEAKER_CHANGE_THRESHOLD", "0.85"))
STRANGER_PREROLL_SEC = float(os.environ.get("SAFEEAR_STRANGER_PREROLL_SEC", "5.0"))
# Idempotency cache for recently processed detect chunks (in-memory)
DETECT_CHUNK_IDEMPOTENCY_WINDOW_MS = int(os.environ.get("SAFEEAR_DETECT_CHUNK_IDEMPOTENCY_WINDOW_MS", "2000"))
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


class RenameSpeakerRequest(BaseModel):
    display_name: str


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


def _extract_bearer(auth_header: Optional[str]) -> str:
    if not auth_header:
        raise HTTPException(status_code=401, detail="missing_authorization_header")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="invalid_authorization_header")
    return auth_header.split(" ", 1)[1].strip()


def _decode_audio_chunk(raw: bytes) -> Tuple[np.ndarray, int, str]:
    return decode_audio_chunk(raw)


def get_current_parent(
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
) -> Dict[str, Any]:
    from app.utils.prd_services import parse_jwt
    token = _extract_bearer(authorization)
    payload = parse_jwt(token)
    parent_id = payload.get("sub")
    if not parent_id:
        raise HTTPException(status_code=401, detail="invalid_token_subject")
    try:
        svc.get_parent(db, parent_id)
    except HTTPException as exc:
        raise HTTPException(status_code=401, detail="parent_not_found") from exc
    return {"parent_id": parent_id}


def _parent_id_from_access_token(token: str) -> str:
    from app.utils.prd_services import parse_jwt

    payload = parse_jwt(token)
    parent_id = payload.get("sub")
    if not parent_id:
        raise HTTPException(status_code=401, detail="invalid_token_subject")
    return str(parent_id)


def _verify_google_like_token(id_token: str) -> Dict[str, Any]:
    allow_dev = os.environ.get("SAFEEAR_ALLOW_DEV_GOOGLE_TOKEN", "true").lower() == "true"

    # Dev token bypass (for testing/development).
    if id_token.startswith("dev:"):
        if not allow_dev:
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
                "GOOGLE_AUTH_AUDIENCE_MISMATCH | aud=%s | allowed=%s",
                aud,
                ",".join(allowed_client_ids),
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
        logger.info("PASSWORD_RESET_EMAIL_SKIPPED | email=%s | reason=sendgrid_not_configured", to_email)
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
        logger.warning("PASSWORD_RESET_EMAIL_FAILED | email=%s | reason=%s", to_email, str(exc))


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


@app.post("/auth/google")
def auth_google(body: GoogleAuthRequest, db: Session = Depends(get_db)):
    from app.utils.prd_services import make_jwt, save_refresh_token
    info = _verify_google_like_token(body.id_token)
    sub = info.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="google_sub_missing")

    parent = svc.upsert_parent(db, str(sub), info.get("email"), info.get("name"))
    access_token = make_jwt(str(parent.id))
    refresh_token = save_refresh_token(str(parent.id))
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 604800,
        "parent": {
            "id": str(parent.id),
            "email": parent.email,
            "display_name": parent.display_name
        },
    }


@app.post("/auth/register-email")
def auth_register_email(body: EmailRegisterRequest, db: Session = Depends(get_db)):
    from app.utils.prd_services import make_jwt, save_refresh_token

    parent = svc.register_parent_with_email(db, body.email, body.password, body.display_name)
    access_token = make_jwt(str(parent.id))
    refresh_token = save_refresh_token(str(parent.id))
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 604800,
        "parent": {
            "id": str(parent.id),
            "email": parent.email,
            "display_name": parent.display_name,
        },
    }


@app.post("/auth/login-email")
def auth_login_email(body: EmailLoginRequest, db: Session = Depends(get_db)):
    from app.utils.prd_services import make_jwt, save_refresh_token

    parent = svc.login_parent_with_email(db, body.email, body.password)
    access_token = make_jwt(str(parent.id))
    refresh_token = save_refresh_token(str(parent.id))
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 604800,
        "parent": {
            "id": str(parent.id),
            "email": parent.email,
            "display_name": parent.display_name,
        },
    }


@app.post("/auth/forgot-password")
def auth_forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    reset_token = svc.create_password_reset_token(db, body.email, ttl_sec=3600)

    reset_base_url = os.environ.get("SAFEEAR_PASSWORD_RESET_URL")
    if reset_token and reset_base_url:
        reset_link = f"{reset_base_url}?token={reset_token}"
        _send_password_reset_email(body.email, reset_link)

    response: Dict[str, Any] = {
        "status": "ok",
        "message": "If this email is registered, a reset link has been issued.",
    }
    if reset_token and os.environ.get("SAFEEAR_DEBUG_RETURN_RESET_TOKEN", "false").lower() == "true":
        response["reset_token"] = reset_token
    if reset_token and not reset_base_url:
        response["note"] = "Set SAFEEAR_PASSWORD_RESET_URL to enable email reset links."
    return response


@app.post("/auth/reset-password")
def auth_reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    svc.reset_password_with_token(db, body.token, body.new_password)
    return {"status": "ok", "message": "password_updated"}


@app.post("/auth/refresh")
def auth_refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    from app.utils.prd_services import consume_refresh_token, make_jwt, save_refresh_token
    parent_id = consume_refresh_token(body.refresh_token)
    access_token = make_jwt(parent_id)
    refresh_token = save_refresh_token(parent_id)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 604800,
    }


@app.api_route("/auth/logout", methods=["POST", "DELETE"])
def auth_logout(
    body: Optional[RefreshRequest] = None,
    refresh_token: Optional[str] = None,
    db: Session = Depends(get_db),
):
    from app.utils.prd_services import revoke_refresh_token

    # Some mobile clients cannot reliably send a JSON body with DELETE.
    token = (body.refresh_token if body else None) or refresh_token
    if not token:
        raise HTTPException(status_code=422, detail="refresh_token_required")

    revoke_refresh_token(token)
    return {"status": "ok"}


@app.post("/enroll/speaker")
async def enroll_speaker(
    request: Request,
    display_name: str = Form(...),
    speaker_id: Optional[str] = Form(None),
    # Change 1: accept speaker role for 3-category decision logic
    role: str = Form("familiar_adult"),
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    upload = file or audio
    if upload is None:
        raise HTTPException(status_code=422, detail="missing_audio_file")

    # Validate role
    allowed_roles = {"familiar_adult", "child_self"}
    if role not in allowed_roles:
        raise HTTPException(status_code=422, detail=f"invalid_role: must be one of {sorted(allowed_roles)}")

    parent_id = current["parent_id"]
    # For updates, ensure the target speaker exists before processing audio.
    speaker = svc.get_speaker(db, parent_id, speaker_id) if speaker_id else None

    os.makedirs(TEMP_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        temp_path = tmp.name

    try:
        embs, stage_info = run_enroll_embedding(temp_path)
    except ValueError as exc:
        logger.warning(f"ENROLL_SPEAKER_FAILED | parent_id={parent_id} | reason={str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    # Only create a new speaker after enrollment audio is validated and embeddings are extracted.
    if speaker is None:
        speaker = svc.create_speaker(db, parent_id, display_name)

    # Persist speaker role (Change 1)
    if hasattr(speaker, "role"):
        speaker.role = role
        db.commit()
        db.refresh(speaker)

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
    _, quality_label = svc.compute_and_store_enrollment_quality(db, parent_id, str(speaker.id))

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
    row = svc.rename_speaker(db, current["parent_id"], speaker_id, body.display_name)
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

    raw = await profile_image.read()
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
def detect_location(body: DetectLocationRequest, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    lat = body.latitude if body.latitude is not None else body.lat
    lon = body.longitude if body.longitude is not None else body.lng
    battery_percent = body.battery_percent if body.battery_percent is not None else body.battery
    row = svc.update_device_location(db, current["parent_id"], body.device_id, lat, lon)
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
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    upload = file or audio
    if upload is None:
        raise HTTPException(status_code=422, detail="missing_audio_chunk")

    parent_id = current["parent_id"]
    normalized_chunk_id = (chunk_id or "").strip() or None
    if normalized_chunk_id:
        cached = _get_detect_chunk_cached_response(parent_id, device_id, normalized_chunk_id)
        if cached is not None:
            cached["idempotent_replay"] = True
            return cached

    raw = await upload.read()
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

    arr = normalize_waveform(arr)

    # Validate device role: only child_device can stream detection chunks
    device = svc.get_device(db, parent_id, device_id)
    if device.role != DeviceRole.child_device:
        raise HTTPException(status_code=403, detail="only_child_devices_can_stream_audio")

    # Mark device as active (for telemetry: is_online, last_seen_at)
    device = svc.update_device_activity(db, parent_id, device_id)

    reported_battery = battery_percent if battery_percent is not None else battery
    if reported_battery is not None:
        device = svc.update_device_battery(db, parent_id, device_id, reported_battery)
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
        _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, svc_result)
        svc_result.pop("session", None)
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
    allow_stranger_increment = (
        (tier2_category == "human_speech" and tier2_confidence >= STRANGER_MIN_TIER2_CONF)
        or (tier2_category == "uncertain" and tier2_confidence >= STRANGER_UNCERTAIN_CONF_FLOOR)
    )
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
                from app.services.alert_service import create_and_escalate_alert
                create_and_escalate_alert(db, parent_id, device_id, session, score=0.0, alert_type="device_obstructed")
                session.device_obstructed_alert_sent_at = now
    elif tier2_pass:
        session.obstruction_counter = 0

    if not tier3_pass:
        logger.warning(
            "DETECT_CHUNK_TIER3_UNAVAILABLE | parent_id=%s | device_id=%s | stage_decision=%s",
            parent_id,
            device_id,
            stage.get("decision", "unknown"),
        )
        response_payload = {
            "status": "ok",
            "decision": "hold_tier3_unavailable",
            "stranger_streak": session.stranger_streak,
            "score": score,
            "thresholds": {"t_high": T_HIGH, "t_low": T_LOW},
            "stage": stage,
            "alert_fired": False,
            "alert_id": None,
        }
        _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, response_payload)
        return response_payload

    if score >= T_HIGH_eff:
        session.familiar_recovery_streak += 1
        if session.stranger_streak > 0 and session.familiar_recovery_streak < FAMILIAR_RESET_MIN_STREAK:
            decision = "uncertain_recovery"
            logger.info(
                "SCORE_DECISION | score=%.4f >= t_high=%.4f but pending_familiar_recovery "
                "| decision=uncertain_recovery | recovery_streak=%s/%s | stranger_streak=%s",
                score, T_HIGH_eff,
                session.familiar_recovery_streak, FAMILIAR_RESET_MIN_STREAK, session.stranger_streak,
            )
        else:
            session.stranger_streak = 0
            session.stranger_weight_sum = 0.0
            session.familiar_recovery_streak = 0
            if closest_speaker_role == "child_self":
                # Change 1: child match — suppress streak, do NOT update grace window
                decision = "child_self_match"
                
                # Change 10: drift tracking
                session.child_drift_scores.append(score)
                # Flush if we have enough scores or it's been a while (handled at end of function or periodically)
                assert closest_speaker_id is not None
                drift_warning = svc.flush_child_drift_scores(db, parent_id, closest_speaker_id, session)
                if drift_warning:
                    # In a real system, you might emit a WS event or push notification here
                    logger.warning("DRIFT_WARNING | parent_id=%s | child_id=%s | drift threshold breached", parent_id, closest_speaker_id)
                
                logger.info(
                    "SCORE_DECISION | score=%.4f >= t_high_eff=%.4f | decision=child_self_match | streak_reset",
                    score, T_HIGH_eff,
                )
            else:
                session.last_familiar_ms = now
                session.last_familiar_speaker_id = str(closest_speaker_id) if closest_speaker_id else None
                svc.clear_stranger_identity(session)
                decision = "familiar"
                logger.info(
                    "SCORE_DECISION | score=%.4f >= t_high=%.4f | decision=familiar | streak_reset",
                    score, T_HIGH_eff,
                )

    elif score <= T_LOW_eff:
        session.familiar_recovery_streak = 0
        within_familiar_grace = (
            session.last_familiar_ms > 0
            and (now - session.last_familiar_ms) <= FAMILIAR_GRACE_SEC * 1000
        )
        same_recent_speaker = (
            bool(closest_speaker_id)
            and bool(session.last_familiar_speaker_id)
            and str(closest_speaker_id) == str(session.last_familiar_speaker_id)
        )
        mild_post_familiar_dip = score >= max(FAMILIAR_HOLD_FLOOR, T_LOW - FAMILIAR_GRACE_MARGIN)
        strong_human_signal = tier2_category == "human_speech" and tier2_confidence >= FAMILIAR_GRACE_MIN_CONF

        if within_familiar_grace and same_recent_speaker and session.stranger_streak == 0 and mild_post_familiar_dip and strong_human_signal:
            session.stranger_streak = 0
            session.stranger_weight_sum = 0.0
            decision = "uncertain_post_familiar"
            logger.info(
                f"SCORE_DECISION | score={score:.4f} <= t_low={T_LOW} but held_by_familiar_grace "
                f"| decision=uncertain_post_familiar | grace_sec={FAMILIAR_GRACE_SEC} | floor={FAMILIAR_HOLD_FLOOR}"
            )
        elif not allow_stranger_increment:
            decision = "uncertain_noise"
            logger.info(
                "SCORE_DECISION | score=%.4f <= t_low=%.2f but blocked_by_tier2_guard "
                "| decision=uncertain_noise | tier2_category=%s | tier2_confidence=%.4f",
                score,
                T_LOW,
                tier2_category,
                tier2_confidence,
            )
        else:
            session.stranger_streak += 1
            from app.services.decision_engine import _streak_weight
            session.stranger_weight_sum += _streak_weight(score, T_LOW_eff, T_HIGH_eff, tier2_confidence)
            decision = "stranger_candidate"
            svc.record_confirmed_stranger_window(session, current_stranger_centroid)
            logger.info(f"SCORE_DECISION | score={score:.4f} <= t_low={T_LOW} | decision=stranger_candidate | streak_incremented={session.stranger_streak} weight={session.stranger_weight_sum:.2f}")
    else:
        session.familiar_recovery_streak = 0
        soft_limit = T_LOW + SOFT_STRANGER_MARGIN
        likely_human_speech = tier2_category == "human_speech" and tier2_confidence >= SOFT_STRANGER_MIN_CONF

        if score <= soft_limit and likely_human_speech:
            session.stranger_streak += 1
            from app.services.decision_engine import _streak_weight
            session.stranger_weight_sum += _streak_weight(score, soft_limit, T_HIGH_eff, tier2_confidence)
            decision = "stranger_candidate_soft"
            svc.record_confirmed_stranger_window(session, current_stranger_centroid)
            logger.info(
                "SCORE_DECISION | soft_stranger_window | score=%.4f | soft_limit=%.4f | "
                "tier2_category=%s | tier2_confidence=%.4f | streak_incremented=%s weight=%.2f",
                score,
                soft_limit,
                tier2_category,
                tier2_confidence,
                session.stranger_streak,
                session.stranger_weight_sum,
            )
        else:
            decision = "uncertain"
            logger.info(f"SCORE_DECISION | t_low < score={score:.4f} < t_high | decision=uncertain")

    # For clearly low scores, fast-track only after at least N consecutive stranger windows.
    if score <= FORCE_ALERT_SCORE and session.stranger_streak >= FORCE_ALERT_MIN_STREAK:
        old_streak = session.stranger_streak
        session.stranger_weight_sum = max(session.stranger_weight_sum, 2.5)  # fast track weight
        logger.info(
            f"FAST_TRACK_ALERT | score={score:.4f} <= force_alert_score={FORCE_ALERT_SCORE} | "
            f"min_streak={FORCE_ALERT_MIN_STREAK} | weight_updated_to={session.stranger_weight_sum:.2f}"
        )

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

    # ------------------------------------------------------------------ #
    # Change 7: Distress alert path (parallel to identity streak)         #
    # Fires independently, bypassing the identity streak and debounce.    #
    # ------------------------------------------------------------------ #
    DISTRESS_ALERT_THRESHOLD = float(os.environ.get("SAFEEAR_DISTRESS_ALERT_THRESHOLD", "0.65"))
    DISTRESS_CONFIRM_WINDOWS = int(os.environ.get("SAFEEAR_DISTRESS_CONFIRM_WINDOWS", "2"))

    distress_score = float(distress_info.get("distress_score", 0.0))
    distress_class = distress_info.get("distress_class")
    if distress_score >= DISTRESS_ALERT_THRESHOLD:
        session.distress_streak = getattr(session, "distress_streak", 0) + 1
    else:
        session.distress_streak = 0

    if session.distress_streak >= DISTRESS_CONFIRM_WINDOWS and not alert_fired:
        distress_debounce_elapsed = (now - session.last_alert_ms) >= 10_000  # 10 s distress debounce
        if distress_debounce_elapsed:
            try:
                from app.services.alert_service import create_and_escalate_alert
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
            except Exception as _da_exc:
                logger.error("DISTRESS_ALERT_ERROR | parent_id=%s | reason=%s", parent_id, _da_exc)
    # ------------------------------------------------------------------ #
    # Identity Alert Triggering (Change 11 & Change 12)                  #
    # ------------------------------------------------------------------ #
    alert_weight_threshold = session.household_config.get("alert_weight_threshold", 2.5) if session.household_config else 2.5
    
    from app.services.decision_engine import _should_alert
    can_alert, block_reason = _should_alert(session, now)
    
    if not can_alert:
        alert_block_reason = block_reason
        session.stranger_streak = 0
        session.stranger_weight_sum = 0.0
        svc.clear_stranger_segment(session)
        svc.clear_stranger_identity(session)
        logger.info("ALERT_SUPPRESSED | block_reason=%s", block_reason)
    elif session.stranger_weight_sum >= alert_weight_threshold:
        debounce_elapsed = (now - session.last_alert_ms) >= DEBOUNCE_SEC * 1000
        if not debounce_elapsed:
            from app.services.decision_engine import _has_speaker_changed
            speaker_changed, similarity = _has_speaker_changed(
                session, current_stranger_centroid, float(os.environ.get("SAFEEAR_SPEAKER_CHANGE_THRESHOLD", "0.45"))
            )
            if speaker_changed:
                session.last_alert_ms = 0
                debounce_elapsed = True
                logger.info(
                    f"DEBOUNCE_RESET_ON_SPEAKER_CHANGE | old_speaker_reset | "
                    f"similarity={similarity:.4f} < {os.environ.get('SAFEEAR_SPEAKER_CHANGE_THRESHOLD', '0.45')}"
                )
        if debounce_elapsed:
            session.stranger_segment_ended_ms = now
            try:
                from app.services.alert_service import create_and_escalate_alert
                aid = create_and_escalate_alert(db, parent_id, device_id, session, score)
                if aid:
                    alert_fired = True
                    alert_id = aid
                    logger.info(f"IDENTITY_ALERT_FIRED | parent_id={parent_id} | device_id={device_id} | score={score:.4f} | alert_id={aid}")
            except Exception as e:
                logger.error(f"ALERT_ESCALATION_FAILED | parent_id={parent_id} | device_id={device_id} | exc={e}")
        else:
            alert_block_reason = "debounced"
            logger.info(
                "ALERT_BLOCKED | alert_id=None | reason=debounced | last_alert_ms=%s | now=%s | debounce_sec=%s",
                session.last_alert_ms, now, DEBOUNCE_SEC,
            )
    else:
        alert_block_reason = "insufficient_stranger_streak"
        logger.info(f"ALERT_BLOCKED | alert_id=None | reason=insufficient_stranger_streak | weight={session.stranger_weight_sum:.2f} | required={alert_weight_threshold}")

    response_payload = {
        "status": "ok",
        "decision": decision,
        "stranger_streak": session.stranger_streak,
        "score": score,
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
    _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, response_payload)
    return response_payload


@app.delete("/detect/session")
def stop_detect_session(device_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
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
    rows = svc.list_alerts(db, current["parent_id"], limit=limit, offset=offset, since_ms=since_ms)
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
    if os.environ.get("SAFEEAR_ENABLE_TEST_ALERT_ENDPOINT", "true").lower() != "true":
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        "DEVICE_REGISTRATION_START | parent_id=%s | role=%s | device_name=%s | device_token=%s",
        parent_id,
        role,
        device_name,
        device_token[:15] + "..." if device_token else None,
    )
    
    device = svc.create_device(db, parent_id, device_name, device_role, installation_id=installation_id, device_token=device_token)
    
    logger.info(
        "DEVICE_REGISTRATION_SUCCESS | device_id=%s | parent_id=%s | role=%s | device_name=%s",
        str(device.id),
        str(device.parent_id),
        device.role.value,
        device.device_name,
    )

    return {
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
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    parent_id = current["parent_id"]
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
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    started_at = time.perf_counter()
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
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    row = svc.update_device_token(db, current["parent_id"], device_id, body.device_token)
    return {"status": "ok", "device": _device_payload(row)}

@app.post("/devices/{device_id}/heartbeat")
def post_device_heartbeat(
    device_id: str,
    body: DeviceHeartbeatRequest,
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
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
        body.monitoring_enabled,
    )
    if _should_publish_device_status(current["parent_id"], device_id):
        publish_parent_event(current["parent_id"], "device_status_changed", _device_payload(row))
    return {"status": "ok", "device": _device_payload(row)}


@app.post("/devices/{device_id}/monitoring/ack")
def post_monitoring_ack(
    device_id: str,
    body: DeviceMonitoringAckRequest,
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
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
        monitoring_enabled=body.monitoring_enabled,
    )
    if _should_publish_device_status(current["parent_id"], device_id):
        publish_parent_event(current["parent_id"], "device_status_changed", _device_payload(row))
    return {"status": "acknowledged", "device": _device_payload(row)}


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
    elif query_token:
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
    return {
        "status": "ok",
        "metrics": {
            "active_ws_clients": active_clients,
            "event_queue_lag_ms": queue_lag_ms,
            "events_published": published_counts,
        },
    }


@app.post("/admin/calibrate-thresholds")
def calibrate_thresholds(body: CalibrateThresholdsRequest, current=Depends(get_current_parent)):
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
