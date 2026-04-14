import io
import asyncio
import logging
import json
import os
import shutil
import tempfile
import threading
import warnings
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import soundfile as sf
import librosa
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketState

try:
    import redis
except Exception:  # pragma: no cover - optional dependency behavior
    redis = None

from app.database.models import Device, DeviceRole, Parent, EnrolledSpeaker
from app.database.session import SessionLocal, get_db, init_db
from app.utils import prd_services_db as svc
from app.utils import feature_extractor as feature_extractor_module
from app.utils import yamnet_classifier as yamnet_classifier_module
from app.utils import vad as vad_module
from app.utils.audio_preprocess import normalize_waveform
from app.utils.notification_worker import escalate_alert, send_fcm_push
from app.utils.prd_services import (
    DEBOUNCE_SEC,
    STRANGER_CONFIRM_COUNT,
    T_HIGH,
    T_LOW,
    WINDOW_SEC,
    HOP_SEC,
    now_ms,
)
from app.utils.verification_pipeline import run_enroll_embedding

ALERT_TRIGGER_STREAK = int(os.environ.get("SAFEEAR_ALERT_TRIGGER_STREAK", str(STRANGER_CONFIRM_COUNT)))
FORCE_ALERT_SCORE = float(os.environ.get("SAFEEAR_FORCE_ALERT_SCORE", "0.15"))
FORCE_ALERT_MIN_STREAK = int(os.environ.get("SAFEEAR_FORCE_ALERT_MIN_STREAK", "2"))
FAMILIAR_GRACE_SEC = int(os.environ.get("SAFEEAR_FAMILIAR_GRACE_SEC", "20"))
FAMILIAR_HOLD_FLOOR = float(os.environ.get("SAFEEAR_FAMILIAR_HOLD_FLOOR", "0.30"))
FAMILIAR_GRACE_MARGIN = float(os.environ.get("SAFEEAR_FAMILIAR_GRACE_MARGIN", "0.05"))
FAMILIAR_GRACE_MIN_CONF = float(os.environ.get("SAFEEAR_FAMILIAR_GRACE_MIN_CONF", "0.65"))
FAMILIAR_RESET_MIN_STREAK = int(os.environ.get("SAFEEAR_FAMILIAR_RESET_MIN_STREAK", "2"))
SOFT_STRANGER_MARGIN = float(os.environ.get("SAFEEAR_SOFT_STRANGER_MARGIN", "0.08"))
SOFT_STRANGER_MIN_CONF = float(os.environ.get("SAFEEAR_SOFT_STRANGER_MIN_CONF", "0.30"))
STRANGER_PREROLL_SEC = float(os.environ.get("SAFEEAR_STRANGER_PREROLL_SEC", "5.0"))
SPEAKER_CHANGE_THRESHOLD = float(os.environ.get("SAFEEAR_SPEAKER_CHANGE_THRESHOLD", "0.30"))

# Keep TF/Google runtime logs readable in local dev.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.lite\.Interpreter is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Google will stop supporting.*",
    category=FutureWarning,
)

app = FastAPI(title="SafeEar Backend", version="2.0.0")

# Configure logging for score and acknowledgement tracking
logger = logging.getLogger(__name__)


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload)


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    handler = root.handlers[0]
    log_format = (os.environ.get("LOG_FORMAT") or "").strip().lower()
    if log_format == "json":
        handler.setFormatter(_JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s"))
    root.setLevel(logging.INFO)


_configure_logging()

WS_PING_INTERVAL_SEC = int(os.environ.get("SAFEEAR_WS_PING_INTERVAL_SEC", "20"))
WS_CLIENT_QUEUE_MAX = int(os.environ.get("SAFEEAR_WS_CLIENT_QUEUE_MAX", "256"))

_WS_LOOP: Optional[asyncio.AbstractEventLoop] = None
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
DEVICE_STATUS_EVENT_MIN_INTERVAL_MS = int(os.environ.get("SAFEEAR_DEVICE_STATUS_EVENT_MIN_INTERVAL_MS", "800"))
DETECT_CHUNK_IDEMPOTENCY_WINDOW_MS = int(os.environ.get("SAFEEAR_DETECT_CHUNK_IDEMPOTENCY_WINDOW_MS", "120000"))
_RECENT_DETECT_CHUNK_RESULTS: Dict[str, Tuple[int, Dict[str, Any]]] = {}
_RECENT_DETECT_CHUNK_RESULTS_LOCK = threading.Lock()

_REDIS_CLIENT = None
_CACHE_WARMUP_SUMMARY: Dict[str, Any] = {
    "parents_loaded": 0,
    "cached_parents": 0,
    "speakers_loaded": 0,
    "embeddings_loaded": 0,
    "failed_files": [],
    "parents_with_zero_embeddings": [],
}


def _normalized_embedding(vec: Any) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return None
    return arr / norm


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = _normalized_embedding(a)
    bn = _normalized_embedding(b)
    if an is None or bn is None:
        return 0.0
    return float(np.dot(an, bn))


def _has_speaker_changed(session: svc.SessionState, current_embedding: Optional[np.ndarray]) -> Tuple[bool, Optional[float]]:
    if session.active_stranger_embedding is None:
        return False, None
    current = _normalized_embedding(current_embedding) if current_embedding is not None else None
    if current is None:
        return False, None
    similarity = _cos_sim(session.active_stranger_embedding, current)
    return similarity < SPEAKER_CHANGE_THRESHOLD, similarity


def _log_pipeline_decision(
    *,
    stage: str,
    device_id: str,
    parent_id: str,
    result: str,
    score: Optional[float],
    streak: Optional[int],
    latency_ms: float,
) -> None:
    payload = {
        "event": "pipeline_decision",
        "stage": stage,
        "device_id": str(device_id),
        "parent_id": str(parent_id),
        "result": result,
        "score": float(score) if score is not None else None,
        "streak": int(streak) if streak is not None else None,
        "latency_ms": float(latency_ms),
    }
    logger.debug(json.dumps(payload))


def _find_eer_threshold(familiar_scores: List[float], stranger_scores: List[float]) -> Tuple[float, float]:
    all_scores = sorted(set(float(s) for s in (familiar_scores + stranger_scores)))
    if not all_scores:
        raise HTTPException(status_code=422, detail="scores_required")

    best_threshold = all_scores[0]
    best_far = 1.0
    best_frr = 1.0
    best_gap = float("inf")
    for t in all_scores:
        far = sum(1 for s in stranger_scores if float(s) >= t) / max(1, len(stranger_scores))
        frr = sum(1 for s in familiar_scores if float(s) < t) / max(1, len(familiar_scores))
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best_threshold = t
            best_far = far
            best_frr = frr

    eer_percent = ((best_far + best_frr) / 2.0) * 100.0
    return float(best_threshold), float(eer_percent)


def _get_redis_client():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    if redis is None:
        return None

    redis_url = (os.environ.get("SAFEEAR_REDIS_URL") or "").strip() or "redis://localhost:6379/0"
    try:
        client = redis.Redis.from_url(redis_url, decode_responses=False, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        _REDIS_CLIENT = client
        return _REDIS_CLIENT
    except Exception as exc:
        logger.warning("CACHE_WARMUP_SKIPPED | reason=redis_unavailable | error=%s", str(exc))
        return None


def _warmup_embeddings_to_redis() -> None:
    global _CACHE_WARMUP_SUMMARY

    failed_files: List[str] = []
    parents_with_zero_embeddings: List[str] = []
    parents_loaded = 0
    cached_parents = 0
    speakers_loaded = 0
    embeddings_loaded = 0

    client = _get_redis_client()
    if client is None:
        _CACHE_WARMUP_SUMMARY = {
            "parents_loaded": 0,
            "cached_parents": 0,
            "speakers_loaded": 0,
            "embeddings_loaded": 0,
            "failed_files": ["redis_unavailable"],
            "parents_with_zero_embeddings": [],
        }
        logger.info(json.dumps({"event": "cache_warmup_complete", **_CACHE_WARMUP_SUMMARY}))
        return

    with SessionLocal() as db:
        parents = db.query(Parent).all()
        parents_loaded = len(parents)

        for parent in parents:
            parent_id = str(parent.id)
            parent_embedding_count = 0

            speakers = svc.list_speakers(db, parent_id)
            enrolled_count = (
                db.query(EnrolledSpeaker)
                .filter(EnrolledSpeaker.parent_id == parent.id)
                .count()
            )
            for speaker in speakers:
                speaker_id = str(speaker.id)
                speaker_dir = svc.get_speaker_embedding_dir(parent_id, speaker_id)
                if not os.path.isdir(speaker_dir):
                    reason = f"{speaker_dir}:missing_dir"
                    failed_files.append(reason)
                    logger.warning("CACHE_WARMUP_FILE_SKIPPED | file=%s | reason=missing_dir", speaker_dir)
                    continue

                speaker_embeddings = 0
                for fname in sorted(os.listdir(speaker_dir)):
                    if not fname.endswith(".npy"):
                        continue
                    fpath = os.path.join(speaker_dir, fname)
                    try:
                        arr = np.asarray(np.load(fpath), dtype=np.float32).reshape(-1)
                    except Exception as exc:
                        failed_files.append(f"{fpath}:load_failed")
                        logger.warning("CACHE_WARMUP_FILE_SKIPPED | file=%s | reason=load_failed | error=%s", fpath, str(exc))
                        continue

                    if arr.shape[0] != feature_extractor_module.config.EMBEDDING_DIM:
                        failed_files.append(f"{fpath}:invalid_dim_{arr.shape[0]}")
                        logger.warning(
                            "CACHE_WARMUP_FILE_SKIPPED | file=%s | reason=invalid_dim | dim=%s",
                            fpath,
                            arr.shape[0],
                        )
                        continue

                    key = f"safeear:emb:{parent_id}:{speaker_id}:{fname}"
                    client.set(key, arr.astype(np.float32).tobytes())
                    embeddings_loaded += 1
                    speaker_embeddings += 1

                if speaker_embeddings > 0:
                    speakers_loaded += 1
                    parent_embedding_count += speaker_embeddings

            if enrolled_count > 0 and parent_embedding_count == 0:
                parents_with_zero_embeddings.append(parent_id)
            if parent_embedding_count > 0:
                cached_parents += 1

    _CACHE_WARMUP_SUMMARY = {
        "parents_loaded": parents_loaded,
        "cached_parents": cached_parents,
        "speakers_loaded": speakers_loaded,
        "embeddings_loaded": embeddings_loaded,
        "failed_files": failed_files,
        "parents_with_zero_embeddings": parents_with_zero_embeddings,
    }
    logger.info(json.dumps({"event": "cache_warmup_complete", **_CACHE_WARMUP_SUMMARY}))


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
        "timestamp_ms": now_ms(),
        "seq": seq,
    }
    queued_at_ms = now_ms()

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


def _should_publish_device_status(parent_id: str, device_id: str) -> bool:
    key = f"{parent_id}:{device_id}"
    now = now_ms()
    with _WS_STATE_LOCK:
        last = _DEVICE_STATUS_EVENT_LAST_SENT.get(key)
        if last is not None and (now - last) < DEVICE_STATUS_EVENT_MIN_INTERVAL_MS:
            return False
        _DEVICE_STATUS_EVENT_LAST_SENT[key] = now
        return True


def _make_detect_chunk_cache_key(parent_id: str, device_id: str, chunk_id: str) -> str:
    return f"{parent_id}:{device_id}:{chunk_id}"


def _get_detect_chunk_cached_response(parent_id: str, device_id: str, chunk_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not chunk_id:
        return None

    key = _make_detect_chunk_cache_key(parent_id, device_id, chunk_id)
    now = now_ms()
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
        "profile_image_url": _speaker_avatar_url(request, row.id) if avatar_exists else None,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


@app.on_event("startup")
def startup_event() -> None:
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
    # 1) Preferred path: decode containerized audio (wav/ogg/webm/etc.) with soundfile.
    try:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        return np.asarray(arr), int(sr), "soundfile"
    except Exception:
        pass

    # 2) Fallback path: librosa may decode formats unavailable to soundfile.
    try:
        arr, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return np.asarray(arr, dtype=np.float32), int(sr), "librosa"
    except Exception:
        pass

    # 3) Last resort: treat payload as raw PCM16LE mono.
    if len(raw) >= 2 and (len(raw) % 2 == 0):
        raw_sr = int(os.environ.get("SAFEEAR_RAW_PCM_SR", "16000"))
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if arr.size > 0:
            return arr, raw_sr, "raw_pcm16le"

    raise ValueError("unsupported_audio_chunk_format")


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
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    upload = file or audio
    if upload is None:
        raise HTTPException(status_code=422, detail="missing_audio_file")

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
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    # Only create a new speaker after enrollment audio is validated and embeddings are extracted.
    if speaker is None:
        speaker = svc.create_speaker(db, parent_id, display_name)

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

    quality_score, quality_label = svc.compute_and_store_enrollment_quality(db, parent_id, str(speaker.id))

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
        "items": refreshed_items,
        "stages": stage_info,
    }
    if quality_label == "poor":
        response["warning"] = "Low enrollment quality. Try recording in a quieter space, speaking naturally, or adding more samples."
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

    session = svc.get_or_create_session(parent_id, device_id)
    if latitude is not None and longitude is not None:
        session.lat = latitude
        session.lon = longitude
    elif session.lat is None and session.lon is None:
        # Fallback to the last location persisted for this device.
        if device.last_location_lat is not None and device.last_location_lon is not None:
            session.lat = float(device.last_location_lat)
            session.lon = float(device.last_location_lon)

    inference_buffer = svc.append_frame(session, arr)
    window_samples = int(WINDOW_SEC * session.sr)
    if inference_buffer.shape[0] < window_samples:
        ack_msg = f"warming_up | ring_samples={int(inference_buffer.shape[0])} | required={window_samples}"
        logger.info(f"ACK | {ack_msg}")
        response_payload = {
            "status": "warming_up",
            "ring_samples": int(inference_buffer.shape[0]),
            "required_samples": window_samples,
        }
        _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, response_payload)
        return response_payload

    if not svc.should_hop(len(arr), session.sr):
        ack_msg = f"no_hop | chunk_samples={int(len(arr))} | required_hop={int(HOP_SEC * session.sr)}"
        logger.info(f"ACK | {ack_msg}")
        response_payload = {"status": "no_hop", "reason": "chunk_too_small", "chunk_samples": int(len(arr))}
        _remember_detect_chunk_response(parent_id, device_id, normalized_chunk_id, response_payload)
        return response_payload

    window = inference_buffer[-window_samples:]
    stage = svc.evaluate_window(db, parent_id, window, session.sr)
    current_stranger_centroid = _normalized_embedding(stage.pop("_query_centroid", None))
    score = float(stage["tier3"].get("score", 0.0))
    closest_speaker_id = stage.get("tier3", {}).get("closest_speaker_id")
    now = now_ms()

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

    if score >= T_HIGH:
        session.familiar_recovery_streak += 1
        if session.stranger_streak > 0 and session.familiar_recovery_streak < FAMILIAR_RESET_MIN_STREAK:
            decision = "uncertain_recovery"
            logger.info(
                "SCORE_DECISION | score=%.4f >= t_high=%.2f but pending_familiar_recovery "
                "| decision=uncertain_recovery | recovery_streak=%s/%s | stranger_streak=%s",
                score,
                T_HIGH,
                session.familiar_recovery_streak,
                FAMILIAR_RESET_MIN_STREAK,
                session.stranger_streak,
            )
        else:
            session.stranger_streak = 0
            session.last_familiar_ms = now
            session.last_familiar_speaker_id = str(closest_speaker_id) if closest_speaker_id else None
            svc.clear_stranger_identity(session)
            decision = "familiar"
            logger.info(f"SCORE_DECISION | score={score:.4f} >= t_high={T_HIGH} | decision=familiar | streak_reset")
    elif score <= T_LOW:
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
            decision = "uncertain_post_familiar"
            logger.info(
                f"SCORE_DECISION | score={score:.4f} <= t_low={T_LOW} but held_by_familiar_grace "
                f"| decision=uncertain_post_familiar | grace_sec={FAMILIAR_GRACE_SEC} | floor={FAMILIAR_HOLD_FLOOR}"
            )
        else:
            session.stranger_streak += 1
            decision = "stranger_candidate"
            svc.record_confirmed_stranger_window(session, current_stranger_centroid)
            logger.info(f"SCORE_DECISION | score={score:.4f} <= t_low={T_LOW} | decision=stranger_candidate | streak_incremented={session.stranger_streak}")
    else:
        session.familiar_recovery_streak = 0
        soft_limit = T_LOW + SOFT_STRANGER_MARGIN
        likely_human_speech = tier2_category in {"human_speech", "uncertain"} and tier2_confidence >= SOFT_STRANGER_MIN_CONF

        if score <= soft_limit and likely_human_speech:
            session.stranger_streak += 1
            decision = "stranger_candidate_soft"
            svc.record_confirmed_stranger_window(session, current_stranger_centroid)
            logger.info(
                "SCORE_DECISION | soft_stranger_window | score=%.4f | soft_limit=%.4f | "
                "tier2_category=%s | tier2_confidence=%.4f | streak_incremented=%s",
                score,
                soft_limit,
                tier2_category,
                tier2_confidence,
                session.stranger_streak,
            )
        else:
            decision = "uncertain"
            logger.info(f"SCORE_DECISION | t_low < score={score:.4f} < t_high | decision=uncertain")

    # For clearly low scores, fast-track only after at least N consecutive stranger windows.
    if score <= FORCE_ALERT_SCORE and session.stranger_streak >= FORCE_ALERT_MIN_STREAK:
        old_streak = session.stranger_streak
        session.stranger_streak = max(session.stranger_streak, ALERT_TRIGGER_STREAK)
        logger.info(
            f"FAST_TRACK_ALERT | score={score:.4f} <= force_alert_score={FORCE_ALERT_SCORE} | "
            f"min_streak={FORCE_ALERT_MIN_STREAK} | streak_updated={old_streak}->{session.stranger_streak}"
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

    if session.stranger_streak >= ALERT_TRIGGER_STREAK:
        debounce_elapsed = (now - session.last_alert_ms) >= DEBOUNCE_SEC * 1000
        if not debounce_elapsed:
            speaker_changed, similarity = _has_speaker_changed(session, current_stranger_centroid)
            if speaker_changed:
                session.last_alert_ms = 0
                debounce_elapsed = True
                logger.info(
                    "DEBOUNCE_RESET_SPEAKER_CHANGE | parent_id=%s | device_id=%s | similarity=%.4f | threshold=%.2f",
                    parent_id,
                    device_id,
                    similarity if similarity is not None else -1.0,
                    SPEAKER_CHANGE_THRESHOLD,
                )

        if debounce_elapsed:
            session.stranger_segment_ended_ms = now
            clip_waveform = svc.get_clip_buffer_waveform(session, inference_buffer)
            clip_path = svc.save_alert_clip(parent_id, session, clip_waveform)
            row = svc.create_alert(
                db=db,
                parent_id=parent_id,
                device_id=device_id,
                audio_clip_path=clip_path,
                confidence_score=score,
                latitude=session.lat,
                longitude=session.lon,
            )
            publish_parent_event(parent_id, "alert_created", _alert_response_payload(row))
            alert_fired = True
            alert_id = str(row.id)
            session.last_alert_ms = now
            session.stranger_streak = 0
            if not session.recent_confirmed_stranger_embeddings:
                svc.record_confirmed_stranger_window(session, current_stranger_centroid)
            svc.set_active_stranger_embedding(session)
            logger.info(
                "ALERT_FIRED | alert_id=%s | score=%.4f | streak_satisfied=%s | clip_path=%s | segment_started_ms=%s | segment_ended_ms=%s | segment_samples=%s",
                alert_id,
                score,
                ALERT_TRIGGER_STREAK,
                clip_path,
                session.stranger_segment_started_ms or None,
                session.stranger_segment_ended_ms or None,
                int(np.asarray(clip_waveform).reshape(-1).shape[0]),
            )
            parent = svc.get_parent(db, parent_id)
            # Stranger alerts must only target the parent device token.
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
            token_source = "parent_device" if parent_fcm_token else "none"
            logger.info(
                "ALERT_ESCALATION_START | alert_id=%s | parent_email=%s | fcm_token_available=%s | token_source=%s",
                alert_id,
                parent.email,
                bool(parent_fcm_token),
                token_source,
            )
            # Wire background thread for notification escalation
            def _run_escalation() -> None:
                try:
                    escalate_alert(
                        parent.email,
                        parent.phone_number,
                        parent_fcm_token,
                        alert_id,
                        device_id,
                        _dt_to_epoch_ms(row.timestamp) or now_ms(),
                        session.lat,
                        session.lon,
                        f"/alerts/{alert_id}/clip",
                        score,
                        recipient_role="parent_device",
                    )
                except Exception:
                    logger.exception("ALERT_ESCALATION_FAILED | alert_id=%s | device_id=%s", alert_id, device_id)

            thread = threading.Thread(
                target=_run_escalation,
                daemon=True,
            )
            thread.start()
            svc.clear_stranger_segment(session)
            svc.clear_clip_buffer(session)
        else:
            alert_block_reason = "debounced"
            logger.info(
                "ALERT_BLOCKED | alert_id=None | reason=debounced | last_alert_ms=%s | now=%s | debounce_sec=%s",
                session.last_alert_ms,
                now,
                DEBOUNCE_SEC,
            )
    else:
        alert_block_reason = "insufficient_stranger_streak"
        logger.info(f"ALERT_BLOCKED | alert_id=None | reason=insufficient_stranger_streak | current_streak={session.stranger_streak} | required={ALERT_TRIGGER_STREAK}")

    response_payload = {
        "status": "ok",
        "decision": decision,
        "stranger_streak": session.stranger_streak,
        "score": score,
        "thresholds": {"t_high": T_HIGH, "t_low": T_LOW},
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
            parent_fcm_token,
            str(row.id),
            str(row.device_id),
            _dt_to_epoch_ms(row.timestamp) or now_ms(),
            row.latitude,
            row.longitude,
            float(row.confidence_score or 0.0),
            recipient_role="parent_device",
        )

    return {
        "status": "test_alert_fired",
        "alert_id": str(row.id),
        "device_id": str(row.device_id),
        "confidence_score": row.confidence_score,
        "timestamp_ms": _dt_to_epoch_ms(row.timestamp),
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
    if not clip_path or not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail="audio_clip_not_found")

    speaker = (
        svc.create_speaker(db, parent_id, body.display_name)
        if not body.speaker_id
        else svc.get_speaker(db, parent_id, body.speaker_id)
    )

    try:
        embs, stage_info = run_enroll_embedding(clip_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for emb in embs:
        svc.save_speaker_embedding(db, parent_id, str(speaker.id), emb)

    refreshed_rows = svc.list_speakers(db, parent_id)
    refreshed_items = [_speaker_payload(row, request, parent_id) for row in refreshed_rows]

    return {
        "status": "flagged_familiar",
        "source_alert_id": alert_id,
        "speaker_id": str(speaker.id),
        "display_name": speaker.display_name,
        "samples_saved": len(embs),
        "items": refreshed_items,
        "stages": stage_info,
    }


@app.get("/alerts/{alert_id}/clip")
def get_alert_clip(alert_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    row = svc.get_alert(db, current["parent_id"], alert_id)
    path = row.audio_clip_path
    # Older alerts may exist without persisted media. Return an empty response cleanly.
    if not path:
        return Response(status_code=204)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="audio_clip_not_found")
    return FileResponse(path, media_type="audio/wav", filename=os.path.basename(path))


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

            done, pending = await asyncio.wait(
                {queue_task, receiver_task},
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
                global _WS_LAST_QUEUE_LAG_MS
                _WS_LAST_QUEUE_LAG_MS = lag_ms
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
        queue_lag_ms = int(_WS_LAST_QUEUE_LAG_MS)
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
