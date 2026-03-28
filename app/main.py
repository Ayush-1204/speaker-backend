import io
import logging
import os
import shutil
import tempfile
import threading
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.models import Device, DeviceRole
from app.database.session import get_db, init_db
from app.utils import prd_services_db as svc
from app.utils.notification_worker import escalate_alert
from app.utils.prd_services import (
    DEBOUNCE_SEC,
    STRANGER_CONFIRM_COUNT,
    T_HIGH,
    T_LOW,
    WINDOW_SEC,
    HOP_SEC,
    append_frame,
    now_ms,
    should_hop,
)
from app.utils.verification_pipeline import run_enroll_embedding

ALERT_TRIGGER_STREAK = int(os.environ.get("SAFEEAR_ALERT_TRIGGER_STREAK", str(STRANGER_CONFIRM_COUNT)))
FORCE_ALERT_SCORE = float(os.environ.get("SAFEEAR_FORCE_ALERT_SCORE", "0.15"))
FORCE_ALERT_MIN_STREAK = int(os.environ.get("SAFEEAR_FORCE_ALERT_MIN_STREAK", "2"))
FAMILIAR_GRACE_SEC = int(os.environ.get("SAFEEAR_FAMILIAR_GRACE_SEC", "20"))
FAMILIAR_HOLD_FLOOR = float(os.environ.get("SAFEEAR_FAMILIAR_HOLD_FLOOR", "0.30"))

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s"
)

TEMP_DIR = os.path.join("app", "data", "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)


@app.on_event("startup")
def startup_event() -> None:
    init_db()


class GoogleAuthRequest(BaseModel):
    id_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


class RenameSpeakerRequest(BaseModel):
    display_name: str


class FlagFamiliarRequest(BaseModel):
    display_name: str
    speaker_id: Optional[str] = None


class TestFireAlertRequest(BaseModel):
    device_id: str
    confidence_score: float = 0.05
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class DetectLocationRequest(BaseModel):
    device_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


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


def _verify_google_like_token(id_token: str) -> Dict[str, Any]:
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    allow_dev = os.environ.get("SAFEEAR_ALLOW_DEV_GOOGLE_TOKEN", "true").lower() == "true"

    if id_token.startswith("dev:"):
        if not allow_dev:
            raise HTTPException(status_code=401, detail="dev_token_disabled")
        dev_sub = id_token.split(":", 1)[1].strip()
        if not dev_sub:
            raise HTTPException(status_code=401, detail="invalid_dev_token")
        return {"sub": dev_sub, "email": f"{dev_sub}@dev.local", "name": dev_sub}

    if not client_id:
        raise HTTPException(status_code=500, detail="missing_google_client_id")

    try:
        import importlib

        grequests = importlib.import_module("google.auth.transport.requests")
        gid = importlib.import_module("google.oauth2.id_token")
        info = gid.verify_oauth2_token(id_token, grequests.Request(), client_id)
        return {"sub": info.get("sub"), "email": info.get("email"), "name": info.get("name")}
    except Exception as exc:
        raise HTTPException(status_code=401, detail="invalid_google_id_token") from exc


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


@app.delete("/auth/logout")
def auth_logout(body: RefreshRequest, db: Session = Depends(get_db)):
    from app.utils.prd_services import revoke_refresh_token
    revoke_refresh_token(body.refresh_token)
    return {"status": "ok"}


@app.post("/enroll/speaker")
async def enroll_speaker(
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

    for emb in embs:
        svc.save_speaker_embedding(db, parent_id, str(speaker.id), emb)

    return {
        "status": "enrolled",
        "speaker_id": str(speaker.id),
        "display_name": speaker.display_name,
        "samples_saved": len(embs),
        "embedding_dim": int(embs[0].shape[0]),
        "stages": stage_info,
    }


@app.get("/enroll/speakers")
def get_enrolled_speakers(current=Depends(get_current_parent), db: Session = Depends(get_db)):
    rows = svc.list_speakers(db, current["parent_id"])
    return {
        "items": [
            {
                "id": str(row.id),
                "parent_id": str(row.parent_id),
                "display_name": row.display_name,
                "sample_count": row.sample_count,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]
    }


@app.patch("/enroll/speakers/{speaker_id}")
def patch_speaker(
    speaker_id: str, body: RenameSpeakerRequest, current=Depends(get_current_parent), db: Session = Depends(get_db)
):
    row = svc.rename_speaker(db, current["parent_id"], speaker_id, body.display_name)
    return {
        "status": "ok",
        "speaker": {
            "id": str(row.id),
            "parent_id": str(row.parent_id),
            "display_name": row.display_name,
            "sample_count": row.sample_count,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        },
    }


@app.delete("/enroll/speakers/{speaker_id}")
def remove_speaker(speaker_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    svc.delete_speaker(db, current["parent_id"], speaker_id)
    return {"status": "deleted", "speaker_id": speaker_id}


@app.post("/detect/location")
def detect_location(body: DetectLocationRequest, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    lat = body.latitude if body.latitude is not None else body.lat
    lon = body.longitude if body.longitude is not None else body.lng
    svc.update_device_location(db, current["parent_id"], body.device_id, lat, lon)
    svc.update_session_location(current["parent_id"], body.device_id, lat, lon)
    return {"status": "ok", "device_id": body.device_id, "latitude": lat, "longitude": lon}


@app.post("/detect/chunk")
async def detect_chunk(
    device_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    current=Depends(get_current_parent),
    db: Session = Depends(get_db),
):
    from app.utils.prd_services import save_alert_clip
    upload = file or audio
    if upload is None:
        raise HTTPException(status_code=422, detail="missing_audio_chunk")

    raw = await upload.read()
    if not raw:
        logger.warning(
            "DETECT_CHUNK_EMPTY | filename=%s | content_type=%s | bytes=0",
            getattr(upload, "filename", None),
            getattr(upload, "content_type", None),
        )
        return {"status": "no_hop", "reason": "empty_chunk", "chunk_samples": 0}

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
        return {"status": "no_hop", "reason": "empty_chunk", "chunk_samples": 0}

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

    parent_id = current["parent_id"]
    # Validate device role: only child_device can stream detection chunks
    device = svc.get_device(db, parent_id, device_id)
    if device.role != DeviceRole.child_device:
        raise HTTPException(status_code=403, detail="only_child_devices_can_stream_audio")

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

    ring = append_frame(session, arr)
    window_samples = int(WINDOW_SEC * session.sr)
    if ring.shape[0] < window_samples:
        ack_msg = f"warming_up | ring_samples={int(ring.shape[0])} | required={window_samples}"
        logger.info(f"ACK | {ack_msg}")
        return {
            "status": "warming_up",
            "ring_samples": int(ring.shape[0]),
            "required_samples": window_samples,
        }

    if not should_hop(len(arr), session.sr):
        ack_msg = f"no_hop | chunk_samples={int(len(arr))} | required_hop={int(HOP_SEC * session.sr)}"
        logger.info(f"ACK | {ack_msg}")
        return {"status": "no_hop", "reason": "chunk_too_small", "chunk_samples": int(len(arr))}

    window = ring[-window_samples:]
    stage = svc.evaluate_window(db, parent_id, window, session.sr)
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
    logger.info(f"STAGE_RESULT | score={score:.4f} | tier1_vad={tier1_pass} | tier2={tier2_pass} | tier3={tier3_pass} | stage_decision={stage.get('decision', 'unknown')}")

    if score >= T_HIGH:
        session.stranger_streak = 0
        session.last_familiar_ms = now
        session.last_familiar_speaker_id = str(closest_speaker_id) if closest_speaker_id else None
        decision = "familiar"
        logger.info(f"SCORE_DECISION | score={score:.4f} >= t_high={T_HIGH} | decision=familiar | streak_reset")
    elif score <= T_LOW:
        within_familiar_grace = (
            session.last_familiar_ms > 0
            and (now - session.last_familiar_ms) <= FAMILIAR_GRACE_SEC * 1000
        )
        same_recent_speaker = (
            bool(closest_speaker_id)
            and bool(session.last_familiar_speaker_id)
            and str(closest_speaker_id) == str(session.last_familiar_speaker_id)
        )

        if within_familiar_grace and same_recent_speaker and score >= FAMILIAR_HOLD_FLOOR:
            session.stranger_streak = 0
            decision = "uncertain_post_familiar"
            logger.info(
                f"SCORE_DECISION | score={score:.4f} <= t_low={T_LOW} but held_by_familiar_grace "
                f"| decision=uncertain_post_familiar | grace_sec={FAMILIAR_GRACE_SEC} | floor={FAMILIAR_HOLD_FLOOR}"
            )
        else:
            session.stranger_streak += 1
            decision = "stranger_candidate"
            logger.info(f"SCORE_DECISION | score={score:.4f} <= t_low={T_LOW} | decision=stranger_candidate | streak_incremented={session.stranger_streak}")
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

    if session.stranger_streak >= ALERT_TRIGGER_STREAK:
        if now - session.last_alert_ms >= DEBOUNCE_SEC * 1000:
            clip_path = save_alert_clip(parent_id, session, ring)
            row = svc.create_alert(
                db=db,
                parent_id=parent_id,
                device_id=device_id,
                audio_clip_path=clip_path,
                confidence_score=score,
                latitude=session.lat,
                longitude=session.lon,
            )
            alert_fired = True
            alert_id = str(row.id)
            session.last_alert_ms = now
            session.stranger_streak = 0
            logger.info(f"ALERT_FIRED | alert_id={alert_id} | score={score:.4f} | streak_satisfied={ALERT_TRIGGER_STREAK} | clip_path={clip_path}")
            parent = svc.get_parent(db, parent_id)
            # Prefer parent-device FCM token (fallback to stored parent token).
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
            parent_fcm_token = parent_device.device_token if parent_device else parent.fcm_token
            logger.info(f"ALERT_ESCALATION_START | alert_id={alert_id} | parent_email={parent.email} | fcm_token_available={bool(parent_fcm_token)}")
            # Wire background thread for notification escalation
            thread = threading.Thread(
                target=escalate_alert,
                args=(
                    parent.email,
                    parent.phone_number,
                    parent_fcm_token,
                    alert_id,
                    session.lat,
                    session.lon,
                    f"/alerts/{alert_id}/clip",
                    score,
                ),
                daemon=True,
            )
            thread.start()
        else:
            alert_block_reason = "debounced"
            logger.info(f"ALERT_BLOCKED | alert_id=None | reason=debounced | last_alert_ms={session.last_alert_ms} | now={now} | debounce_sec={DEBOUNCE_SEC}")
    else:
        alert_block_reason = "insufficient_stranger_streak"
        logger.info(f"ALERT_BLOCKED | alert_id=None | reason=insufficient_stranger_streak | current_streak={session.stranger_streak} | required={ALERT_TRIGGER_STREAK}")

    return {
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
            "debounce_sec": DEBOUNCE_SEC,
            "block_reason": alert_block_reason,
        },
        "stage": stage,
        "alert_fired": alert_fired,
        "alert_id": alert_id,
    }


@app.delete("/detect/session")
def stop_detect_session(device_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    svc.stop_session(current["parent_id"], device_id)
    return {"status": "stopped", "device_id": device_id}


@app.get("/alerts")
def get_alert_history(limit: int = 50, offset: int = 0, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    rows = svc.list_alerts(db, current["parent_id"], limit=limit, offset=offset)
    return {
        "items": [
            {
                "id": str(row.id),
                "parent_id": str(row.parent_id),
                "device_id": str(row.device_id),
                "timestamp": row.timestamp,
                "confidence_score": row.confidence_score,
                "audio_clip_path": row.audio_clip_path,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "lat": row.latitude,
                "lng": row.longitude,
                "acknowledged_at": row.acknowledged_at,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]
    }


@app.post("/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    row = svc.ack_alert(db, current["parent_id"], alert_id)
    alert_payload = {
        "id": str(row.id),
        "parent_id": str(row.parent_id),
        "device_id": str(row.device_id),
        "timestamp": row.timestamp,
        "confidence_score": row.confidence_score,
        "audio_clip_path": row.audio_clip_path,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "lat": row.latitude,
        "lng": row.longitude,
        "acknowledged_at": row.acknowledged_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
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

    return {
        "status": "test_alert_fired",
        "alert_id": str(row.id),
        "device_id": str(row.device_id),
        "confidence_score": row.confidence_score,
        "clip_url": f"/alerts/{row.id}/clip",
    }


@app.post("/alerts/{alert_id}/flag-familiar")
def flag_alert_as_familiar(
    alert_id: str,
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

    return {
        "status": "flagged_familiar",
        "source_alert_id": alert_id,
        "speaker_id": str(speaker.id),
        "display_name": speaker.display_name,
        "samples_saved": len(embs),
        "stages": stage_info,
    }


@app.get("/alerts/{alert_id}/clip")
def get_alert_clip(alert_id: str, current=Depends(get_current_parent), db: Session = Depends(get_db)):
    row = svc.get_alert(db, current["parent_id"], alert_id)
    path = row.audio_clip_path
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="audio_clip_not_found")
    return FileResponse(path, media_type="audio/wav", filename=os.path.basename(path))


@app.post("/devices")
def create_device(
    device_name: str = Form(...),
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
    device = svc.create_device(db, parent_id, device_name, device_role, device_token)

    return {
        "status": "created",
        "device": {
            "id": str(device.id),
            "parent_id": str(device.parent_id),
            "device_name": device.device_name,
            "role": device.role.value,
            "device_token": device.device_token
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return {
        "name": "SafeEar Backend",
        "version": "2.0.0",
        "message": "PRD-aligned API is running",
    }
