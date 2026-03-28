"""PRD services with SQLAlchemy database backing and device role support."""

import logging
import os
import secrets
import shutil
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.models import Parent, Device, DeviceRole, EnrolledSpeaker, Alert as AlertModel
from app.utils.feature_extractor import embed_waveform_chunks
from app.utils.speech_gate import assess_speech_likeness
from app.utils.vad import get_speech_segments
from app.utils.yamnet_classifier import classify_audio_event

TENANTS_ROOT = os.path.join("app", "data", "tenants")

JWT_SECRET = os.environ.get("SAFEEAR_JWT_SECRET", "change-me-in-production")
JWT_EXPIRE_SEC = int(os.environ.get("SAFEEAR_JWT_EXPIRE_SEC", "604800"))

WINDOW_SEC = float(os.environ.get("SAFEEAR_WINDOW_SEC", "1.5"))
HOP_SEC = float(os.environ.get("SAFEEAR_HOP_SEC", "0.25"))
_REDIMNET_HUB_REPO = os.environ.get("REDIMNET_HUB_REPO", "PalabraAI/redimnet2")
_REDIMNET_HUB_ENTRY = os.environ.get("REDIMNET_HUB_ENTRY", "redimnet2")
_IS_REDIMNET2 = ("redimnet2" in _REDIMNET_HUB_REPO.lower()) or ("redimnet2" in _REDIMNET_HUB_ENTRY.lower())

_DEFAULT_T_HIGH = "0.68" if _IS_REDIMNET2 else "0.72"
_DEFAULT_T_LOW = "0.40" if _IS_REDIMNET2 else "0.60"
_DEFAULT_CONFIRM_WINDOWS = "4" if _IS_REDIMNET2 else "3"

T_HIGH = float(os.environ.get("SAFEEAR_T_HIGH", _DEFAULT_T_HIGH))
T_LOW = float(os.environ.get("SAFEEAR_T_LOW", _DEFAULT_T_LOW))
STRANGER_CONFIRM_COUNT = int(os.environ.get("SAFEEAR_CONFIRM_WINDOWS", _DEFAULT_CONFIRM_WINDOWS))
DEBOUNCE_SEC = int(os.environ.get("SAFEEAR_DEBOUNCE_SEC", "60"))

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SESSIONS: Dict[str, "SessionState"] = {}


def _as_uuid(value: Any, field_name: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid_{field_name}") from exc


def _ensure_parent_dirs(parent_id: str) -> Dict[str, str]:
    base = os.path.join(TENANTS_ROOT, str(parent_id))
    embeddings = os.path.join(base, "embeddings")
    alerts = os.path.join(base, "alerts")
    os.makedirs(embeddings, exist_ok=True)
    os.makedirs(alerts, exist_ok=True)
    return {"base": base, "embeddings": embeddings, "alerts": alerts}


def now_ms() -> int:
    return int(time.time() * 1000)


def _b64(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64d(data: str) -> bytes:
    import base64
    pad = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + pad)


def make_jwt(parent_id: str) -> str:
    import hmac
    import hashlib
    import json

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": str(parent_id), "iat": int(time.time()), "exp": int(time.time()) + JWT_EXPIRE_SEC}
    h = _b64(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(JWT_SECRET.encode("utf-8"), f"{h}.{p}".encode("ascii"), hashlib.sha256).digest()
    s = _b64(sig)
    return f"{h}.{p}.{s}"


def parse_jwt(token: str) -> Dict[str, Any]:
    import hmac
    import hashlib
    import json

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
        parent.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(parent)
    _ensure_parent_dirs(parent.id)
    return parent


def get_parent(db: Session, parent_id: str) -> Parent:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    parent = db.query(Parent).filter(Parent.id == parent_uuid).first()
    if not parent:
        raise HTTPException(status_code=401, detail="parent_not_found")
    return parent


def create_device(db: Session, parent_id: str, device_name: str, role: DeviceRole, device_token: Optional[str] = None) -> Device:
    parent_uuid = _as_uuid(parent_id, "parent_id")

    token = (device_token or "").strip() or None
    if token is not None:
        existing = db.query(Device).filter(Device.device_token == token).first()
        if existing is not None:
            # Reuse the existing row for this installation token and refresh its binding.
            existing.parent_id = parent_uuid
            existing.device_name = device_name
            existing.role = role
            existing.device_token = token
            existing.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            return existing

    device = Device(
        id=uuid.uuid4(),
        parent_id=parent_uuid,
        device_name=device_name,
        role=role,
        device_token=token
    )
    db.add(device)
    db.commit()
    db.refresh(device)
    return device


def get_device(db: Session, parent_id: str, device_id: str) -> Device:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    device_uuid = _as_uuid(device_id, "device_id")
    device = db.query(Device).filter(Device.id == device_uuid, Device.parent_id == parent_uuid).first()
    if not device:
        raise HTTPException(status_code=404, detail="device_not_found")
    return device


def update_device_location(db: Session, parent_id: str, device_id: str, lat: Optional[float], lon: Optional[float]) -> Device:
    device = get_device(db, parent_id, device_id)
    if lat is not None:
        device.last_location_lat = lat
    if lon is not None:
        device.last_location_lon = lon
    device.last_location_ts = datetime.utcnow()
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
        primary = matches[0]
        primary_dir = _speaker_dir(parent_id, primary.id)
        os.makedirs(primary_dir, exist_ok=True)

        # Merge duplicate rows/files into the primary row so UI shows a single entry.
        for dup in matches[1:]:
            dup_dir = _speaker_dir(parent_id, dup.id)
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

            primary.sample_count = int(primary.sample_count or 0) + int(max(moved, int(dup.sample_count or 0)))
            db.delete(dup)

        primary.display_name = normalized_name
        primary.updated_at = datetime.utcnow()
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
    os.makedirs(_speaker_dir(parent_id, speaker.id), exist_ok=True)
    return speaker


def _speaker_dir(parent_id: str, speaker_id: str) -> str:
    dirs = _ensure_parent_dirs(parent_id)
    return os.path.join(dirs["embeddings"], str(speaker_id))


def list_speakers(db: Session, parent_id: str) -> List[EnrolledSpeaker]:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    return (
        db.query(EnrolledSpeaker)
        .filter(
            EnrolledSpeaker.parent_id == parent_uuid,
            EnrolledSpeaker.sample_count > 0,
        )
        .order_by(EnrolledSpeaker.created_at.desc())
        .all()
    )


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
    speaker.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(speaker)
    return speaker


def delete_speaker(db: Session, parent_id: str, speaker_id: str) -> None:
    import shutil
    speaker = get_speaker(db, parent_id, speaker_id)
    db.delete(speaker)
    db.commit()
    sdir = _speaker_dir(parent_id, speaker_id)
    if os.path.exists(sdir):
        shutil.rmtree(sdir)


def save_speaker_embedding(db: Session, parent_id: str, speaker_id: str, emb: np.ndarray) -> str:
    sdir = _speaker_dir(parent_id, speaker_id)
    os.makedirs(sdir, exist_ok=True)
    files = [f for f in os.listdir(sdir) if f.endswith(".npy")]
    idx = len(files) + 1
    fpath = os.path.join(sdir, f"emb_{idx}.npy")
    np.save(fpath, np.asarray(emb, dtype=np.float32).reshape(-1))

    speaker = get_speaker(db, parent_id, speaker_id)
    speaker.sample_count = idx
    speaker.updated_at = datetime.utcnow()
    db.commit()
    return fpath


def load_parent_embeddings(db: Session, parent_id: str) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {}
    for sp in list_speakers(db, parent_id):
        sdir = _speaker_dir(parent_id, sp.id)
        if not os.path.isdir(sdir):
            continue
        embs: List[np.ndarray] = []
        for fname in sorted(os.listdir(sdir)):
            if not fname.endswith(".npy"):
                continue
            try:
                arr = np.asarray(np.load(os.path.join(sdir, fname)), dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if arr.ndim == 1 and arr.shape[0] > 0:
                embs.append(arr)
        if embs:
            out[str(sp.id)] = embs
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def score_against_parent(db: Session, parent_id: str, query_embs: List[np.ndarray]) -> Tuple[float, Optional[str]]:
    candidates = load_parent_embeddings(db, parent_id)
    best = -1.0
    best_speaker: Optional[str] = None
    speaker_scores = {}
    
    for sid, embs in candidates.items():
        sims: List[float] = []
        for q in query_embs:
            for ref in embs:
                sims.append(_cos(q, ref))
        if not sims:
            continue
        # Robust against one-off outliers: use mean of top-k similarities instead of global max.
        top_k = sorted(sims, reverse=True)[: min(5, len(sims))]
        speaker_score = float(np.mean(top_k))
        speaker_scores[sid] = speaker_score
        if speaker_score > best:
            best = speaker_score
            best_speaker = sid
    
    if speaker_scores:
        logger.debug(f"SPEAKER_SCORING | parent_id={parent_id} | scores={{{', '.join(f'{name}:{score:.4f}' for name, score in speaker_scores.items())}}}")
    
    if best < 0:
        logger.info(f"NO_SPEAKER_MATCH | parent_id={parent_id} | score=0.0")
        return 0.0, None
    
    logger.info(f"BEST_SPEAKER_MATCHED | parent_id={parent_id} | speaker_id={best_speaker} | score={best:.4f}")
    return float(best), best_speaker


def create_alert(
    db: Session,
    parent_id: str,
    device_id: str,
    audio_clip_path: str,
    confidence_score: float,
    latitude: Optional[float],
    longitude: Optional[float],
) -> AlertModel:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    device_uuid = _as_uuid(device_id, "device_id")
    alert = AlertModel(
        id=uuid.uuid4(),
        parent_id=parent_uuid,
        device_id=device_uuid,
        timestamp=datetime.utcnow(),
        confidence_score=float(confidence_score),
        audio_clip_path=audio_clip_path,
        latitude=latitude,
        longitude=longitude,
        acknowledged_at=None,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    logger.info(f"ALERT_CREATED | alert_id={alert.id} | parent_id={parent_id} | device_id={device_id} | confidence_score={confidence_score:.4f} | location=({latitude}, {longitude})")
    return alert


def list_alerts(db: Session, parent_id: str, limit: int = 50, offset: int = 0) -> List[AlertModel]:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    return db.query(AlertModel).filter(
        AlertModel.parent_id == parent_uuid
    ).order_by(AlertModel.timestamp.desc()).offset(offset).limit(limit).all()


def ack_alert(db: Session, parent_id: str, alert_id: str) -> AlertModel:
    parent_uuid = _as_uuid(parent_id, "parent_id")
    alert_uuid = _as_uuid(alert_id, "alert_id")
    alert = db.query(AlertModel).filter(
        AlertModel.id == alert_uuid,
        AlertModel.parent_id == parent_uuid
    ).first()
    if not alert:
        raise HTTPException(status_code=404, detail="alert_not_found")
    alert.acknowledged_at = datetime.utcnow()
    alert.updated_at = datetime.utcnow()
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


@dataclass
class SessionState:
    parent_id: str
    device_id: str
    sr: int = 16000
    ring: Optional[np.ndarray] = None
    stranger_streak: int = 0
    last_alert_ms: int = 0
    lat: Optional[float] = None
    lon: Optional[float] = None
    last_familiar_ms: int = 0
    last_familiar_speaker_id: Optional[str] = None


def _session_key(parent_id: str, device_id: str) -> str:
    return f"{parent_id}:{device_id}"


def get_or_create_session(parent_id: str, device_id: str) -> SessionState:
    key = _session_key(parent_id, device_id)
    with _LOCK:
        row = _SESSIONS.get(key)
        if row is None:
            row = SessionState(parent_id=parent_id, device_id=device_id)
            _SESSIONS[key] = row
        return row


def stop_session(parent_id: str, device_id: str) -> None:
    key = _session_key(parent_id, device_id)
    with _LOCK:
        _SESSIONS.pop(key, None)


def update_session_location(parent_id: str, device_id: str, lat: Optional[float], lon: Optional[float]) -> None:
    s = get_or_create_session(parent_id, device_id)
    s.lat = lat
    s.lon = lon


def evaluate_window(db: Session, parent_id: str, waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(waveform**2) + 1e-12))
    if rms < 0.003:
        logger.info(f"TIER1_VAD_REJECTED | parent_id={parent_id} | reason=low_rms | rms={rms:.6f}")
        return {
            "tier1_vad": {"passed": False, "voiced_ms": 0.0, "rms": rms},
            "tier2": {"passed": False, "confidence": 0.0, "reason": "low_rms"},
            "tier3": {"passed": False, "score": 0.0, "closest_speaker_id": None},
            "decision": "rejected_silence",
        }

    try:
        segments = get_speech_segments(waveform, sr=sr, use_silero=True)
    except Exception:
        segments = get_speech_segments(waveform, sr=sr, use_silero=False)

    voiced_samples = sum(max(0, e - s) for s, e in segments)
    voiced_ms = round((voiced_samples * 1000.0) / sr, 1)
    if voiced_samples <= int(0.5 * sr):
        logger.info(f"TIER1_VAD_REJECTED | parent_id={parent_id} | reason=insufficient_voice | voiced_ms={voiced_ms} | threshold_ms=500")
        return {
            "tier1_vad": {"passed": False, "voiced_ms": voiced_ms},
            "tier2": {"passed": False, "confidence": 0.0},
            "tier3": {"passed": False, "score": 0.0, "closest_speaker_id": None},
            "decision": "rejected",
        }

    logger.info(f"TIER1_VAD_PASSED | parent_id={parent_id} | voiced_ms={voiced_ms} | rms={rms:.6f}")

    longest = max(segments, key=lambda ab: ab[1] - ab[0])
    gate_audio = waveform[longest[0] : longest[1]]
    speech_ok, speech_metrics = assess_speech_likeness(gate_audio, sr)
    
    yamnet_result = classify_audio_event(gate_audio, sr)
    category = yamnet_result.get("category", "uncertain")
    confidence = float(yamnet_result.get("confidence", 0.0))

    logger.info(f"TIER2_EVALUATION | parent_id={parent_id} | speech_ok={speech_ok} | yamnet_category={category} | yamnet_confidence={confidence:.4f} | rms={speech_metrics.get('rms', 0.0):.6f} | flatness={speech_metrics.get('spectral_flatness', 0.0):.4f} | centroid_hz={speech_metrics.get('spectral_centroid_hz', 0.0):.1f}")

    # Only hard-reject clear non-human audio. If YAMNet is uncertain,
    # continue to tier-3 so speaker scoring can still run.
    hard_reject = (
        not speech_ok
        or speech_metrics.get("rms", 0.0) < 0.003
        or speech_metrics.get("spectral_flatness", 1.0) > 0.85
        or (
            speech_metrics.get("spectral_centroid_hz", 0.0) < 120.0
            or speech_metrics.get("spectral_centroid_hz", 0.0) > 7800.0
        )
        or
        category == "reject"
        or (category == "vocal_non_speech" and confidence >= 0.70)
    )

    if hard_reject:
        logger.info(f"TIER2_REJECTED | parent_id={parent_id} | speech_ok={speech_ok} | yamnet_category={category} | yamnet_confidence={confidence:.4f}")
        return {
            "tier1_vad": {"passed": True, "voiced_ms": voiced_ms},
            "tier2": {"passed": False, "confidence": confidence, **yamnet_result, **speech_metrics},
            "tier3": {"passed": False, "score": 0.0, "closest_speaker_id": None},
            "decision": "discard_non_human_or_low_confidence",
        }

    logger.info(f"TIER2_PASSED | parent_id={parent_id} | proceeding_to_tier3_speaker_match")

    query_embs = embed_waveform_chunks(waveform, sr=sr, segments=segments)
    score, closest_speaker_id = score_against_parent(db, parent_id, query_embs)
    
    logger.info(f"TIER3_SCORED | parent_id={parent_id} | score={score:.4f} | closest_speaker_id={closest_speaker_id} | t_high={T_HIGH} | t_low={T_LOW}")
    
    return {
        "tier1_vad": {"passed": True, "voiced_ms": voiced_ms},
        "tier2": {"passed": True, "confidence": confidence, **yamnet_result, **speech_metrics},
        "tier3": {
            "passed": True,
            "score": float(score),
            "closest_speaker_id": closest_speaker_id,
            "t_high": T_HIGH,
            "t_low": T_LOW,
        },
        "decision": "tier3_scored",
    }


def append_frame(session: SessionState, frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32).reshape(-1)
    if session.ring is None:
        session.ring = frame
    else:
        session.ring = np.concatenate([session.ring, frame])
    keep = int(max(6.0, WINDOW_SEC) * session.sr)
    if session.ring.shape[0] > keep:
        session.ring = session.ring[-keep:]
    return session.ring


def should_hop(frame_samples: int, sr: int = 16000) -> bool:
    return frame_samples >= int(HOP_SEC * sr)


def save_alert_clip(parent_id: str, session: SessionState, waveform: np.ndarray) -> str:
    dirs = _ensure_parent_dirs(parent_id)
    alert_id = str(uuid.uuid4())
    adir = os.path.join(dirs["alerts"], alert_id)
    os.makedirs(adir, exist_ok=True)
    clip_path = os.path.join(adir, "clip.wav")
    sf.write(clip_path, waveform, session.sr)
    return clip_path
