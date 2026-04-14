import json
import logging
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import HTTPException

from .feature_extractor import embed_waveform_chunks
from .speech_gate import assess_speech_likeness
from .vad import get_speech_segments

STATE_PATH = os.path.join("app", "database", "app_state.json")
TENANTS_ROOT = os.path.join("app", "data", "tenants")

JWT_SECRET = os.environ.get("SAFEEAR_JWT_SECRET", "change-me-in-production")
JWT_EXPIRE_SEC = int(os.environ.get("SAFEEAR_JWT_EXPIRE_SEC", "604800"))

WINDOW_SEC = float(os.environ.get("SAFEEAR_WINDOW_SEC", "1.5"))
HOP_SEC = float(os.environ.get("SAFEEAR_HOP_SEC", "0.25"))
ALERT_RING_SEC = float(os.environ.get("SAFEEAR_ALERT_RING_SEC", "10.0"))
STRANGER_PREROLL_SEC = float(os.environ.get("SAFEEAR_STRANGER_PREROLL_SEC", "5.0"))
_REDIMNET_HUB_REPO = os.environ.get("REDIMNET_HUB_REPO", "PalabraAI/redimnet2")
_REDIMNET_HUB_ENTRY = os.environ.get("REDIMNET_HUB_ENTRY", "redimnet2")
_IS_REDIMNET2 = ("redimnet2" in _REDIMNET_HUB_REPO.lower()) or ("redimnet2" in _REDIMNET_HUB_ENTRY.lower())

_DEFAULT_T_HIGH = "0.40" if _IS_REDIMNET2 else "0.72"
_DEFAULT_T_LOW = "0.24" if _IS_REDIMNET2 else "0.60"
_DEFAULT_CONFIRM_WINDOWS = "3"

T_HIGH = float(os.environ.get("SAFEEAR_T_HIGH", _DEFAULT_T_HIGH))
T_LOW = float(os.environ.get("SAFEEAR_T_LOW", _DEFAULT_T_LOW))
STRANGER_CONFIRM_COUNT = int(os.environ.get("SAFEEAR_CONFIRM_WINDOWS", _DEFAULT_CONFIRM_WINDOWS))
DEBOUNCE_SEC = int(os.environ.get("SAFEEAR_DEBOUNCE_SEC", "60"))

_LOCK = threading.Lock()
_SESSIONS: Dict[str, "SessionState"] = {}
logger = logging.getLogger(__name__)


def _ensure_parent_dirs(parent_id: str) -> Dict[str, str]:
    base = os.path.join(TENANTS_ROOT, parent_id)
    embeddings = os.path.join(base, "embeddings")
    alerts = os.path.join(base, "alerts")
    os.makedirs(embeddings, exist_ok=True)
    os.makedirs(alerts, exist_ok=True)
    return {"base": base, "embeddings": embeddings, "alerts": alerts}


def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        initial = {"parents": {}, "devices": {}, "speakers": {}, "alerts": {}, "refresh_tokens": {}}
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2)
        return initial
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


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

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": parent_id,
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
    import hmac
    import hashlib

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


def upsert_parent(google_sub: str, email: Optional[str], display_name: Optional[str]) -> Dict[str, Any]:
    with _LOCK:
        state = _load_state()
        row = state["parents"].get(google_sub)
        if row is None:
            row = {
                "id": google_sub,
                "google_sub": google_sub,
                "email": email,
                "display_name": display_name,
                "created_at": now_ms(),
                "updated_at": now_ms(),
                "phone_number": None,
                "fcm_token": None,
            }
            state["parents"][google_sub] = row
        else:
            row["email"] = email or row.get("email")
            row["display_name"] = display_name or row.get("display_name")
            row["updated_at"] = now_ms()
        _save_state(state)
        _ensure_parent_dirs(google_sub)
        return row


def save_refresh_token(parent_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with _LOCK:
        state = _load_state()
        state["refresh_tokens"][token] = {
            "parent_id": parent_id,
            "created_at": now_ms(),
            "expires_at": now_ms() + 30 * 24 * 3600 * 1000,
        }
        _save_state(state)
    return token


def consume_refresh_token(refresh_token: str) -> str:
    with _LOCK:
        state = _load_state()
        row = state["refresh_tokens"].get(refresh_token)
        if not row:
            raise HTTPException(status_code=401, detail="invalid_refresh_token")
        if int(row.get("expires_at", 0)) < now_ms():
            del state["refresh_tokens"][refresh_token]
            _save_state(state)
            raise HTTPException(status_code=401, detail="refresh_token_expired")
        parent_id = row["parent_id"]
        del state["refresh_tokens"][refresh_token]
        _save_state(state)
        return parent_id


def revoke_refresh_token(refresh_token: str) -> None:
    with _LOCK:
        state = _load_state()
        if refresh_token in state["refresh_tokens"]:
            del state["refresh_tokens"][refresh_token]
            _save_state(state)


def _speaker_dir(parent_id: str, speaker_id: str) -> str:
    dirs = _ensure_parent_dirs(parent_id)
    return os.path.join(dirs["embeddings"], speaker_id)


def create_speaker(parent_id: str, display_name: str) -> Dict[str, Any]:
    normalized_name = (display_name or "").strip()
    if not normalized_name:
        raise HTTPException(status_code=422, detail="display_name_required")

    with _LOCK:
        state = _load_state()
        existing = [
            r for r in state["speakers"].values()
            if r.get("parent_id") == parent_id and str(r.get("display_name", "")).strip().lower() == normalized_name.lower()
        ]
        if existing:
            existing.sort(key=lambda r: (int(r.get("sample_count", 0)), int(r.get("updated_at", 0))), reverse=True)
            return existing[0]

        speaker_id = str(uuid.uuid4())
        row = {
            "id": speaker_id,
            "parent_id": parent_id,
            "display_name": normalized_name,
            "sample_count": 0,
            "created_at": now_ms(),
            "updated_at": now_ms(),
        }
        state["speakers"][speaker_id] = row
        _save_state(state)
    os.makedirs(_speaker_dir(parent_id, speaker_id), exist_ok=True)
    return row


def list_speakers(parent_id: str) -> List[Dict[str, Any]]:
    state = _load_state()
    rows = [r for r in state["speakers"].values() if r.get("parent_id") == parent_id]
    rows.sort(key=lambda r: int(r.get("created_at", 0)), reverse=True)
    return rows


def get_speaker(parent_id: str, speaker_id: str) -> Dict[str, Any]:
    state = _load_state()
    row = state["speakers"].get(speaker_id)
    if not row or row.get("parent_id") != parent_id:
        raise HTTPException(status_code=404, detail="speaker_not_found")
    return row


def rename_speaker(parent_id: str, speaker_id: str, new_name: str) -> Dict[str, Any]:
    with _LOCK:
        state = _load_state()
        row = state["speakers"].get(speaker_id)
        if not row or row.get("parent_id") != parent_id:
            raise HTTPException(status_code=404, detail="speaker_not_found")
        row["display_name"] = new_name
        row["updated_at"] = now_ms()
        _save_state(state)
        return row


def delete_speaker(parent_id: str, speaker_id: str) -> None:
    import shutil

    with _LOCK:
        state = _load_state()
        row = state["speakers"].get(speaker_id)
        if not row or row.get("parent_id") != parent_id:
            raise HTTPException(status_code=404, detail="speaker_not_found")
        del state["speakers"][speaker_id]
        _save_state(state)
    sdir = _speaker_dir(parent_id, speaker_id)
    if os.path.exists(sdir):
        shutil.rmtree(sdir)


def save_speaker_embedding(parent_id: str, speaker_id: str, emb: np.ndarray) -> str:
    sdir = _speaker_dir(parent_id, speaker_id)
    os.makedirs(sdir, exist_ok=True)
    files = [f for f in os.listdir(sdir) if f.endswith(".npy")]
    idx = len(files) + 1
    fpath = os.path.join(sdir, f"emb_{idx}.npy")
    np.save(fpath, np.asarray(emb, dtype=np.float32).reshape(-1))
    with _LOCK:
        state = _load_state()
        row = state["speakers"].get(speaker_id)
        if row:
            row["sample_count"] = int(row.get("sample_count", 0)) + 1
            row["updated_at"] = now_ms()
            _save_state(state)
    return fpath


def load_parent_embeddings(parent_id: str) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {}
    for sp in list_speakers(parent_id):
        sid = sp["id"]
        sdir = _speaker_dir(parent_id, sid)
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
            out[sid] = embs
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def score_against_parent(parent_id: str, query_embs: List[np.ndarray]) -> Tuple[float, Optional[str]]:
    candidates = load_parent_embeddings(parent_id)
    best = -1.0
    best_speaker: Optional[str] = None
    for sid, embs in candidates.items():
        for q in query_embs:
            for ref in embs:
                s = _cos(q, ref)
                if s > best:
                    best = s
                    best_speaker = sid
    if best < 0:
        return 0.0, None
    return float(best), best_speaker


def create_alert(
    parent_id: str,
    device_id: str,
    audio_clip_path: str,
    confidence_score: float,
    latitude: Optional[float],
    longitude: Optional[float],
) -> Dict[str, Any]:
    aid = str(uuid.uuid4())
    row = {
        "id": aid,
        "parent_id": parent_id,
        "device_id": device_id,
        "timestamp": now_ms(),
        "confidence_score": float(confidence_score),
        "audio_clip_path": audio_clip_path,
        "latitude": latitude,
        "longitude": longitude,
        "acknowledged_at": None,
        "created_at": now_ms(),
        "updated_at": now_ms(),
    }
    with _LOCK:
        state = _load_state()
        state["alerts"][aid] = row
        _save_state(state)
    return row


def list_alerts(parent_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    state = _load_state()
    rows = [r for r in state["alerts"].values() if r.get("parent_id") == parent_id]
    rows.sort(key=lambda r: int(r.get("timestamp", 0)), reverse=True)
    return rows[offset : offset + limit]


def ack_alert(parent_id: str, alert_id: str) -> Dict[str, Any]:
    with _LOCK:
        state = _load_state()
        row = state["alerts"].get(alert_id)
        if not row or row.get("parent_id") != parent_id:
            raise HTTPException(status_code=404, detail="alert_not_found")
        row["acknowledged_at"] = now_ms()
        row["updated_at"] = now_ms()
        _save_state(state)
        return row


def get_alert(parent_id: str, alert_id: str) -> Dict[str, Any]:
    state = _load_state()
    row = state["alerts"].get(alert_id)
    if not row or row.get("parent_id") != parent_id:
        raise HTTPException(status_code=404, detail="alert_not_found")
    return row


@dataclass
class SessionState:
    parent_id: str
    device_id: str
    sr: int = 16000
    ring: Optional[np.ndarray] = None
    stranger_segment: Optional[np.ndarray] = None
    stranger_segment_started_ms: int = 0
    stranger_segment_ended_ms: int = 0
    stranger_streak: int = 0
    last_alert_ms: int = 0
    lat: Optional[float] = None
    lon: Optional[float] = None


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


def start_stranger_segment(
    session: SessionState,
    waveform: np.ndarray,
    started_ms: int,
    pre_roll_waveform: Optional[np.ndarray] = None,
) -> None:
    segment = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if pre_roll_waveform is not None:
        pre_roll = np.asarray(pre_roll_waveform, dtype=np.float32).reshape(-1)
        if pre_roll.size > 0:
            segment = np.concatenate([pre_roll, segment])
    session.stranger_segment = segment.copy()
    session.stranger_segment_started_ms = int(started_ms)
    session.stranger_segment_ended_ms = 0


def append_stranger_segment(session: SessionState, waveform: np.ndarray) -> None:
    segment = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if session.stranger_segment is None or session.stranger_segment.size == 0:
        session.stranger_segment = segment.copy()
        return
    session.stranger_segment = np.concatenate([session.stranger_segment, segment])


def clear_stranger_segment(session: SessionState) -> None:
    session.stranger_segment = None
    session.stranger_segment_started_ms = 0
    session.stranger_segment_ended_ms = 0


def get_stranger_segment_waveform(session: SessionState, fallback: np.ndarray) -> np.ndarray:
    if session.stranger_segment is not None and session.stranger_segment.size > 0:
        return session.stranger_segment
    return np.asarray(fallback, dtype=np.float32).reshape(-1)


def update_session_location(parent_id: str, device_id: str, lat: Optional[float], lon: Optional[float]) -> None:
    s = get_or_create_session(parent_id, device_id)
    s.lat = lat
    s.lon = lon


def evaluate_window(parent_id: str, waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    try:
        segments = get_speech_segments(waveform, sr=sr, use_silero=True)
    except Exception:
        segments = get_speech_segments(waveform, sr=sr, use_silero=False)

    voiced_samples = sum(max(0, e - s) for s, e in segments)
    if voiced_samples <= int(0.5 * sr):
        return {
            "tier1_vad": {"passed": False, "voiced_ms": round((voiced_samples * 1000.0) / sr, 1)},
            "tier2": {"passed": False, "confidence": 0.0},
            "tier3": {"passed": False, "score": 0.0, "closest_speaker_id": None},
            "decision": "rejected",
        }

    longest = max(segments, key=lambda ab: ab[1] - ab[0])
    gate_audio = waveform[longest[0] : longest[1]]
    speech_ok, speech_metrics = assess_speech_likeness(gate_audio, sr)
    confidence = 0.8 if speech_ok else 0.2

    hard_reject = (not speech_ok) and confidence >= 0.70
    if hard_reject:
        return {
            "tier1_vad": {"passed": True, "voiced_ms": round((voiced_samples * 1000.0) / sr, 1)},
            "tier2": {"passed": False, "confidence": confidence, **speech_metrics},
            "tier3": {"passed": False, "score": 0.0, "closest_speaker_id": None},
            "decision": "discard_non_human_or_low_confidence",
        }

    query_embs = embed_waveform_chunks(waveform, sr=sr, segments=segments)
    score, closest_speaker_id = score_against_parent(parent_id, query_embs)
    return {
        "tier1_vad": {"passed": True, "voiced_ms": round((voiced_samples * 1000.0) / sr, 1)},
        "tier2": {"passed": True, "confidence": confidence, **speech_metrics},
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
    keep = int(max(ALERT_RING_SEC, WINDOW_SEC) * session.sr)
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
    tmp_clip_path = f"{clip_path}.tmp"
    metadata_path = os.path.join(adir, "metadata.json")
    metadata = {
        "segment_started_ms": int(session.stranger_segment_started_ms) if session.stranger_segment_started_ms else None,
        "segment_ended_ms": int(session.stranger_segment_ended_ms) if session.stranger_segment_ended_ms else None,
        "sample_rate": int(session.sr),
        "samples": int(np.asarray(waveform).reshape(-1).shape[0]),
    }
    try:
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
