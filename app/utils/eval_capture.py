"""Evaluation capture: record every scored detection window for offline replay.

Enabled only when SAFEEAR_EVAL_CAPTURE_DIR is set.  Each window becomes one
JSON line in <dir>/<device_id>/<YYYY-MM-DD>.jsonl; scripts/eval/replay.py
re-runs the decision engine over these files.

SAFEEAR_EVAL_CAPTURE_AUDIO=true additionally saves each 1.5 s window as a wav
(about 170 MB per hour of monitoring).  Recordings contain other people's
voices: collect consent and do not commit them to version control.
"""

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

CAPTURE_DIR = (os.environ.get("SAFEEAR_EVAL_CAPTURE_DIR") or "").strip()
CAPTURE_AUDIO = os.environ.get("SAFEEAR_EVAL_CAPTURE_AUDIO", "false").strip().lower() in {"1", "true", "yes", "on"}

_LOCK = threading.Lock()


def is_enabled() -> bool:
    return bool(CAPTURE_DIR)


def capture_window(
    *,
    parent_id: str,
    device_id: str,
    now_ms: int,
    stage: Dict[str, Any],
    decision: str,
    alert_fired: bool,
    stranger_streak: int,
    centroid: Optional[np.ndarray],
    window: Optional[np.ndarray] = None,
    sr: int = 16000,
) -> None:
    """Append one window record.  Never raises: capture must not affect detection."""
    if not CAPTURE_DIR:
        return
    try:
        tier1 = stage.get("tier1_vad", {}) or {}
        tier2 = stage.get("tier2", {}) or {}
        tier3 = stage.get("tier3", {}) or {}
        device_dir = os.path.join(CAPTURE_DIR, str(device_id))
        day = datetime.fromtimestamp(now_ms / 1000.0).strftime("%Y-%m-%d")

        wav_rel = None
        if CAPTURE_AUDIO and window is not None and len(window) > 0:
            import soundfile as sf

            wav_rel = os.path.join("wav", day, f"{now_ms}.wav")
            wav_path = os.path.join(device_dir, wav_rel)
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            sf.write(wav_path, np.asarray(window, dtype=np.float32), sr, format="WAV")

        record = {
            "ts_ms": int(now_ms),
            "parent_id": str(parent_id),
            "device_id": str(device_id),
            "tier1_pass": bool(tier1.get("passed", False)),
            "voiced_ms": tier1.get("voiced_ms"),
            "tier2_pass": bool(tier2.get("passed", False)),
            "tier2_category": str(tier2.get("category", "uncertain")),
            "tier2_confidence": float(tier2.get("confidence", 0.0) or 0.0),
            "tier3_pass": bool(tier3.get("passed", False)),
            "score": float(tier3.get("score", 0.0) or 0.0),
            "closest_speaker_id": tier3.get("closest_speaker_id"),
            "closest_speaker_role": tier3.get("closest_speaker_role"),
            "speaker_scores": tier3.get("speaker_scores") or {},
            "t_high": tier3.get("t_high"),
            "t_low": tier3.get("t_low"),
            "decision": decision,
            "alert_fired": bool(alert_fired),
            "stranger_streak": int(stranger_streak),
            "centroid": [round(float(x), 5) for x in np.asarray(centroid).reshape(-1)] if centroid is not None else None,
            "wav": wav_rel,
        }
        line = json.dumps(record, separators=(",", ":"))
        with _LOCK:
            os.makedirs(device_dir, exist_ok=True)
            with open(os.path.join(device_dir, f"{day}.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:
        logger.warning("EVAL_CAPTURE_FAILED | device_id=%s | reason=%s", device_id, exc)
