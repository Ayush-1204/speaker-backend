import threading
import logging
from typing import Optional
import numpy as np
from app.utils import prd_services_db as svc
from app.services import websocket_manager as ws_manager
from app.utils.notification_worker import escalate_alert
from app.database.models import Device, DeviceRole
from app.database.session import SessionLocal

logger = logging.getLogger(__name__)


def create_and_escalate_alert(
    db, parent_id: str, device_id: str, session, score: float,
    alert_type: str = "unenrolled_contact",
) -> Optional[str]:
    try:
        session.stranger_segment_ended_ms = svc.now_ms()
        buffer_src = getattr(session, "inference_buffer", None)
        if buffer_src is None:
            buffer_src = getattr(session, "clip_buffer", np.zeros(0, dtype=np.float32))
        clip_waveform = svc.get_clip_buffer_waveform(session, buffer_src)
        clip_path = svc.save_alert_clip(parent_id, session, clip_waveform)
        row = svc.create_alert(
            db, parent_id, device_id, clip_path, score, session.lat, session.lon,
            alert_type=alert_type,
        )
        # Build a minimal alert payload for websocket subscribers
        ts_ms = None
        try:
            if getattr(row, "timestamp", None) is not None:
                ts_ms = int(row.timestamp.timestamp() * 1000)
        except Exception:
            ts_ms = svc.now_ms()

        ws_manager.publish_parent_event(
            parent_id,
            "alert_created",
            {
                "id": str(row.id),
                "parent_id": str(row.parent_id),
                "device_id": str(row.device_id),
                "timestamp_ms": ts_ms,
                "confidence_score": row.confidence_score,
                "audio_clip_path": row.audio_clip_path,
                # Change 7: include alert_type so mobile clients can show distress vs identity UX
                "alert_type": getattr(row, "alert_type", "unenrolled_contact"),
            },
        )

        alert_id = str(row.id)
        session.last_alert_ms = svc.now_ms()
        session.stranger_streak = 0
        if not session.recent_confirmed_stranger_embeddings:
            svc.record_confirmed_stranger_window(session, None)
        svc.set_active_stranger_embedding(session)

        # Escalation background thread — opens its own session (SQLAlchemy sessions
        # are not thread-safe and must not be shared across threads).
        def _run_escalation():
            thread_db = SessionLocal()
            try:
                parent = svc.get_parent(thread_db, parent_id)
                parent_device = (
                    thread_db.query(Device)
                    .filter(
                        Device.parent_id == parent.id,
                        Device.role == DeviceRole.parent_device,
                        Device.device_token.isnot(None),
                    )
                    .order_by(Device.updated_at.desc())
                    .first()
                )
                parent_fcm_token = parent_device.device_token if parent_device else None
                ts = None
                try:
                    ts = int(row.timestamp.timestamp() * 1000) if getattr(row, "timestamp", None) is not None else svc.now_ms()
                except Exception:
                    ts = svc.now_ms()
                escalate_alert(
                    str(parent.email) if parent.email else None,
                    str(parent.phone_number) if parent.phone_number else None,
                    str(parent_fcm_token) if parent_fcm_token else None,
                    alert_id,
                    device_id,
                    ts,
                    session.lat,
                    session.lon,
                    f"/alerts/{alert_id}/clip",
                    score,
                    recipient_role="parent_device",
                )
            except Exception:
                logger.exception("ALERT_ESCALATION_FAILED | alert_id=%s | device_id=%s", alert_id, device_id)
            finally:
                thread_db.close()

        t = threading.Thread(target=_run_escalation, daemon=True)
        t.start()

        svc.clear_stranger_segment(session)
        svc.clear_clip_buffer(session)

        logger.info(
            "ALERT_FIRED_SERVICE | alert_id=%s | score=%.4f | clip_path=%s",
            alert_id,
            score,
            clip_path,
        )
        return alert_id
    except Exception:
        logger.exception("ALERT_SERVICE_FAILED | parent_id=%s | device_id=%s", parent_id, device_id)
        return None
