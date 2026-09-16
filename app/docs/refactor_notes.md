Refactor Notes - June 1, 2026

Summary:
- Extracted detection buffering into `app/services/detection_service.py`.
- Extracted alert creation/escalation into `app/services/alert_service.py`.
- Extracted decision and alert orchestration into `app/services/decision_engine.py`.
- Added Redis-backed session metadata and buffer persistence helpers in `app/services/session_redis.py` and `app/services/session_manager.py`.
- Made native `soundfile` optional for tests via a local shim and made `librosa` optional at import time.

Key files:
- `app/services/detection_service.py` - buffering and window evaluation.
- `app/services/decision_engine.py` - decision logic previously in `detect_chunk()`.
- `app/services/alert_service.py` - alert persistence, websocket publish, escalation.
- `app/services/session_redis.py` - redis helpers for session metadata and buffers.
- `app/services/session_manager.py` - safer wrapper used by services.
- `app/utils/prd_services_db.py` - SessionState now persists buffers to Redis (best-effort).

Next steps:
- Move more of `prd_services_db` domain logic into service layer.
- Add background worker for alert escalation and retry semantics.
- Run full test suite by installing runtime deps (`pip install -r requirements.txt`) or by adding targeted shims for imports in CI.
