import pytest
from app.services import session_manager


def test_save_and_load_buffer_no_redis(monkeypatch):
    # If redis is unavailable the functions should not raise
    monkeypatch.setattr("app.services.session_redis.get_redis_client", lambda: None)
    session_manager.save_buffer("p", "d", "clip_buffer", [0.0, 0.1, 0.2])
    val = session_manager.load_buffer("p", "d", "clip_buffer")
    assert val is None
