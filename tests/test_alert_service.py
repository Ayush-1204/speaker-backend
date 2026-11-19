import types
import time
import pytest

from app.services import alert_service


class DummyRow:
    def __init__(self, id):
        self.id = id
        self.parent_id = "p"
        self.device_id = "d"
        self.timestamp = None
        self.confidence_score = 0.5
        self.audio_clip_path = "/tmp/clip.wav"


class DummySession:
    def __init__(self):
        self.stranger_segment = None
        self.stranger_segment_started_ms = 0
        self.stranger_segment_ended_ms = 0
        self.stranger_streak = 0
        self.last_alert_ms = 0
        self.recent_confirmed_stranger_embeddings = []
        self.active_stranger_embedding = None
        self.clip_buffer = None
        self.sr = 16000
        self.parent_id = "p"
        self.device_id = "d"
        self.lat = None
        self.lon = None


def test_create_and_escalate_alert_monkeypatched(monkeypatch):
    db = object()
    parent_id = "p"
    device_id = "d"
    session = DummySession()
    score = 0.12

    # Monkeypatch svc.create_alert to return a DummyRow
    def fake_create_alert(db_arg, parent_id_arg, device_id_arg, audio_clip_path, confidence_score, latitude, longitude, alert_type="identity"):
        return DummyRow(123)

    monkeypatch.setattr("app.services.alert_service.svc.create_alert", fake_create_alert)

    # Monkeypatch svc.get_parent to return a simple object with email/phone
    class Parent:
        email = "x@example.com"
        phone_number = "+1000000000"

    monkeypatch.setattr("app.services.alert_service.svc.get_parent", lambda db, pid: Parent())

    # Monkeypatch escalate_alert to be a no-op
    monkeypatch.setattr("app.services.alert_service.escalate_alert", lambda *a, **k: True)

    session.stranger_streak = 3
    session.stranger_weight_sum = 2.7
    alert_id = alert_service.create_and_escalate_alert(db, parent_id, device_id, session, score)
    assert alert_id == "123"
    # An identity alert must clear both counters, or the next window re-alerts after the debounce.
    assert session.stranger_streak == 0
    assert session.stranger_weight_sum == 0.0
    assert session.last_alert_ms > 0
