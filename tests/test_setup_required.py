"""Before any family voice is enrolled, nothing may be reported as a stranger."""

import uuid

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.database.session import SessionLocal, init_db
from app.main import app
from app.utils import prd_services_db as svc


@pytest.fixture
async def client():
    init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_window_with_speech_but_empty_gallery_is_held_not_scored(monkeypatch):
    monkeypatch.setattr(svc, "get_household_thresholds", lambda db, pid: (svc.T_HIGH, svc.T_LOW))
    monkeypatch.setattr(svc, "get_speech_segments", lambda w, sr=16000, use_silero=True: [(0, len(w))])
    monkeypatch.setattr(svc, "assess_speech_likeness", lambda w, sr: (True, {}))
    monkeypatch.setattr(svc, "classify_audio_event_windowed",
                        lambda w, sr=16000: {"category": "human_speech", "confidence": 0.95})
    monkeypatch.setattr(svc, "load_parent_embeddings", lambda db, pid: {})

    def no_embedding(*_a, **_k):
        raise AssertionError("no embedding should be computed without a gallery")

    monkeypatch.setattr(svc, "embed_waveform_chunks", no_embedding)
    speech = (0.1 * np.random.default_rng(0).standard_normal(24000)).astype(np.float32)
    stage = svc.evaluate_window(None, str(uuid.uuid4()), speech)
    assert stage["decision"] == "setup_required"
    assert stage["tier3"]["passed"] is False


async def _parent_and_child(client):
    reg = (await client.post(
        "/auth/register-email",
        json={"email": f"su_{uuid.uuid4().hex}@example.com", "password": "password123", "display_name": "P"},
    )).json()
    token = reg["access_token"]
    resp = await client.post(
        "/devices", headers={"Authorization": f"Bearer {token}"},
        data={"device_name": "kid", "role": "child_device", "device_token": f"fcm_{uuid.uuid4().hex}",
              "installation_id": uuid.uuid4().hex},
    )
    return reg, resp.json()["device"]


@pytest.mark.anyio
async def test_monitoring_cannot_be_turned_on_until_a_voice_is_enrolled(client):
    reg, child = await _parent_and_child(client)
    auth = {"Authorization": f"Bearer {reg['access_token']}"}
    url = f"/devices/{child['id']}/monitoring"

    resp = await client.patch(url, headers=auth, json={"monitoring_enabled": True})
    assert resp.status_code == 409
    # Turning it off is always allowed.
    assert (await client.patch(url, headers=auth, json={"monitoring_enabled": False})).status_code == 200

    db = SessionLocal()
    try:
        pid = reg["parent"]["id"]
        speaker = svc.create_speaker(db, pid, "Mom")
        # A speaker row alone is not enough: it needs a stored voiceprint.
        assert (await client.patch(url, headers=auth, json={"monitoring_enabled": True})).status_code == 409
        svc.save_speaker_embedding(db, pid, str(speaker.id), np.ones(192, dtype=np.float32))
    finally:
        db.close()
    resp = await client.patch(url, headers=auth, json={"monitoring_enabled": True})
    assert resp.status_code == 200 and resp.json()["monitoring_enabled"] is True
