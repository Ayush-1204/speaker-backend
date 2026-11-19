"""Phase 2 regressions: VAD actually runs, YAMNet labels map correctly,
chunk stream ordering, and silent-device detection."""

import io
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from app.database.models import Device
from app.database.session import SessionLocal, init_db
from app.main import app
from app.services import health_monitor
from app.utils import prd_services_db as svc
from app.utils import vad
from app.utils import yamnet_classifier as yc


# --------------------------------------------------------------------------- #
# VAD
# --------------------------------------------------------------------------- #

def test_silero_vad_is_live_and_rejects_noise_and_hum():
    assert vad.status()["backend"] == "silero"
    rng = np.random.default_rng(0)
    t = np.arange(24000) / 16000
    assert vad.get_speech_segments(rng.normal(0, 0.05, 24000).astype(np.float32)) == []
    assert vad.get_speech_segments((0.1 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)) == []


def test_loading_silero_does_not_leave_torch_single_threaded():
    import torch

    vad._get_silero()
    vad.restore_torch_threads()
    assert torch.get_num_threads() == vad.configured_torch_threads()


def test_energy_backend_switch_for_ab_comparison(monkeypatch):
    monkeypatch.setenv("SAFEEAR_VAD_BACKEND", "energy")
    noise = np.random.default_rng(1).normal(0, 0.05, 24000).astype(np.float32)
    assert vad.get_speech_segments(noise) != []  # the old fallback calls noise "speech"


# --------------------------------------------------------------------------- #
# YAMNet category mapping (pure, no TensorFlow needed)
# --------------------------------------------------------------------------- #

def _scores(**named):
    labels = yc.load_labels()
    s = np.zeros(len(labels), dtype=np.float32)
    for name, value in named.items():
        s[labels.index(name.replace("_", " "))] = value
    return s


def test_yell_is_human_speech_not_rejected():
    r = yc.decide_from_scores(_scores(Yell=0.9))
    assert r["category"] == "human_speech"
    assert r["distress_score"] == pytest.approx(0.9)
    assert r["distress_class"] == "Yell"


def test_music_and_tv_are_rejected():
    assert yc.decide_from_scores(_scores(Music=0.9))["category"] == "reject"
    assert yc.decide_from_scores(_scores(Television=0.8))["category"] == "reject"
    labels = yc.load_labels()
    s = np.zeros(len(labels), dtype=np.float32)
    s[labels.index("Electric guitar")] = 0.9  # inside the Music subtree
    assert yc.decide_from_scores(s)["category"] == "reject"


def test_uncertain_confidence_is_speech_evidence_not_top_class():
    r = yc.decide_from_scores(_scores(Speech=0.3, Dog=0.95))
    assert r["category"] == "uncertain"
    assert r["confidence"] == pytest.approx(0.3)


def test_crowd_babble_is_not_stranger_evidence():
    r = yc.decide_from_scores(_scores(Speech=0.65, **{"Hubbub, speech noise, speech babble": 0.9}))
    assert r["category"] == "crowd"


def test_clear_speech_is_human_speech():
    r = yc.decide_from_scores(_scores(Speech=0.95))
    assert r["category"] == "human_speech"
    assert r["confidence"] == pytest.approx(0.95)


# --------------------------------------------------------------------------- #
# Chunk stream ordering
# --------------------------------------------------------------------------- #

@pytest.fixture
async def client():
    init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _wav(seconds=1.0):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


async def _child(client):
    reg = await client.post(
        "/auth/register-email",
        json={"email": f"o_{uuid.uuid4().hex}@example.com", "password": "password123", "display_name": "P"},
    )
    parent = reg.json()
    dev = await client.post(
        "/devices",
        headers={"Authorization": f"Bearer {parent['access_token']}"},
        data={"device_name": "kid", "role": "child_device", "installation_id": uuid.uuid4().hex},
    )
    return parent["parent"]["id"], dev.json()["device"]


async def _send(client, device, seq, t_start, sent_at=None, stream_id=None, silent_skipped=None):
    data = {"device_id": device["id"], "seq": str(seq), "t_start_ms": str(t_start)}
    if silent_skipped is not None:
        data["silent_skipped"] = str(silent_skipped)
    if stream_id is not None:
        data["stream_id"] = stream_id
    if sent_at is not None:
        data["sent_at_ms"] = str(sent_at)
    resp = await client.post(
        "/detect/chunk",
        headers={"Authorization": f"Bearer {device['device_access_token']}"},
        data=data,
        files={"file": ("c.wav", _wav(), "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.anyio
async def test_stream_gap_resets_streak_and_old_chunks_are_dropped(client):
    parent_id, device = await _child(client)
    t0 = 1_700_000_000_000
    await _send(client, device, 1, t0)
    await _send(client, device, 2, t0 + 1000)

    session = svc.get_or_create_session(parent_id, device["id"])
    session.stranger_streak = 2
    session.stranger_weight_sum = 1.5

    # seq jumps 2 -> 5: audio was lost, so the streak must not continue across it.
    await _send(client, device, 5, t0 + 4000)
    assert session.stranger_streak == 0 and session.stranger_weight_sum == 0.0

    # A late chunk from before the gap is dropped, not analysed out of order.
    late = await _send(client, device, 3, t0 + 2000)
    assert late["status"] == "out_of_order_dropped"

    # Audio recorded long before it was sent is stale (offline replay).
    stale = await _send(client, device, 6, t0 + 5000, sent_at=t0 + 5000 + 60_000)
    assert stale["status"] == "stale_dropped"


@pytest.mark.anyio
async def test_new_stream_id_restarts_sequence(client):
    _, device = await _child(client)
    t0 = 1_700_000_000_000
    await _send(client, device, 1, t0, stream_id="a")
    await _send(client, device, 2, t0 + 1000, stream_id="a")
    # App restarted: its counter starts over under a new stream id.
    resp = await _send(client, device, 1, t0 + 60_000, stream_id="b")
    assert resp["status"] != "out_of_order_dropped"
    resp = await _send(client, device, 1, t0 + 61_000, stream_id="b")
    assert resp["status"] == "out_of_order_dropped"


@pytest.mark.anyio
async def test_skipped_silence_is_not_a_gap(client):
    parent_id, device = await _child(client)
    t0 = 1_700_000_000_000
    await _send(client, device, 1, t0)
    session = svc.get_or_create_session(parent_id, device["id"])
    session.stranger_streak = 2
    # Four 1 s chunks were silent and not uploaded; the next chunk says so.
    await _send(client, device, 2, t0 + 5000, silent_skipped=4)
    assert session.stranger_streak == 2
    # Without that information the same 4 s jump is lost audio: reset.
    await _send(client, device, 3, t0 + 10_000)
    assert session.stranger_streak == 0


# --------------------------------------------------------------------------- #
# Silent-device watchdog
# --------------------------------------------------------------------------- #

@pytest.mark.anyio
async def test_silent_monitored_device_is_reported_once_per_episode(client):
    health_monitor.reset_state()
    _, device = await _child(client)
    db = SessionLocal()
    try:
        row = db.query(Device).filter(Device.id == uuid.UUID(device["id"])).first()
        row.last_activity_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)
        db.commit()

        first = {str(d.id) for d in health_monitor.find_newly_silent_devices(db)}
        assert device["id"] in first
        assert device["id"] not in {str(d.id) for d in health_monitor.find_newly_silent_devices(db)}

        # Activity resumes, then stops again: a new episode is reported again.
        row.last_activity_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        assert device["id"] not in {str(d.id) for d in health_monitor.find_newly_silent_devices(db)}
        row.last_activity_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=7)
        db.commit()
        assert device["id"] in {str(d.id) for d in health_monitor.find_newly_silent_devices(db)}
    finally:
        db.close()
