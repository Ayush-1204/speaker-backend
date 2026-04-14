import io
import os
import uuid
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.main import _resolve_redis_url
from app.database.models import Alert, Device
from app.database.session import SessionLocal, init_db
from app.utils import prd_services_db as svc


@pytest.fixture
async def client():
    init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _wav_bytes(sr: int = 16000, duration_sec: float = 1.0) -> bytes:
    samples = int(sr * duration_sec)
    waveform = np.zeros(samples, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sr, format="WAV")
    return buffer.getvalue()


async def _register_parent(client: AsyncClient):
    email = f"u_{uuid.uuid4().hex}@example.com"
    resp = await client.post(
        "/auth/register-email",
        json={"email": email, "password": "password123", "display_name": "Test Parent"},
    )
    assert resp.status_code == 200
    data = resp.json()
    return data["access_token"], data["parent"]["id"]


async def _create_device(
    client: AsyncClient,
    token: str,
    name: str,
    role: str,
    device_token: str,
    installation_id: str | None = None,
):
    payload = {"device_name": name, "role": role, "device_token": device_token}
    if installation_id is not None:
        payload["installation_id"] = installation_id
    resp = await client.post(
        "/devices",
        headers={"Authorization": f"Bearer {token}"},
        data=payload,
    )
    assert resp.status_code == 200
    return resp.json()["device"]["id"]


@pytest.mark.anyio
async def test_child_never_receives_stranger_push(client: AsyncClient):
    token, _ = await _register_parent(client)

    child_token = f"child_{uuid.uuid4().hex}"
    parent_token = f"parent_{uuid.uuid4().hex}"

    child_device_id = await _create_device(client, token, "child-phone", "child_device", child_token)
    await _create_device(client, token, "parent-phone", "parent_device", parent_token)

    with patch("app.main.send_fcm_push", return_value=True) as mocked_send:
        resp = await client.post(
            "/alerts/test-fire",
            headers={"Authorization": f"Bearer {token}"},
            json={"device_id": child_device_id, "confidence_score": 0.11},
        )

    assert resp.status_code == 200
    assert mocked_send.call_count == 1
    called_token = mocked_send.call_args.args[0]
    assert called_token == parent_token
    assert called_token != child_token


@pytest.mark.anyio
async def test_tenant_isolation_alerts(client: AsyncClient):
    token_a, _ = await _register_parent(client)
    token_b, _ = await _register_parent(client)

    device_a = await _create_device(client, token_a, "child-a", "child_device", f"tok_a_{uuid.uuid4().hex}")
    device_b = await _create_device(client, token_b, "child-b", "child_device", f"tok_b_{uuid.uuid4().hex}")

    with patch("app.main.send_fcm_push", return_value=True):
        fire_a = await client.post(
            "/alerts/test-fire",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"device_id": device_a, "confidence_score": 0.21},
        )
        fire_b = await client.post(
            "/alerts/test-fire",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"device_id": device_b, "confidence_score": 0.22},
        )

    assert fire_a.status_code == 200
    assert fire_b.status_code == 200
    alert_a_id = fire_a.json()["alert_id"]
    alert_b_id = fire_b.json()["alert_id"]

    list_a = await client.get("/alerts", headers={"Authorization": f"Bearer {token_a}"})
    list_b = await client.get("/alerts", headers={"Authorization": f"Bearer {token_b}"})

    assert list_a.status_code == 200
    assert list_b.status_code == 200

    items_a = list_a.json()["items"]
    items_b = list_b.json()["items"]

    assert len(items_a) == 1
    assert len(items_b) == 1
    assert items_a[0]["id"] == alert_a_id
    assert items_b[0]["id"] == alert_b_id
    assert alert_b_id not in [x["id"] for x in items_a]
    assert alert_a_id not in [x["id"] for x in items_b]


@pytest.mark.anyio
async def test_tenant_isolation_speakers(client: AsyncClient):
    token_a, _ = await _register_parent(client)
    token_b, _ = await _register_parent(client)

    fake_emb_a = np.ones(192, dtype=np.float32)
    fake_emb_b = np.full(192, 0.5, dtype=np.float32)

    with patch("app.main.run_enroll_embedding", side_effect=[([fake_emb_a], {"ok": True}), ([fake_emb_b], {"ok": True})]):
        enroll_a = await client.post(
            "/enroll/speaker",
            headers={"Authorization": f"Bearer {token_a}"},
            data={"display_name": "Alice"},
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        )
        enroll_b = await client.post(
            "/enroll/speaker",
            headers={"Authorization": f"Bearer {token_b}"},
            data={"display_name": "Bob"},
            files={"file": ("b.wav", _wav_bytes(), "audio/wav")},
        )

    assert enroll_a.status_code == 200
    assert enroll_b.status_code == 200

    list_a = await client.get("/enroll/speakers", headers={"Authorization": f"Bearer {token_a}"})
    list_b = await client.get("/enroll/speakers", headers={"Authorization": f"Bearer {token_b}"})

    assert list_a.status_code == 200
    assert list_b.status_code == 200

    items_a = list_a.json()["items"]
    items_b = list_b.json()["items"]
    assert len(items_a) == 1
    assert len(items_b) == 1
    assert items_a[0]["display_name"] == "Alice"
    assert items_b[0]["display_name"] == "Bob"
    assert items_a[0]["id"] != items_b[0]["id"]


@pytest.mark.anyio
async def test_enrolled_speaker_list_does_not_default_to_poor(client: AsyncClient):
    token, _ = await _register_parent(client)

    emb_1 = np.ones(192, dtype=np.float32)
    emb_2 = np.ones(192, dtype=np.float32) * 0.92

    with patch("app.main.run_enroll_embedding", return_value=([emb_1, emb_2], {"ok": True})):
        enroll = await client.post(
            "/enroll/speaker",
            headers={"Authorization": f"Bearer {token}"},
            data={"display_name": "Quality Check"},
            files={"file": ("q.wav", _wav_bytes(), "audio/wav")},
        )

    assert enroll.status_code == 200

    listed = await client.get("/enroll/speakers", headers={"Authorization": f"Bearer {token}"})
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert len(items) == 1
    assert items[0]["quality_label"] != "poor"


def test_resolve_redis_url_prefers_render_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://render.example:6379/0")
    monkeypatch.setenv("SAFEEAR_REDIS_URL", "redis://legacy.example:6379/0")
    assert _resolve_redis_url() == "redis://render.example:6379/0"

    monkeypatch.delenv("REDIS_URL", raising=False)
    assert _resolve_redis_url() == "redis://legacy.example:6379/0"

    monkeypatch.delenv("SAFEEAR_REDIS_URL", raising=False)
    assert _resolve_redis_url() == "redis://localhost:6379/0"


@pytest.mark.anyio
async def test_detect_chunk_blocked_for_parent_device(client: AsyncClient):
    token, _ = await _register_parent(client)
    parent_device_id = await _create_device(
        client,
        token,
        "parent-only",
        "parent_device",
        f"p_{uuid.uuid4().hex}",
    )

    resp = await client.post(
        "/detect/chunk",
        headers={"Authorization": f"Bearer {token}"},
        data={"device_id": parent_device_id},
        files={"file": ("chunk.wav", _wav_bytes(), "audio/wav")},
    )

    assert resp.status_code == 403


def test_alert_clip_atomic_write(tmp_path):
    original_tenants_root = svc.TENANTS_ROOT
    svc.TENANTS_ROOT = str(tmp_path / "tenants")

    try:
        parent_id = str(uuid.uuid4())
        device_id = str(uuid.uuid4())
        session = svc.get_or_create_session(parent_id, device_id)
        waveform = np.zeros(16000, dtype=np.float32)

        clip_path = svc.save_alert_clip(parent_id, session, waveform)

        assert os.path.exists(clip_path)
        assert not os.path.exists(f"{clip_path}.tmp")

        with open(clip_path, "rb") as handle:
            header = handle.read(4)
        assert header == b"RIFF"
    finally:
        svc.TENANTS_ROOT = original_tenants_root
        svc.stop_session(parent_id, device_id)


def test_embedding_dimension_guard(tmp_path):
    db = SessionLocal()
    parent = None
    try:
        email = f"dim_{uuid.uuid4().hex}@example.com"
        parent = svc.register_parent_with_email(db, email, "password123", "Dim Guard")
        speaker = svc.create_speaker(db, str(parent.id), "BadDimSpeaker")

        speaker_dir = svc.get_speaker_embedding_dir(str(parent.id), str(speaker.id))
        os.makedirs(speaker_dir, exist_ok=True)
        np.save(os.path.join(speaker_dir, "emb_1.npy"), np.zeros(128, dtype=np.float32))

        with pytest.raises(ValueError, match="dim mismatch"):
            svc.load_parent_embeddings(db, str(parent.id))
    finally:
        if parent is not None:
            try:
                parent_row = svc.get_parent(db, str(parent.id))
                db.delete(parent_row)
                db.commit()
            except Exception:
                pass
        db.close()


def test_enrollment_quality_monotone_chunks_not_labeled_good(tmp_path):
    db = SessionLocal()
    parent = None
    try:
        email = f"mono_{uuid.uuid4().hex}@example.com"
        parent = svc.register_parent_with_email(db, email, "password123", "Mono Guard")
        speaker = svc.create_speaker(db, str(parent.id), "Monotone")

        base = np.ones(192, dtype=np.float32)
        for _ in range(4):
            svc.save_speaker_embedding(db, str(parent.id), str(speaker.id), base)

        score, label = svc.compute_and_store_enrollment_quality(db, str(parent.id), str(speaker.id))

        assert score is not None
        assert label in {"fair", "poor"}
        assert label != "good"
    finally:
        if parent is not None:
            try:
                parent_row = svc.get_parent(db, str(parent.id))
                db.delete(parent_row)
                db.commit()
            except Exception:
                pass
        db.close()


@pytest.mark.anyio
async def test_refresh_returns_fresh_access_and_refresh_tokens(client: AsyncClient):
    register = await client.post(
        "/auth/register-email",
        json={"email": f"r_{uuid.uuid4().hex}@example.com", "password": "password123", "display_name": "Refresh"},
    )
    assert register.status_code == 200
    original = register.json()

    refreshed = await client.post(
        "/auth/refresh",
        json={"refresh_token": original["refresh_token"]},
    )
    assert refreshed.status_code == 200
    payload = refreshed.json()
    assert payload["access_token"]
    assert payload["refresh_token"]
    assert payload["access_token"] != original["access_token"]
    assert payload["refresh_token"] != original["refresh_token"]


@pytest.mark.anyio
async def test_device_upsert_same_installation_reuses_row(client: AsyncClient):
    token, _ = await _register_parent(client)
    installation_id = f"inst_{uuid.uuid4().hex}"

    create_resp = await client.post(
        "/devices/upsert",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "device_name": "Kid Phone",
            "installation_id": installation_id,
            "role": "child_device",
            "device_token": f"tok_{uuid.uuid4().hex}",
        },
    )
    assert create_resp.status_code == 200
    first_id = create_resp.json()["device"]["id"]

    switch_resp = await client.post(
        "/devices/upsert",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "device_name": "Parent Phone",
            "installation_id": installation_id,
            "role": "parent_device",
            "device_token": f"tok_{uuid.uuid4().hex}",
        },
    )
    assert switch_resp.status_code == 200
    second = switch_resp.json()["device"]
    assert second["id"] == first_id
    assert second["role"] == "parent_device"

    listed = await client.get("/devices", headers={"Authorization": f"Bearer {token}"})
    assert listed.status_code == 200
    items = listed.json()["items"]
    same_installation = [d for d in items if d.get("installation_id") == installation_id]
    assert len(same_installation) == 1


@pytest.mark.anyio
async def test_patch_device_token_reassigns_uniquely(client: AsyncClient):
    token, _ = await _register_parent(client)
    shared = f"shared_{uuid.uuid4().hex}"
    token_a = f"a_{uuid.uuid4().hex}"
    token_b = f"b_{uuid.uuid4().hex}"

    a_id = await _create_device(client, token, "A", "child_device", token_a, installation_id=f"{shared}_a")
    b_id = await _create_device(client, token, "B", "parent_device", token_b, installation_id=f"{shared}_b")

    patch_resp = await client.patch(
        f"/devices/{a_id}/token",
        headers={"Authorization": f"Bearer {token}"},
        json={"device_token": token_b},
    )
    assert patch_resp.status_code == 200

    listed = await client.get("/devices", headers={"Authorization": f"Bearer {token}"})
    assert listed.status_code == 200
    devices = {d["id"]: d for d in listed.json()["items"]}
    assert devices[a_id]["id"] == a_id
    assert devices[b_id]["id"] == b_id

    db = SessionLocal()
    try:
        row_a = db.query(Device).filter(Device.id == uuid.UUID(a_id)).first()
        row_b = db.query(Device).filter(Device.id == uuid.UUID(b_id)).first()
        assert row_a is not None and row_b is not None
        assert row_a.device_token == token_b
        assert row_b.device_token is None
    finally:
        db.close()


@pytest.mark.anyio
async def test_alert_clip_without_path_returns_empty_response(client: AsyncClient):
    token, parent_id = await _register_parent(client)
    device_id = await _create_device(client, token, "child", "child_device", f"t_{uuid.uuid4().hex}")

    with patch("app.main.send_fcm_push", return_value=True):
        fired = await client.post(
            "/alerts/test-fire",
            headers={"Authorization": f"Bearer {token}"},
            json={"device_id": device_id, "confidence_score": 0.12},
        )
    assert fired.status_code == 200
    alert_id = fired.json()["alert_id"]

    db = SessionLocal()
    try:
        row = db.query(Alert).filter(Alert.parent_id == uuid.UUID(parent_id), Alert.id == uuid.UUID(alert_id)).first()
        assert row is not None
        row.audio_clip_path = None
        db.commit()
    finally:
        db.close()

    clip = await client.get(f"/alerts/{alert_id}/clip", headers={"Authorization": f"Bearer {token}"})
    assert clip.status_code == 204


@pytest.mark.anyio
async def test_detect_chunk_duplicate_chunk_id_returns_cached_response(client: AsyncClient):
    token, _ = await _register_parent(client)
    child_device_id = await _create_device(client, token, "child-dedupe", "child_device", f"dc_{uuid.uuid4().hex}")

    fake_stage = {
        "tier1_vad": {"passed": True},
        "tier2": {"passed": True, "category": "human_speech", "confidence": 0.9},
        "tier3": {"passed": True, "score": 0.9, "closest_speaker_id": None},
        "decision": "familiar",
        "perf": {"processing_ms": 1.0},
        "_query_centroid": np.ones(192, dtype=np.float32),
    }

    with patch("app.main._decode_audio_chunk", return_value=(np.ones(24000, dtype=np.float32), 16000, "mock")), patch(
        "app.utils.prd_services_db.evaluate_window", return_value=fake_stage
    ):
        first = await client.post(
            "/detect/chunk",
            headers={"Authorization": f"Bearer {token}"},
            data={"device_id": child_device_id, "chunk_id": "chunk-42"},
            files={"file": ("c.wav", _wav_bytes(), "audio/wav")},
        )
        second = await client.post(
            "/detect/chunk",
            headers={"Authorization": f"Bearer {token}"},
            data={"device_id": child_device_id, "chunk_id": "chunk-42"},
            files={"file": ("c.wav", _wav_bytes(), "audio/wav")},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json().get("idempotent_replay") is None
    assert second.json().get("idempotent_replay") is True


@pytest.mark.anyio
async def test_detect_chunk_low_confidence_uncertain_does_not_increment_stranger_streak(client: AsyncClient):
    token, _ = await _register_parent(client)
    child_device_id = await _create_device(client, token, "child-noise-guard", "child_device", f"ng_{uuid.uuid4().hex}")

    fake_stage = {
        "tier1_vad": {"passed": True},
        "tier2": {"passed": True, "category": "uncertain", "confidence": 0.2},
        "tier3": {"passed": True, "score": 0.1, "closest_speaker_id": None},
        "decision": "tier3_scored",
        "perf": {"processing_ms": 1.0},
        "_query_centroid": np.ones(192, dtype=np.float32),
    }

    with patch("app.main._decode_audio_chunk", return_value=(np.ones(24000, dtype=np.float32), 16000, "mock")), patch(
        "app.utils.prd_services_db.evaluate_window", return_value=fake_stage
    ):
        resp = await client.post(
            "/detect/chunk",
            headers={"Authorization": f"Bearer {token}"},
            data={"device_id": child_device_id},
            files={"file": ("c.wav", _wav_bytes(), "audio/wav")},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["decision"] == "uncertain_noise"
    assert payload["stranger_streak"] == 0
    assert payload["alert_fired"] is False


@pytest.mark.anyio
async def test_detect_chunk_borderline_scores_tighten_monitoring(client: AsyncClient):
    token, _ = await _register_parent(client)
    child_device_id = await _create_device(client, token, "child-strict-monitoring", "child_device", f"sm_{uuid.uuid4().hex}")

    low_stage = {
        "tier1_vad": {"passed": True},
        "tier2": {"passed": True, "category": "human_speech", "confidence": 0.92},
        "tier3": {"passed": True, "score": 0.25, "closest_speaker_id": None},
        "decision": "tier3_scored",
        "perf": {"processing_ms": 1.0},
        "_query_centroid": np.ones(192, dtype=np.float32),
    }
    borderline_stage = {
        "tier1_vad": {"passed": True},
        "tier2": {"passed": True, "category": "human_speech", "confidence": 0.92},
        "tier3": {"passed": True, "score": 0.4348, "closest_speaker_id": None},
        "decision": "tier3_scored",
        "perf": {"processing_ms": 1.0},
        "_query_centroid": np.ones(192, dtype=np.float32),
    }

    with patch("app.main._decode_audio_chunk", return_value=(np.ones(24000, dtype=np.float32), 16000, "mock")), patch(
        "app.utils.prd_services_db.evaluate_window", side_effect=[low_stage, borderline_stage]
    ):
        first = await client.post(
            "/detect/chunk",
            headers={"Authorization": f"Bearer {token}"},
            data={"device_id": child_device_id},
            files={"file": ("c1.wav", _wav_bytes(), "audio/wav")},
        )
        second = await client.post(
            "/detect/chunk",
            headers={"Authorization": f"Bearer {token}"},
            data={"device_id": child_device_id},
            files={"file": ("c2.wav", _wav_bytes(), "audio/wav")},
        )

    assert first.status_code == 200
    assert first.json()["decision"] == "stranger_candidate"
    assert first.json()["stranger_streak"] == 1

    assert second.status_code == 200
    assert second.json()["decision"] == "uncertain"
    assert second.json()["stranger_streak"] == 1
    assert second.json()["alert_fired"] is False
