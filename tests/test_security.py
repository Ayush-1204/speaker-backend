"""Security regressions: token kinds, device scoping, refresh rotation, upload limits."""

import io
import uuid

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from app.core import security
from app.database.models import Device
from app.database.session import SessionLocal, init_db
from app.main import app


@pytest.fixture
async def client():
    init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _wav(seconds: float = 1.0) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


async def _register(client):
    resp = await client.post(
        "/auth/register-email",
        json={"email": f"s_{uuid.uuid4().hex}@example.com", "password": "password123", "display_name": "P"},
    )
    assert resp.status_code == 200
    return resp.json()


async def _device(client, token, role="child_device", fcm=None, installation=None):
    resp = await client.post(
        "/devices",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "device_name": f"d_{uuid.uuid4().hex[:6]}",
            "role": role,
            "device_token": fcm or f"fcm_{uuid.uuid4().hex}",
            "installation_id": installation or uuid.uuid4().hex,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["device"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.anyio
async def test_dev_google_token_is_rejected_by_default(client):
    resp = await client.post("/auth/google", json={"id_token": "dev:email:victim@example.com"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "dev_token_disabled"


def test_production_refuses_default_jwt_secret(monkeypatch):
    monkeypatch.setenv("SAFEEAR_ENV", "production")
    monkeypatch.setattr(security, "JWT_SECRET", "change-me-in-production")
    with pytest.raises(RuntimeError):
        security.validate_security_config()
    monkeypatch.setattr(security, "JWT_SECRET", "x" * 48)
    security.validate_security_config()
    assert security.dev_tokens_allowed() is False


@pytest.mark.anyio
async def test_child_device_token_cannot_use_parent_api(client):
    parent = await _register(client)
    child = await _device(client, parent["access_token"])
    device_token = child["device_access_token"]
    other = await _device(client, parent["access_token"])

    for method, url in [
        ("GET", "/alerts"),
        ("DELETE", "/alerts"),
        ("GET", "/enroll/speakers"),
        ("DELETE", f"/enroll/speakers/{uuid.uuid4()}"),
    ]:
        resp = await client.request(method, url, headers=_auth(device_token))
        assert resp.status_code == 403, (method, url, resp.text)

    resp = await client.patch(
        f"/devices/{child['id']}/monitoring", headers=_auth(device_token), json={"monitoring_enabled": False}
    )
    assert resp.status_code == 403

    # It sees only itself.
    resp = await client.get("/devices", headers=_auth(device_token))
    assert resp.status_code == 200
    assert [d["id"] for d in resp.json()["items"]] == [child["id"]]

    # It cannot stream as another device.
    resp = await client.post(
        "/detect/chunk", headers=_auth(device_token),
        data={"device_id": other["id"]}, files={"file": ("c.wav", _wav(), "audio/wav")},
    )
    assert resp.status_code == 403

    # A heartbeat from the device cannot switch monitoring off.
    resp = await client.post(
        f"/devices/{child['id']}/heartbeat", headers=_auth(device_token),
        json={"battery_percent": 50, "is_online": True, "monitoring_enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json()["device"]["monitoring_enabled"] is True


@pytest.mark.anyio
async def test_reregistration_revokes_old_device_token(client):
    parent = await _register(client)
    installation = uuid.uuid4().hex
    first = await _device(client, parent["access_token"], installation=installation)
    second = await _device(client, parent["access_token"], installation=installation)
    assert first["id"] == second["id"]

    assert (await client.get("/devices", headers=_auth(second["device_access_token"]))).status_code == 200
    resp = await client.get("/devices", headers=_auth(first["device_access_token"]))
    assert resp.status_code == 401
    assert resp.json()["detail"] == "device_token_revoked"


@pytest.mark.anyio
async def test_fcm_token_cannot_move_another_familys_device(client):
    victim = await _register(client)
    attacker = await _register(client)
    fcm = f"fcm_{uuid.uuid4().hex}"
    victim_device = await _device(client, victim["access_token"], fcm=fcm)
    attacker_device = await _device(client, attacker["access_token"], fcm=fcm)

    assert attacker_device["id"] != victim_device["id"]
    db = SessionLocal()
    try:
        row = db.query(Device).filter(Device.id == uuid.UUID(victim_device["id"])).first()
        assert str(row.parent_id) == victim["parent"]["id"]
        assert row.device_token is None  # the token now belongs to the other install
    finally:
        db.close()


@pytest.mark.anyio
async def test_refresh_token_rotation_and_reuse_detection(client):
    parent = await _register(client)
    first = parent["refresh_token"]

    resp = await client.post("/auth/refresh", json={"refresh_token": first})
    assert resp.status_code == 200
    second = resp.json()["refresh_token"]
    assert second != first

    # Replaying the used token is treated as theft: it fails and kills the new one too.
    resp = await client.post("/auth/refresh", json={"refresh_token": first})
    assert resp.status_code == 401
    resp = await client.post("/auth/refresh", json={"refresh_token": second})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_logout_revokes_refresh_token(client):
    parent = await _register(client)
    assert (await client.post("/auth/logout", json={"refresh_token": parent["refresh_token"]})).status_code == 200
    resp = await client.post("/auth/refresh", json={"refresh_token": parent["refresh_token"]})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_oversized_chunk_is_rejected(client):
    parent = await _register(client)
    child = await _device(client, parent["access_token"])
    big = b"\x00" * (600 * 1024)
    resp = await client.post(
        "/detect/chunk", headers=_auth(child["device_access_token"]),
        data={"device_id": child["id"]}, files={"file": ("c.wav", big, "audio/wav")},
    )
    assert resp.status_code == 413


@pytest.mark.anyio
async def test_chunks_are_dropped_when_monitoring_is_off(client):
    parent = await _register(client)
    child = await _device(client, parent["access_token"])
    resp = await client.patch(
        f"/devices/{child['id']}/monitoring", headers=_auth(parent["access_token"]), json={"monitoring_enabled": False}
    )
    assert resp.status_code == 200
    resp = await client.post(
        "/detect/chunk", headers=_auth(child["device_access_token"]),
        data={"device_id": child["id"]}, files={"file": ("c.wav", _wav(), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "monitoring_disabled"


@pytest.mark.anyio
async def test_auth_endpoints_are_rate_limited(client, monkeypatch):
    monkeypatch.setenv("SAFEEAR_AUTH_RATE_LIMIT_PER_MIN", "3")
    for _ in range(3):
        resp = await client.post("/auth/login-email", json={"email": "nobody@example.com", "password": "password123"})
        assert resp.status_code == 401
    resp = await client.post("/auth/login-email", json={"email": "nobody@example.com", "password": "password123"})
    assert resp.status_code == 429


@pytest.mark.anyio
async def test_adding_a_sample_keeps_child_role(client, monkeypatch):
    from unittest.mock import patch
    parent = await _register(client)
    emb = np.ones(192, dtype=np.float32)
    with patch("app.main.run_enroll_embedding", return_value=([emb], {"ok": True})):
        first = await client.post(
            "/enroll/speaker", headers=_auth(parent["access_token"]),
            data={"display_name": "Kid", "role": "child_self"}, files={"audio": ("a.wav", _wav(), "audio/wav")},
        )
        sid = first.json()["speaker_id"]
        await client.post(
            "/enroll/speaker", headers=_auth(parent["access_token"]),
            data={"display_name": "Kid", "speaker_id": sid}, files={"audio": ("b.wav", _wav(), "audio/wav")},
        )
    speakers = (await client.get("/enroll/speakers", headers=_auth(parent["access_token"]))).json()["items"]
    assert [s["role"] for s in speakers if s["id"] == sid] == ["child_self"]

    resp = await client.patch(f"/enroll/speakers/{sid}", headers=_auth(parent["access_token"]), json={"role": "familiar_adult"})
    assert resp.status_code == 200 and resp.json()["speaker"]["role"] == "familiar_adult"
