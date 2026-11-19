"""Child-watch features: code pairing, device-started monitoring, household config,
parent location, parent-nearby pause, SOS (+ clip)."""

import io
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

import app.main as main_module
from app.database.models import Device, DeviceRole
from app.database.session import SessionLocal, init_db
from app.main import app
from app.services import child_safety
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


@pytest.fixture(autouse=True)
def _no_pushes(monkeypatch):
    """No Firebase/SMS/email from tests."""
    sent = []
    monkeypatch.setattr(main_module, "_push_sos", lambda *a, **k: sent.append(("sos", a)))
    monkeypatch.setattr(svc, "_send_monitoring_command_to_device", lambda device, enabled: False)
    from app.utils import notification_worker

    monkeypatch.setattr(notification_worker, "send_status_push", lambda *a, **k: sent.append(("status", a)))
    monkeypatch.setattr(child_safety, "PROXIMITY_CACHE_MS", 0)
    return sent


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _wav(seconds: float = 1.5, amp: float = 0.1) -> bytes:
    buf = io.BytesIO()
    t = np.arange(int(16000 * seconds)) / 16000.0
    sf.write(buf, (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000, format="WAV")
    return buf.getvalue()


async def _register(client):
    resp = await client.post(
        "/auth/register-email",
        json={"email": f"w_{uuid.uuid4().hex}@example.com", "password": "password123", "display_name": "P"},
    )
    assert resp.status_code == 200
    return resp.json()


async def _parent_phone(client, token):
    resp = await client.post(
        "/devices",
        headers=_auth(token),
        data={"device_name": f"phone_{uuid.uuid4().hex[:6]}", "role": "parent_device",
              "device_token": f"fcm_{uuid.uuid4().hex}", "installation_id": uuid.uuid4().hex},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["device"]


async def _pair_watch(client, parent_token, name="Galaxy Watch4"):
    start = await client.post("/pair/start", json={"installation_id": uuid.uuid4().hex, "device_name": name})
    assert start.status_code == 200, start.text
    s = start.json()
    assert len(s["code"]) == 6 and s["expires_in"] > 0
    pending = await client.post("/pair/status", json={"pair_id": s["pair_id"], "pair_secret": s["pair_secret"]})
    assert pending.json() == {"status": "pending"}
    claim = await client.post("/pair/claim", headers=_auth(parent_token), json={"code": s["code"]})
    assert claim.status_code == 200, claim.text
    assert claim.json()["device"]["role"] == "child_device"
    done = await client.post("/pair/status", json={"pair_id": s["pair_id"], "pair_secret": s["pair_secret"]})
    assert done.status_code == 200
    body = done.json()
    assert body["status"] == "paired" and body["device_access_token"]
    return s, body


# --------------------------------------------------------------------------- pairing

@pytest.mark.anyio
async def test_pairing_flow_gives_a_working_device_token_once(client):
    parent = await _register(client)
    s, paired = await _pair_watch(client, parent["access_token"])
    # The token works for the device's own endpoints.
    devs = await client.get("/devices", headers=_auth(paired["device_access_token"]))
    assert devs.status_code == 200
    item = devs.json()["items"][0]
    assert item["id"] == paired["device_id"] and item["device_name"] == "Galaxy Watch4"
    assert item["precise_location"] is False
    # ...but not for parent-only endpoints.
    assert (await client.get("/alerts", headers=_auth(paired["device_access_token"]))).status_code == 403
    # The token is handed out only once, and the code cannot be reused.
    again = await client.post("/pair/status", json={"pair_id": s["pair_id"], "pair_secret": s["pair_secret"]})
    assert again.status_code == 410
    reuse = await client.post("/pair/claim", headers=_auth(parent["access_token"]), json={"code": s["code"]})
    assert reuse.status_code == 404


@pytest.mark.anyio
async def test_pairing_rejects_wrong_secret_and_unknown_code(client):
    parent = await _register(client)
    s = (await client.post("/pair/start", json={"installation_id": uuid.uuid4().hex})).json()
    bad = await client.post("/pair/status", json={"pair_id": s["pair_id"], "pair_secret": "nope"})
    assert bad.status_code == 404
    unknown = "000000" if s["code"] != "000000" else "111111"
    resp = await client.post("/pair/claim", headers=_auth(parent["access_token"]), json={"code": unknown})
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_pairing_code_expires(client):
    parent = await _register(client)
    s = (await client.post("/pair/start", json={"installation_id": uuid.uuid4().hex})).json()
    with SessionLocal() as db:
        from app.database.models import PairingRequest

        req = db.query(PairingRequest).filter(PairingRequest.id == uuid.UUID(s["pair_id"])).one()
        req.expires_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
        db.commit()
    claim = await client.post("/pair/claim", headers=_auth(parent["access_token"]), json={"code": s["code"]})
    assert claim.status_code == 404
    status = await client.post("/pair/status", json={"pair_id": s["pair_id"], "pair_secret": s["pair_secret"]})
    assert status.status_code == 410


# --------------------------------------------------------------------------- monitoring

@pytest.mark.anyio
async def test_device_may_start_but_not_stop_monitoring(client, monkeypatch):
    parent = await _register(client)
    _, paired = await _pair_watch(client, parent["access_token"])
    dev_token, dev_id = paired["device_access_token"], paired["device_id"]
    # No enrolled voice yet: starting is refused for the device as for the parent.
    resp = await client.patch(f"/devices/{dev_id}/monitoring", headers=_auth(dev_token), json={"monitoring_enabled": True})
    assert resp.status_code == 409
    monkeypatch.setattr(svc, "list_speakers", lambda db, pid: [object()])
    resp = await client.patch(f"/devices/{dev_id}/monitoring", headers=_auth(dev_token), json={"monitoring_enabled": True})
    assert resp.status_code == 200 and resp.json()["monitoring_enabled"] is True
    resp = await client.patch(f"/devices/{dev_id}/monitoring", headers=_auth(dev_token), json={"monitoring_enabled": False})
    assert resp.status_code == 403
    # Another device id is out of reach.
    other = str(uuid.uuid4())
    resp = await client.patch(f"/devices/{other}/monitoring", headers=_auth(dev_token), json={"monitoring_enabled": True})
    assert resp.status_code == 403
    # The parent can stop it.
    resp = await client.patch(
        f"/devices/{dev_id}/monitoring", headers=_auth(parent["access_token"]), json={"monitoring_enabled": False}
    )
    assert resp.status_code == 200 and resp.json()["monitoring_enabled"] is False


# --------------------------------------------------------------------------- config + proximity

@pytest.mark.anyio
async def test_household_config_defaults_and_validation(client):
    parent = await _register(client)
    h = _auth(parent["access_token"])
    assert (await client.get("/household/config", headers=h)).json() == {
        "parent_proximity_enabled": False, "parent_proximity_radius_m": 10.0,
    }
    resp = await client.patch("/household/config", headers=h, json={"parent_proximity_enabled": True,
                                                                      "parent_proximity_radius_m": 15})
    assert resp.json() == {"parent_proximity_enabled": True, "parent_proximity_radius_m": 15.0}
    assert (await client.patch("/household/config", headers=h, json={"parent_proximity_radius_m": 1})).status_code == 422
    _, paired = await _pair_watch(client, parent["access_token"])
    dev_h = _auth(paired["device_access_token"])
    assert (await client.get("/household/config", headers=dev_h)).status_code == 403
    assert (await client.get("/devices", headers=dev_h)).json()["items"][0]["precise_location"] is True


@pytest.mark.anyio
async def test_parent_location_requires_parent_device(client):
    parent = await _register(client)
    h = _auth(parent["access_token"])
    _, paired = await _pair_watch(client, parent["access_token"])
    resp = await client.post("/parent/location", headers=h,
                             json={"device_id": paired["device_id"], "latitude": 1.0, "longitude": 1.0})
    assert resp.status_code == 403
    resp = await client.post("/parent/location", headers=_auth(paired["device_access_token"]),
                             json={"device_id": paired["device_id"], "latitude": 1.0, "longitude": 1.0})
    assert resp.status_code == 403


def test_haversine_scale():
    # 0.0001 degree of latitude is about 11.1 m.
    assert 10.5 < child_safety.haversine_m(12.0, 77.0, 12.0001, 77.0) < 11.7


async def _setup_nearby(client, child_offset_deg=0.00002, accuracy=5.0):
    parent = await _register(client)
    h = _auth(parent["access_token"])
    phone = await _parent_phone(client, parent["access_token"])
    _, paired = await _pair_watch(client, parent["access_token"])
    dev_h = _auth(paired["device_access_token"])
    await client.patch("/household/config", headers=h, json={"parent_proximity_enabled": True,
                                                              "parent_proximity_radius_m": 10})
    r = await client.post("/parent/location", headers=h, json={"device_id": phone["id"], "latitude": 12.9716,
                                                               "longitude": 77.5946, "accuracy_m": accuracy})
    assert r.status_code == 200
    r = await client.post("/detect/location", headers=dev_h,
                          json={"device_id": paired["device_id"], "latitude": 12.9716 + child_offset_deg,
                                "longitude": 77.5946, "accuracy_m": accuracy, "battery_percent": 80})
    assert r.status_code == 200
    return parent, phone, paired


@pytest.mark.anyio
async def test_chunk_is_paused_when_parent_is_near(client):
    parent, _, paired = await _setup_nearby(client)
    resp = await client.post(
        "/detect/chunk", headers=_auth(paired["device_access_token"]),
        data={"device_id": paired["device_id"], "seq": "0", "battery_percent": "80"},
        files={"audio": ("c.wav", _wav(), "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "paused_parent_nearby"
    # Rule off -> detection runs again (any status other than paused).
    await client.patch("/household/config", headers=_auth(parent["access_token"]),
                       json={"parent_proximity_enabled": False})
    resp = await client.post(
        "/detect/chunk", headers=_auth(paired["device_access_token"]),
        data={"device_id": paired["device_id"], "seq": "1"},
        files={"audio": ("c.wav", _wav(), "audio/wav")},
    )
    assert resp.json()["status"] != "paused_parent_nearby"


@pytest.mark.anyio
async def test_proximity_is_fail_safe(client):
    # Too far.
    parent, phone, paired = await _setup_nearby(client, child_offset_deg=0.0005)
    with SessionLocal() as db:
        child = svc.get_device(db, parent["parent"]["id"], paired["device_id"])
        res = child_safety.parent_nearby(db, parent["parent"]["id"], child)
        assert res["near"] is False and res["distance_m"] > 10
    # Inaccurate fixes.
    parent, phone, paired = await _setup_nearby(client, accuracy=80.0)
    with SessionLocal() as db:
        child = svc.get_device(db, parent["parent"]["id"], paired["device_id"])
        assert child_safety.parent_nearby(db, parent["parent"]["id"], child)["near"] is False
    # Stale parent fix.
    parent, phone, paired = await _setup_nearby(client)
    with SessionLocal() as db:
        p = svc.get_device(db, parent["parent"]["id"], phone["id"])
        p.last_location_ts = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)
        db.commit()
        child = svc.get_device(db, parent["parent"]["id"], paired["device_id"])
        res = child_safety.parent_nearby(db, parent["parent"]["id"], child)
        assert res == {"near": False, "reason": "parent_fix_unusable"}


# --------------------------------------------------------------------------- SOS

@pytest.mark.anyio
async def test_sos_creates_alert_even_when_monitoring_is_off(client, _no_pushes):
    parent = await _register(client)
    h = _auth(parent["access_token"])
    _, paired = await _pair_watch(client, parent["access_token"])
    dev_h = _auth(paired["device_access_token"])
    await client.patch(f"/devices/{paired['device_id']}/monitoring", headers=h, json={"monitoring_enabled": False})
    resp = await client.post("/sos", headers=dev_h, json={"device_id": paired["device_id"], "latitude": 12.97,
                                                          "longitude": 77.59, "accuracy_m": 8, "battery_percent": 55})
    assert resp.status_code == 200, resp.text
    alert = resp.json()["alert"]
    assert alert["alert_type"] == "sos" and alert["latitude"] == 12.97 and resp.json()["deduplicated"] is False
    assert any(kind == "sos" for kind, _ in _no_pushes)
    # A second press within the dedupe window returns the same alert.
    again = await client.post("/sos", headers=dev_h, json={"device_id": paired["device_id"]})
    assert again.json()["deduplicated"] is True and again.json()["alert"]["id"] == alert["id"]
    # The parent sees it with its type.
    items = (await client.get("/alerts", headers=h)).json()["items"]
    assert [a["alert_type"] for a in items if a["id"] == alert["id"]] == ["sos"]

    # Audio clip upload.
    clip = await client.post(f"/sos/{alert['id']}/clip", headers=dev_h,
                             files={"audio": ("sos.wav", _wav(10.0), "audio/wav")})
    assert clip.status_code == 200, clip.text
    assert clip.json()["alert"]["audio_clip_path"]
    got = await client.get(f"/alerts/{alert['id']}/clip", headers=h)
    assert got.status_code == 200 and len(got.content) > 1000


@pytest.mark.anyio
async def test_sos_is_scoped_to_the_own_device(client):
    parent = await _register(client)
    _, a = await _pair_watch(client, parent["access_token"], "A")
    _, b = await _pair_watch(client, parent["access_token"], "B")
    resp = await client.post("/sos", headers=_auth(a["device_access_token"]), json={"device_id": b["device_id"]})
    assert resp.status_code == 403
    phone = await _parent_phone(client, parent["access_token"])
    resp = await client.post("/sos", headers=_auth(parent["access_token"]), json={"device_id": phone["id"]})
    assert resp.status_code == 403


# --------------------------------------------------------------------------- alert clip

def test_alert_clip_gets_post_roll_and_silence_padding(tmp_path):
    session = svc.SessionState(parent_id=str(uuid.uuid4()), device_id=str(uuid.uuid4()))
    sr = session.sr
    speech = np.full(int(1.5 * sr), 0.1, dtype=np.float32)
    svc.append_frame(session, speech)
    svc.pad_clip_silence(session, sr)  # 1 s the device skipped as silent
    svc.append_frame(session, speech)
    pre = svc.get_clip_buffer_waveform(session, np.zeros(0, dtype=np.float32))
    assert pre.shape[0] == int(4.0 * sr)
    assert np.all(pre[int(1.5 * sr): int(2.5 * sr)] == 0)

    path = str(tmp_path / "clip.wav")
    sf.write(path, pre, sr)
    svc.start_post_roll(session, path, pre)
    svc.append_frame(session, speech)
    assert sf.info(path).frames == pre.shape[0]  # not rewritten yet
    for _ in range(4):
        svc.append_frame(session, speech)
    assert session.post_roll_clip_path is None
    expected = pre.shape[0] + int(svc.ALERT_POST_ROLL_SEC * sr)
    assert sf.info(path).frames == expected
