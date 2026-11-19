"""Phase 6: personal-data lifecycle (clip retention, alert and account deletion)."""

import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.database.models import Alert, Device, EnrolledSpeaker, Parent, RefreshToken
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


def _make_alert_with_clip(db, parent_id, device_id, age_days):
    session = svc.SessionState(parent_id=parent_id, device_id=device_id)
    clip = svc.save_alert_clip(parent_id, session, np.zeros(16000, dtype=np.float32))
    alert = Alert(
        id=uuid.uuid4(), parent_id=uuid.UUID(parent_id), device_id=uuid.UUID(device_id),
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=age_days),
        confidence_score=0.2, audio_clip_path=clip, alert_type="unenrolled_contact",
    )
    db.add(alert)
    db.commit()
    return alert, clip


def _family(db):
    parent = svc.register_parent_with_email(db, f"d_{uuid.uuid4().hex}@example.com", "password123", "D")
    device = svc.create_device(db, str(parent.id), "kid", svc.DeviceRole.child_device, installation_id=uuid.uuid4().hex)
    return str(parent.id), str(device.id)


def test_old_alert_clips_are_purged_but_alerts_kept():
    init_db()
    db = SessionLocal()
    try:
        pid, did = _family(db)
        old, old_clip = _make_alert_with_clip(db, pid, did, age_days=40)
        new, new_clip = _make_alert_with_clip(db, pid, did, age_days=1)

        assert svc.purge_old_alert_clips(db, retention_days=30) >= 1
        db.refresh(old)
        db.refresh(new)
        assert old.audio_clip_path is None and not os.path.exists(os.path.dirname(old_clip))
        assert new.audio_clip_path == new_clip and os.path.exists(new_clip)
    finally:
        db.close()


def test_deleting_an_alert_removes_its_folder():
    init_db()
    db = SessionLocal()
    try:
        pid, did = _family(db)
        alert, clip = _make_alert_with_clip(db, pid, did, age_days=0)
        assert os.path.exists(os.path.join(os.path.dirname(clip), "metadata.json"))
        svc.delete_alert(db, pid, str(alert.id))
        assert not os.path.exists(os.path.dirname(clip))
    finally:
        db.close()


@pytest.mark.anyio
async def test_account_deletion_removes_everything(client):
    reg = (await client.post(
        "/auth/register-email",
        json={"email": f"x_{uuid.uuid4().hex}@example.com", "password": "password123", "display_name": "X"},
    )).json()
    token, pid = reg["access_token"], reg["parent"]["id"]
    auth = {"Authorization": f"Bearer {token}"}

    db = SessionLocal()
    try:
        speaker = svc.create_speaker(db, pid, "Mom")
        svc.save_speaker_embedding(db, pid, str(speaker.id), np.ones(192, dtype=np.float32))
        _, did = pid, svc.create_device(db, pid, "kid", svc.DeviceRole.child_device, installation_id="i1").id
        _make_alert_with_clip(db, pid, str(did), age_days=0)
    finally:
        db.close()
    tenant_dir = os.path.join(svc.TENANTS_ROOT, pid)
    assert os.path.isdir(tenant_dir)

    assert (await client.request("DELETE", "/auth/account", headers=auth, json={"confirm": "nope"})).status_code == 422
    resp = await client.request("DELETE", "/auth/account", headers=auth, json={"confirm": "DELETE"})
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        puid = uuid.UUID(pid)
        assert db.query(Parent).filter(Parent.id == puid).count() == 0
        for model in (Device, EnrolledSpeaker, Alert, RefreshToken):
            assert db.query(model).filter(model.parent_id == puid).count() == 0
    finally:
        db.close()
    assert not os.path.exists(tenant_dir)
    # The old session no longer works.
    assert (await client.get("/alerts", headers=auth)).status_code == 401
    assert (await client.post("/auth/refresh", json={"refresh_token": reg["refresh_token"]})).status_code == 401
