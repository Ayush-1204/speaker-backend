"""Parent refresh tokens stored in the database (hashed, rotated, revocable)."""

import json
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Tuple

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.security import REFRESH_TOKEN_TTL_SEC, hash_token
from app.database.models import Parent, RefreshToken

logger = logging.getLogger(__name__)

LEGACY_STATE_PATH = os.path.join("app", "database", "app_state.json")


def _now() -> datetime:
    # Stored naive-UTC to match how other DateTime columns are compared.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def issue_refresh_token(db: Session, parent_id: str) -> str:
    raw = secrets.token_urlsafe(32)
    db.add(RefreshToken(
        id=uuid.uuid4(),
        parent_id=uuid.UUID(str(parent_id)),
        token_hash=hash_token(raw),
        created_at=_now(),
        expires_at=_now() + timedelta(seconds=REFRESH_TOKEN_TTL_SEC),
    ))
    db.commit()
    return raw


def rotate_refresh_token(db: Session, raw: str) -> Tuple[str, str]:
    """Consume a refresh token and issue its replacement: returns (parent_id, new_token).

    Presenting an already-used token means it was copied; every session of that
    parent is revoked so the copy and the original both stop working.
    """
    row = db.query(RefreshToken).filter(RefreshToken.token_hash == hash_token(raw or "")).first()
    if row is None:
        raise HTTPException(status_code=401, detail="invalid_refresh_token")
    if row.revoked_at is not None:
        revoke_all_for_parent(db, str(row.parent_id))
        logger.warning("REFRESH_TOKEN_REUSE_DETECTED | parent_id=%s | all sessions revoked", row.parent_id)
        raise HTTPException(status_code=401, detail="refresh_token_reused")
    if row.expires_at < _now():
        row.revoked_at = _now()
        db.commit()
        raise HTTPException(status_code=401, detail="refresh_token_expired")
    row.revoked_at = _now()
    db.commit()
    return str(row.parent_id), issue_refresh_token(db, str(row.parent_id))


def revoke_refresh_token(db: Session, raw: str) -> None:
    row = db.query(RefreshToken).filter(RefreshToken.token_hash == hash_token(raw or "")).first()
    if row is not None and row.revoked_at is None:
        row.revoked_at = _now()
        db.commit()


def revoke_all_for_parent(db: Session, parent_id: str) -> int:
    count = (
        db.query(RefreshToken)
        .filter(RefreshToken.parent_id == uuid.UUID(str(parent_id)), RefreshToken.revoked_at.is_(None))
        .update({RefreshToken.revoked_at: _now()}, synchronize_session=False)
    )
    db.commit()
    return int(count or 0)


def import_legacy_refresh_tokens(db: Session, path: str = LEGACY_STATE_PATH) -> int:
    """One-time import of refresh tokens from the old app_state.json file.

    Keeps existing app sessions (including child phones) logged in across the
    upgrade.  The file is renamed afterwards so the import never repeats.
    """
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception as exc:
        logger.warning("LEGACY_REFRESH_IMPORT_SKIPPED | reason=%s", exc)
        return 0

    imported = 0
    now_ms = int(_now().replace(tzinfo=timezone.utc).timestamp() * 1000)
    known_parents = {str(pid) for (pid,) in db.query(Parent.id).all()}
    for raw, meta in (state.get("refresh_tokens") or {}).items():
        try:
            parent_id = str(meta.get("parent_id"))
            expires_ms = int(meta.get("expires_at", 0))
            if parent_id not in known_parents or expires_ms <= now_ms:
                continue
            token_hash = hash_token(raw)
            if db.query(RefreshToken.id).filter(RefreshToken.token_hash == token_hash).first():
                continue
            db.add(RefreshToken(
                id=uuid.uuid4(),
                parent_id=uuid.UUID(parent_id),
                token_hash=token_hash,
                created_at=_now(),
                expires_at=datetime.fromtimestamp(expires_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None),
            ))
            imported += 1
        except Exception:
            continue
    db.commit()
    try:
        os.replace(path, path + ".migrated")
    except OSError as exc:
        logger.warning("LEGACY_REFRESH_IMPORT_RENAME_FAILED | reason=%s", exc)
    logger.info("LEGACY_REFRESH_IMPORT_DONE | imported=%d", imported)
    return imported
