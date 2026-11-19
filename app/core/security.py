"""Security primitives: environment detection, signed tokens, rate limiting.

Token kinds (the `typ` claim):
  access  - a parent's session token (also the default for tokens issued before
            `typ` existed).  Grants the full parent API.
  device  - a child/monitoring device's token.  Carries `did` (device id) and
            `ver` (Device.access_token_version) and only grants the device API:
            streaming audio, location, heartbeat and its own FCM token.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

_DEFAULT_SECRET = "change-me-in-production"
MIN_SECRET_LENGTH = 32


def is_production() -> bool:
    """SAFEEAR_ENV wins; otherwise Render deployments (RENDER is set) count as production."""
    env = (os.environ.get("SAFEEAR_ENV") or "").strip().lower()
    if env:
        return env in {"prod", "production"}
    return bool(os.environ.get("RENDER"))


def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "true" if default else "false").strip().lower() in {"1", "true", "yes", "on"}


JWT_SECRET = os.environ.get("SAFEEAR_JWT_SECRET", _DEFAULT_SECRET)
ACCESS_TOKEN_TTL_SEC = int(os.environ.get("SAFEEAR_JWT_EXPIRE_SEC", "86400"))
DEVICE_TOKEN_TTL_SEC = int(os.environ.get("SAFEEAR_DEVICE_TOKEN_EXPIRE_SEC", str(365 * 86400)))
REFRESH_TOKEN_TTL_SEC = int(os.environ.get("SAFEEAR_REFRESH_TOKEN_EXPIRE_SEC", str(30 * 86400)))


def validate_security_config() -> None:
    """Refuse to start in production with an unsafe configuration."""
    weak_secret = JWT_SECRET == _DEFAULT_SECRET or len(JWT_SECRET) < MIN_SECRET_LENGTH
    if is_production():
        if weak_secret:
            raise RuntimeError(
                "SAFEEAR_JWT_SECRET must be set to a random value of at least "
                f"{MIN_SECRET_LENGTH} characters in production "
                "(e.g. python -c \"import secrets; print(secrets.token_urlsafe(48))\")."
            )
    elif weak_secret:
        logger.warning("SECURITY_WEAK_JWT_SECRET | using a development secret; never deploy like this")


def dev_tokens_allowed() -> bool:
    """`dev:<sub>` Google tokens are a local-testing shortcut, never available in production."""
    return not is_production() and _env_flag("SAFEEAR_ALLOW_DEV_GOOGLE_TOKEN", False)


def test_endpoints_allowed() -> bool:
    return _env_flag("SAFEEAR_ENABLE_TEST_ENDPOINTS", not is_production())


# --------------------------------------------------------------------------- #
# Signed tokens (HS256 JWT)
# --------------------------------------------------------------------------- #

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64d(data: str) -> bytes:
    pad = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _sign(claims: Dict[str, Any]) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    payload = _b64(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(JWT_SECRET.encode("utf-8"), f"{header}.{payload}".encode("ascii"), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64(sig)}"


def make_access_token(parent_id: str) -> str:
    now = int(time.time())
    return _sign({
        "sub": str(parent_id),
        "typ": "access",
        "iat": now,
        "exp": now + ACCESS_TOKEN_TTL_SEC,
        "jti": secrets.token_urlsafe(8),
    })


def make_device_token(parent_id: str, device_id: str, version: int) -> str:
    now = int(time.time())
    return _sign({
        "sub": str(parent_id),
        "typ": "device",
        "did": str(device_id),
        "ver": int(version),
        "iat": now,
        "exp": now + DEVICE_TOKEN_TTL_SEC,
        "jti": secrets.token_urlsafe(8),
    })


def parse_token(token: str) -> Dict[str, Any]:
    try:
        header, payload, sig = token.split(".")
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid_token_format") from exc
    expected = _b64(hmac.new(JWT_SECRET.encode("utf-8"), f"{header}.{payload}".encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=401, detail="invalid_token_signature")
    try:
        claims = json.loads(_b64d(payload).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="invalid_token_payload") from exc
    if int(claims.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="token_expired")
    claims.setdefault("typ", "access")
    return claims


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Rate limiting (single-process sliding window)
# --------------------------------------------------------------------------- #

class RateLimiter:
    """In-memory sliding-window limiter.  Per process: fine for the single-worker
    deployment; a multi-worker deployment would need a shared store."""

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()

    def hit(self, key: str, limit: int, window_sec: float) -> bool:
        """Record a hit; return False when the key is over its limit."""
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > window_sec:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if now - self._last_prune > 300:
                self._prune(now)
            return True

    def _prune(self, now: float) -> None:
        stale = [k for k, q in self._hits.items() if not q or now - q[-1] > 3600]
        for k in stale:
            del self._hits[k]
        self._last_prune = now

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()


def client_ip(request: Request) -> str:
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(key: str, limit: int, window_sec: float) -> None:
    if not rate_limiter.hit(key, limit, window_sec):
        raise HTTPException(status_code=429, detail="too_many_requests")


def auth_rate_limit(request: Request) -> None:
    """FastAPI dependency for authentication endpoints: per-IP limit."""
    limit = int(os.environ.get("SAFEEAR_AUTH_RATE_LIMIT_PER_MIN", "30"))
    enforce_rate_limit(f"auth:{client_ip(request)}", limit, 60.0)
