"""
Pytest configuration and shared fixtures for the SafeEar backend test suite.

IMPORTANT: Run tests using the project venv, not the system Python:
    .\\venv\\Scripts\\python.exe -m pytest tests/ -v

The system Python may have an incompatible SQLAlchemy version which will cause
an ImportError when the sqlite dialect attempts to load.
"""
import os
import pytest

# Ensure tests use an in-memory SQLite database to avoid polluting dev data.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_safeear.db")


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Each test starts with empty rate-limit windows (all tests share one client IP)."""
    from app.core.security import rate_limiter

    rate_limiter.reset()
    yield
