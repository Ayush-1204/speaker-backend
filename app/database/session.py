"""Database session and connection management."""

import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from app.database.models import Base

# Default to SQLite for local development; set DATABASE_URL env var to override (e.g., PostgreSQL for production).
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./safeear.db"
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Dependency for FastAPI to inject database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    _ensure_device_columns()


def _ensure_device_columns() -> None:
    """Backfill newly added columns for existing deployments without Alembic."""
    inspector = inspect(engine)
    if "devices" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("devices")}
    statements = []

    if "battery_percent" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN battery_percent INTEGER")
    if "is_online" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN is_online BOOLEAN NOT NULL DEFAULT 0")
    if "monitoring_enabled" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN monitoring_enabled BOOLEAN NOT NULL DEFAULT 1")
    if "last_heartbeat_at" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN last_heartbeat_at DATETIME")
    if "last_activity_at" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN last_activity_at DATETIME")

    if not statements:
        return

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
