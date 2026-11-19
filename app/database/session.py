"""Database session and connection management."""

import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from app.database.models import Base

# Default to SQLite for local development; set DATABASE_URL env var to override (e.g., PostgreSQL for production).
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./safeear.db"
)
# Hosts (Render, Heroku, docker-compose) often hand out postgres:// URLs, which
# SQLAlchemy 2.x no longer accepts.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to inject database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables and backfill newly added columns for existing deployments."""
    Base.metadata.create_all(bind=engine)
    _ensure_device_columns()
    _ensure_speaker_columns()
    _ensure_alert_columns()
    _ensure_household_config_table()


def _ensure_device_columns() -> None:
    """Backfill newly added columns for existing deployments without Alembic."""
    inspector = inspect(engine)
    if "devices" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("devices")}
    statements = []

    if "battery_percent" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN battery_percent INTEGER")
    if "installation_id" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN installation_id VARCHAR")
    if "is_online" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN is_online BOOLEAN NOT NULL DEFAULT FALSE")
    if "monitoring_enabled" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN monitoring_enabled BOOLEAN NOT NULL DEFAULT TRUE")
    if "last_heartbeat_at" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN last_heartbeat_at TIMESTAMP")
    if "last_activity_at" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN last_activity_at TIMESTAMP")
    if "access_token_version" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN access_token_version INTEGER NOT NULL DEFAULT 0")
    if "last_location_accuracy_m" not in existing:
        statements.append("ALTER TABLE devices ADD COLUMN last_location_accuracy_m FLOAT")

    if not statements:
        return

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _ensure_speaker_columns() -> None:
    """Backfill enrolled_speakers columns for quality scoring, role, and reliability features."""
    inspector = inspect(engine)
    if "enrolled_speakers" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("enrolled_speakers")}
    statements = []

    # Original quality columns
    if "quality_score" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN quality_score FLOAT")
    if "quality_label" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN quality_label VARCHAR")

    # Change 1: speaker role + per-speaker threshold override
    if "role" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN role VARCHAR NOT NULL DEFAULT 'familiar_adult'")
    if "t_high_override" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN t_high_override FLOAT")

    # Change 4: confusability
    if "confusability_score" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN confusability_score FLOAT")
    if "confusability_label" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN confusability_label VARCHAR")

    # Change 6: ZT-norm cohort statistics
    if "cohort_mean" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN cohort_mean FLOAT")
    if "cohort_std" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN cohort_std FLOAT")

    # Change 10: drift tracking
    if "drift_score_ema" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN drift_score_ema FLOAT")
    if "last_drift_check_at" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN last_drift_check_at TIMESTAMP")
    if "drift_alert_sent_at" not in existing:
        statements.append("ALTER TABLE enrolled_speakers ADD COLUMN drift_alert_sent_at TIMESTAMP")

    if not statements:
        return

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _ensure_alert_columns() -> None:
    """Backfill alerts columns for Change 7 (alert_type)."""
    inspector = inspect(engine)
    if "alerts" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("alerts")}
    statements = []

    # Change 7: distress vs identity alert distinction
    if "alert_type" not in existing:
        statements.append("ALTER TABLE alerts ADD COLUMN alert_type VARCHAR NOT NULL DEFAULT 'unenrolled_contact'")

    if not statements:
        return

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _ensure_household_config_table() -> None:
    """Ensure household_configs table exists (Change 5 + Change 12).

    SQLAlchemy create_all already handles this for fresh installs; this function
    is a no-op if the table already exists, so it is safe to call on every startup.
    """
    inspector = inspect(engine)
    if "household_configs" not in inspector.get_table_names():
        # create_all already ran above; if the table is still missing the model
        # import failed, which is a fatal error — let it surface naturally.
        return

    existing = {col["name"] for col in inspector.get_columns("household_configs")}
    statements = []
    if "parent_proximity_enabled" not in existing:
        statements.append(
            "ALTER TABLE household_configs ADD COLUMN parent_proximity_enabled BOOLEAN NOT NULL DEFAULT FALSE"
        )
    if "parent_proximity_radius_m" not in existing:
        statements.append(
            "ALTER TABLE household_configs ADD COLUMN parent_proximity_radius_m FLOAT NOT NULL DEFAULT 10.0"
        )
    if statements:
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
