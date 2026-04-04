"""Database session and connection management."""

import os
from sqlalchemy import create_engine
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
