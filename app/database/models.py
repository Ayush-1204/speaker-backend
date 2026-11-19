"""SQLAlchemy models for SafeEar PRD."""

from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import DateTime, Float, ForeignKey, String, UUID, Integer, Enum as SQLEnum, Boolean
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column
import uuid
import enum

Base = declarative_base()


class SpeakerRole(str, enum.Enum):
    """Role of an enrolled speaker in the household threat model.

    familiar_adult  — parent, sibling, carer, teacher; a high-confidence match
                      suppresses the stranger streak AND updates the familiar
                      grace window.
    child_self      — the monitored child; a match suppresses the stranger
                      streak but does NOT update last_familiar_ms / grace window
                      (the child's presence is irrelevant to the threat model).
                      Uses a looser t_high_override threshold because adult-trained
                      speaker verification degrades on children.
    """
    familiar_adult = "familiar_adult"
    child_self = "child_self"


class Parent(Base):
    __tablename__ = "parents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    google_sub: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    email: Mapped[Optional[str]] = mapped_column(String)
    display_name: Mapped[Optional[str]] = mapped_column(String)
    phone_number: Mapped[Optional[str]] = mapped_column(String)
    fcm_token: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    devices: Mapped[List["Device"]] = relationship("Device", back_populates="parent", cascade="all, delete-orphan")
    speakers: Mapped[List["EnrolledSpeaker"]] = relationship("EnrolledSpeaker", back_populates="parent", cascade="all, delete-orphan")
    alerts: Mapped[List["Alert"]] = relationship("Alert", back_populates="parent", cascade="all, delete-orphan")
    credentials: Mapped[List["AuthCredential"]] = relationship("AuthCredential", back_populates="parent", cascade="all, delete-orphan")


class AuthProvider(str, enum.Enum):
    email_password = "email_password"


class AuthCredential(Base):
    __tablename__ = "auth_credentials"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=False, index=True)
    provider: Mapped[AuthProvider] = mapped_column(SQLEnum(AuthProvider), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    reset_token_hash: Mapped[Optional[str]] = mapped_column(String)
    reset_token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    parent: Mapped["Parent"] = relationship("Parent", back_populates="credentials")


class RefreshToken(Base):
    """Parent refresh tokens.  Only the SHA-256 of the token is stored."""
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class DeviceRole(str, enum.Enum):
    child_device = "child_device"
    parent_device = "parent_device"


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=False, index=True)
    installation_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    device_name: Mapped[Optional[str]] = mapped_column(String)
    role: Mapped[DeviceRole] = mapped_column(SQLEnum(DeviceRole), nullable=False, index=True)
    device_token: Mapped[Optional[str]] = mapped_column(String, unique=True, index=True)
    battery_percent: Mapped[Optional[int]] = mapped_column(Integer)
    is_online: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    monitoring_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Bumped whenever the device is (re-)registered or its role changes; device
    # tokens carrying an older version are rejected, which revokes them.
    access_token_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_location_lat: Mapped[Optional[float]] = mapped_column(Float)
    last_location_lon: Mapped[Optional[float]] = mapped_column(Float)
    last_location_ts: Mapped[Optional[datetime]] = mapped_column(DateTime)
    # Reported horizontal accuracy (metres) of the last fix; used by the parent-nearby rule.
    last_location_accuracy_m: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    parent: Mapped["Parent"] = relationship("Parent", back_populates="devices")
    alerts: Mapped[List["Alert"]] = relationship("Alert", back_populates="device", cascade="all, delete-orphan")


class EnrolledSpeaker(Base):
    __tablename__ = "enrolled_speakers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    sample_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    quality_score: Mapped[Optional[float]] = mapped_column(Float)
    quality_label: Mapped[Optional[str]] = mapped_column(String)

    # --- Change 1: Speaker role for 3-category decision logic ---
    role: Mapped[str] = mapped_column(String, nullable=False, default="familiar_adult")
    t_high_override: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- Change 4: Confusability detection ---
    confusability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confusability_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # --- Change 6: ZT-norm cohort statistics ---
    cohort_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cohort_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- Change 10: Drift tracking for child speaker ---
    drift_score_ema: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_drift_check_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    drift_alert_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    parent: Mapped["Parent"] = relationship("Parent", back_populates="speakers")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=False, index=True)
    device_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("devices.id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    audio_clip_path: Mapped[Optional[str]] = mapped_column(String)
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # "unenrolled_contact" (default) | "distress"
    alert_type: Mapped[str] = mapped_column(String, nullable=False, default="unenrolled_contact")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    parent: Mapped["Parent"] = relationship("Parent", back_populates="alerts")
    device: Mapped["Device"] = relationship("Device", back_populates="alerts")


class HouseholdConfig(Base):
    """Per-household calibrated detection thresholds and context gates."""
    __tablename__ = "household_configs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=False, unique=True, index=True)

    t_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    t_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calibrated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    calibration_sample_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    quiet_hours_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    quiet_hours_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_radius_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=100.0)
    party_mode_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Parent-nearby rule: no detection while the child device is within
    # parent_proximity_radius_m of a parent phone (both fixes fresh and accurate).
    parent_proximity_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    parent_proximity_radius_m: Mapped[float] = mapped_column(Float, nullable=False, default=10.0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    parent: Mapped["Parent"] = relationship("Parent", foreign_keys=[parent_id])


class PairingRequest(Base):
    """Short-lived code shown on a new child device (e.g. a watch) and entered by the
    parent on their phone.  The device polls with pair_id + secret and receives its
    device token exactly once after the parent has claimed the code."""
    __tablename__ = "pairing_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code: Mapped[str] = mapped_column(String, nullable=False, index=True)
    secret_hash: Mapped[str] = mapped_column(String, nullable=False)
    installation_id: Mapped[str] = mapped_column(String, nullable=False)
    device_name: Mapped[Optional[str]] = mapped_column(String)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    claimed_parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=True)
    device_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    consumed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
