"""SQLAlchemy models for SafeEar PRD."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text, UUID, Integer, Enum, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid
import enum

Base = declarative_base()


class Parent(Base):
    __tablename__ = "parents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    google_sub = Column(String, unique=True, nullable=False, index=True)
    email = Column(String)
    display_name = Column(String)
    phone_number = Column(String)
    fcm_token = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    devices = relationship("Device", back_populates="parent", cascade="all, delete-orphan")
    speakers = relationship("EnrolledSpeaker", back_populates="parent", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="parent", cascade="all, delete-orphan")


class DeviceRole(str, enum.Enum):
    child_device = "child_device"
    parent_device = "parent_device"


class Device(Base):
    __tablename__ = "devices"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=False, index=True)
    device_name = Column(String)
    role = Column(Enum(DeviceRole), nullable=False, index=True)
    device_token = Column(String, unique=True, index=True)
    last_location_lat = Column(Float)
    last_location_lon = Column(Float)
    last_location_ts = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    parent = relationship("Parent", back_populates="devices")
    alerts = relationship("Alert", back_populates="device", cascade="all, delete-orphan")


class EnrolledSpeaker(Base):
    __tablename__ = "enrolled_speakers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=False, index=True)
    display_name = Column(String, nullable=False)
    sample_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    parent = relationship("Parent", back_populates="speakers")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("parents.id"), nullable=False, index=True)
    device_id = Column(UUID(as_uuid=True), ForeignKey("devices.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    confidence_score = Column(Float)
    audio_clip_path = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    acknowledged_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    parent = relationship("Parent", back_populates="alerts")
    device = relationship("Device", back_populates="alerts")
