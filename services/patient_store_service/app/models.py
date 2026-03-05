import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    mrn: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    first_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dob: Mapped[str | None] = mapped_column(String(32), nullable=True)
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    documents: Mapped[list["Document"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)

    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    patient: Mapped["Patient"] = relationship(back_populates="documents")
    entities: Mapped[list["NEREntity"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class NEREntity(Base):
    __tablename__ = "ner_entities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)

    label: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    start_char: Mapped[int | None] = mapped_column(nullable=True)
    end_char: Mapped[int | None] = mapped_column(nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)

    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped["Document"] = relationship(back_populates="entities")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)

    # must match tokenization prefixes: enc/obs/med/proc/cp/imm/alg
    source: Mapped[str] = mapped_column(String(16), index=True)

    event_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    value: Mapped[str | None] = mapped_column(String(128), nullable=True)
    units: Mapped[str | None] = mapped_column(String(64), nullable=True)
    type: Mapped[str | None] = mapped_column(String(64), nullable=True)

    reason_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reason_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    patient: Mapped["Patient"] = relationship(back_populates="events")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    actor: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(64))

    patient_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    document_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    route: Mapped[str | None] = mapped_column(String(256), nullable=True)
    method: Mapped[str | None] = mapped_column(String(16), nullable=True)

    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)


class Cohort(Base):
    __tablename__ = "cohorts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), index=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    filters: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
