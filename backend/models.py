"""Domain models for CityCare Clinic."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Appointment:
    id: str
    name: str
    phone: str
    date: str
    time: str
    service: str


@dataclass
class Bill:
    phone: str
    amount: str
    due_date: str
    status: str = "Due"


@dataclass
class Prescription:
    id: str
    patient_name: str
    phone: str
    medication: str
    refills_remaining: int
    prescribing_doctor: str
    last_filled: Optional[str] = None
    status: str = "Active"


@dataclass
class RefillRecord:
    refill_id: str
    patient_name: str
    medication: str
    pharmacy: str
    refills_remaining: int
    status: str
    estimated_ready: str
