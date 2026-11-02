"""Pydantic request and response schemas for CityCare Clinic."""

from pydantic import BaseModel, Field


class AppointmentCreate(BaseModel):
    name: str = Field(..., min_length=2)
    phone: str = Field(..., min_length=3)
    date: str = Field(..., min_length=8, description="YYYY-MM-DD")
    time: str = Field(..., min_length=4, description="HH:MM")
    service: str = Field(..., min_length=2)


class AppointmentUpdate(BaseModel):
    date: str = Field(..., min_length=8, description="YYYY-MM-DD")
    time: str = Field(..., min_length=4, description="HH:MM")


class AppointmentResponse(BaseModel):
    id: str
    name: str
    phone: str
    date: str
    time: str
    service: str


class AppointmentDeleteResponse(BaseModel):
    message: str
    appointment: AppointmentResponse


class BillResponse(BaseModel):
    phone: str
    amount: str
    due_date: str
    status: str


class PrescriptionRefillRequest(BaseModel):
    name: str = Field(..., min_length=2)
    phone: str = Field(..., min_length=3)
    medication: str = Field(..., min_length=2)
    pharmacy: str = Field(
        default="CityCare In-House Pharmacy",
        min_length=2,
    )


class RefillResponse(BaseModel):
    refill_id: str
    patient_name: str
    medication: str
    pharmacy: str
    refills_remaining: int
    status: str
    estimated_ready: str


class TokenResponse(BaseModel):
    serverUrl: str
    url: str
    participantName: str
    roomName: str
    participantToken: str
    token: str
