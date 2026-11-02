"""CityCare Clinic API — FastAPI application entry point."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

from backend.database import (
    SUPPORTED_PHARMACIES,
    get_connection,
    initialize_database,
)
from backend.schemas import (
    AppointmentCreate,
    AppointmentUpdate,
    PrescriptionRefillRequest,
)
from backend.services.appointment_service import AppointmentService
from backend.services.calendar_service import CalendarService

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env.local")
load_dotenv(ROOT_DIR / ".env")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

app = FastAPI(
    title="CityCare Clinic API",
    description="Backend API for CityCare Clinic Voice AI Agent",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    """Create required tables before serving requests."""
    initialize_database()


@app.get("/")
async def root():
    return {"message": "CityCare Clinic API is running.", "status": "ok"}


@app.get("/token")
async def get_token(
    room: str = Query(default="citycare-clinic", min_length=1, max_length=100),
    name: str = Query(default="Guest", min_length=1, max_length=50),
):
    if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials are not configured.",
        )

    participant_name = name.strip()
    room_name = room.strip()

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(participant_name)
        .with_name(participant_name)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .with_room_config(
            api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name="citycare")])
        )
    )

    token_jwt = token.to_jwt()

    return {
        "serverUrl": LIVEKIT_URL,
        "url": LIVEKIT_URL,
        "participantName": participant_name,
        "roomName": room_name,
        "participantToken": token_jwt,
        "token": token_jwt,
    }


@app.get("/slots")
def get_slots(date: str):
    return CalendarService.get_available_slots(date)


@app.post("/appointments")
def create_appointment(appointment: AppointmentCreate):
    return AppointmentService.create_appointment(appointment)


@app.patch("/appointments/{appointment_id}")
def update_appointment(
    appointment_id: str,
    appointment: AppointmentUpdate,
):
    return AppointmentService.update_appointment(appointment_id, appointment)


@app.delete("/appointments/{appointment_id}")
def delete_appointment(appointment_id: str):
    return AppointmentService.delete_appointment(appointment_id)


@app.get("/appointments")
def get_appointments(phone: str = Query(..., min_length=3)):
    return AppointmentService.get_appointments_by_phone(phone)


@app.get("/appointments/{appointment_id}")
def get_appointment(appointment_id: str):
    return AppointmentService.get_appointment_by_id(appointment_id)


@app.get("/bills")
def get_bill(phone: str = Query(..., min_length=3)):
    normalized_phone = AppointmentService.validate_phone(phone)

    with get_connection() as connection:
        bill = connection.execute(
            """
            SELECT phone, amount, due_date, status
            FROM bills
            WHERE phone = %s
            """,
            (normalized_phone,),
        ).fetchone()

    if bill is None:
        raise HTTPException(
            status_code=404,
            detail="No billing record was found for this phone number.",
        )

    return dict(bill)


@app.get("/prescriptions")
def get_prescriptions(phone: str = Query(..., min_length=3)):
    normalized_phone = AppointmentService.validate_phone(phone)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, patient_name, phone, medication,
                   refills_remaining, prescribing_doctor,
                   last_filled, status
            FROM prescriptions
            WHERE phone = %s
            ORDER BY medication
            """,
            (normalized_phone,),
        ).fetchall()

    return [dict(row) for row in rows]


@app.get("/prescriptions/pharmacies")
def get_pharmacies():
    return SUPPORTED_PHARMACIES


@app.post("/prescriptions/refill")
def refill_prescription(refill: PrescriptionRefillRequest):
    normalized_phone = AppointmentService.validate_phone(refill.phone)
    patient_name = refill.name.strip()
    medication = refill.medication.strip()
    medication_lower = medication.casefold()
    pharmacy = (
        refill.pharmacy.strip()
        if refill.pharmacy and refill.pharmacy.strip()
        else "CityCare In-House Pharmacy"
    )

    if pharmacy not in SUPPORTED_PHARMACIES:
        raise HTTPException(
            status_code=400,
            detail="Please choose a supported pharmacy.",
        )

    with get_connection() as connection:
        prescriptions = connection.execute(
            """
            SELECT id, patient_name, phone, medication,
                   refills_remaining, prescribing_doctor, status
            FROM prescriptions
            WHERE phone = %s
            FOR UPDATE
            """,
            (normalized_phone,),
        ).fetchall()

        matched = next(
            (
                row
                for row in prescriptions
                if row["patient_name"].strip().casefold() == patient_name.casefold()
                and (
                    medication_lower in row["medication"].casefold()
                    or row["medication"].casefold() in medication_lower
                )
            ),
            None,
        )

        if matched is None:
            raise HTTPException(
                status_code=404,
                detail="No matching prescription was found for this patient.",
            )

        if matched["status"].casefold() != "active":
            raise HTTPException(
                status_code=400,
                detail="This prescription is not active. Please contact the clinic.",
            )

        if matched["refills_remaining"] <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No refills remain for {matched['medication']}. "
                    f"Please contact {matched['prescribing_doctor']} "
                    "for review."
                ),
            )

        remaining = matched["refills_remaining"] - 1
        refill_id = f"refill-{uuid4().hex[:12]}"
        status = "Submitted for pharmacy processing"
        estimated_ready = "Awaiting pharmacy confirmation"

        connection.execute(
            """
            UPDATE prescriptions
            SET refills_remaining = %s
            WHERE id = %s
            """,
            (remaining, matched["id"]),
        )

        connection.execute(
            """
            INSERT INTO refill_records (
                refill_id, patient_name, phone, prescription_id,
                medication, pharmacy, refills_remaining, status,
                estimated_ready
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                refill_id,
                matched["patient_name"],
                normalized_phone,
                matched["id"],
                matched["medication"],
                pharmacy,
                remaining,
                status,
                estimated_ready,
            ),
        )

    return {
        "refill_id": refill_id,
        "patient_name": matched["patient_name"],
        "medication": matched["medication"],
        "pharmacy": pharmacy,
        "refills_remaining": remaining,
        "status": status,
        "estimated_ready": estimated_ready,
    }


@app.get("/slow")
async def slow_system():
    await asyncio.sleep(3)
    return {
        "status": "ok",
        "message": "Slow system recovered successfully.",
    }


@app.get("/broken")
async def broken_system():
    raise HTTPException(status_code=503, detail="Mock system failure")
