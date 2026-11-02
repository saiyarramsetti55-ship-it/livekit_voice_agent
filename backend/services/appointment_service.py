"""Persistent appointment management for CityCare Clinic."""

import contextlib
import logging
import re
from datetime import datetime
from uuid import uuid4

import psycopg
from fastapi import HTTPException

from backend import database
from backend.database import CLINIC_SERVICES
from backend.schemas import AppointmentCreate, AppointmentUpdate
from backend.services.calendar_service import (
    CalendarService,
    validate_date,
    validate_time,
)
from backend.services.google_calendar_service import (
    CLINIC_TIMEZONE,
    GoogleCalendarService,
)

logger = logging.getLogger("citycare-backend")


class AppointmentService:
    """Create, retrieve, reschedule, and cancel appointments."""

    @staticmethod
    def normalize_phone(phone: str) -> str:
        return re.sub(r"\D", "", str(phone))

    @staticmethod
    def validate_phone(phone: str) -> str:
        normalized = AppointmentService.normalize_phone(phone)

        if len(normalized) == 14 and normalized.startswith("0091"):
            normalized = normalized[4:]
        elif len(normalized) == 12 and normalized.startswith("91"):
            normalized = normalized[2:]
        elif (len(normalized) == 11 and normalized.startswith("0")) or (
            len(normalized) == 11 and normalized.startswith("1")
        ):
            normalized = normalized[1:]

        if len(normalized) != 10:
            raise HTTPException(
                status_code=400,
                detail="Please provide a valid 10-digit phone number.",
            )

        return normalized

    @staticmethod
    def validate_service(service: str) -> str:
        cleaned = service.strip().lower()

        aliases = {
            "check-up": "general check-up",
            "checkup": "general check-up",
            "general checkup": "general check-up",
            "blood test": "blood tests",
            "child doctor": "children's doctor",
            "pediatrician": "children's doctor",
            "paediatrician": "children's doctor",
        }

        cleaned = aliases.get(cleaned, cleaned)

        if cleaned not in CLINIC_SERVICES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported service. Available services: "
                    "general check-up, blood tests, vaccines, "
                    "children's doctor."
                ),
            )

        return cleaned

    @staticmethod
    def _mark_calendar_status(appointment_id: str, status: str) -> None:
        """Update calendar synchronization status in PostgreSQL."""
        with database.get_connection() as connection:
            connection.execute(
                """
                UPDATE appointments
                SET calendar_sync_status = %s
                WHERE id = %s
                """,
                (status, appointment_id),
            )

    @staticmethod
    def create_appointment(appointment: AppointmentCreate) -> dict:
        """Reserve a slot, create its calendar event, and save the event ID."""
        phone = AppointmentService.validate_phone(appointment.phone)
        service = AppointmentService.validate_service(appointment.service)
        date = validate_date(appointment.date)
        time = validate_time(appointment.time)
        name = appointment.name.strip()

        if not name:
            raise HTTPException(
                status_code=400,
                detail="Patient name is required.",
            )

        today_kolkata = datetime.now(CLINIC_TIMEZONE).date().isoformat()
        if date < today_kolkata:
            raise HTTPException(
                status_code=400,
                detail="Appointments cannot be booked in the past.",
            )

        # Check existing appointment in PostgreSQL for idempotency and conflict
        with database.get_connection() as connection:
            existing = connection.execute(
                "SELECT * FROM appointments WHERE date = %s AND time = %s",
                (date, time),
            ).fetchone()

        if existing:
            if existing["phone"] == phone:
                logger.info(
                    f"[IDEMPOTENCY] Appointment for {phone} on {date} at {time} already exists (ID: {existing['id']})."
                )
                if existing.get("calendar_sync_status") != "synced" or not existing.get(
                    "google_calendar_event_id"
                ):
                    try:
                        event_id = GoogleCalendarService.create_event(
                            name=name,
                            phone=phone,
                            appointment_date=date,
                            appointment_time=time,
                            service=service,
                        )
                        with database.get_connection() as connection:
                            connection.execute(
                                """
                                UPDATE appointments
                                SET google_calendar_event_id = %s,
                                    calendar_sync_status = 'synced'
                                WHERE id = %s
                                """,
                                (event_id, existing["id"]),
                            )
                        existing["google_calendar_event_id"] = event_id
                        existing["calendar_sync_status"] = "synced"
                    except Exception as gcal_err:
                        logger.warning(
                            f"Could not sync existing appointment to Google Calendar: {gcal_err}"
                        )
                return dict(existing)
            else:
                logger.warning(
                    f"[SLOT UNAVAILABLE] Slot {date} {time} already booked by another patient in PostgreSQL."
                )
                raise HTTPException(
                    status_code=409,
                    detail="Requested slot is unavailable. Please choose another available time.",
                )

        available_slots = CalendarService.get_available_slots(date)
        if time not in available_slots:
            logger.warning(
                f"[SLOT UNAVAILABLE] Requested slot {date} {time} is not available. Available: {available_slots}"
            )
            raise HTTPException(
                status_code=409,
                detail="Requested slot is unavailable. Please choose another available time.",
            )

        item = {
            "id": str(uuid4()),
            "name": name,
            "phone": phone,
            "date": date,
            "time": time,
            "service": service,
        }

        # First reserve the slot in PostgreSQL.
        try:
            with database.get_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO appointments
                        (id, name, phone, date, time, service,
                         calendar_sync_status)
                    VALUES
                        (%(id)s, %(name)s, %(phone)s, %(date)s,
                         %(time)s, %(service)s, 'pending')
                    """,
                    item,
                )
        except psycopg.errors.UniqueViolation:
            raise HTTPException(
                status_code=409,
                detail=(
                    "That appointment slot has just been booked. Please choose another."
                ),
            ) from None

        # Then create the corresponding Google Calendar event.
        try:
            event_id = GoogleCalendarService.create_event(
                name=name,
                phone=phone,
                appointment_date=date,
                appointment_time=time,
                service=service,
            )

            with database.get_connection() as connection:
                connection.execute(
                    """
                    UPDATE appointments
                    SET google_calendar_event_id = %s,
                        calendar_sync_status = 'synced'
                    WHERE id = %s
                    """,
                    (event_id, item["id"]),
                )

        except Exception as exc:
            with contextlib.suppress(Exception):
                AppointmentService._mark_calendar_status(item["id"], "failed")

            raise HTTPException(
                status_code=503,
                detail=(
                    "The appointment slot was reserved in PostgreSQL, "
                    "but Google Calendar synchronization failed. "
                    f"Appointment ID: {item['id']}. "
                    "Do not retry as a new booking; reconcile this "
                    "appointment before confirming it. Error: {exc}"
                ).replace("{exc}", str(exc)),
            ) from exc

        return {
            **item,
            "google_calendar_event_id": event_id,
            "calendar_sync_status": "synced",
        }

    @staticmethod
    def update_appointment(
        appointment_id: str,
        update: AppointmentUpdate,
    ) -> dict:
        """Reschedule an appointment and update its calendar event."""
        date = validate_date(update.date)
        time = validate_time(update.time)

        try:
            with database.get_connection() as connection:
                current = connection.execute(
                    """
                    SELECT *
                    FROM appointments
                    WHERE id = %s
                    FOR UPDATE
                    """,
                    (appointment_id,),
                ).fetchone()

                if current is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Appointment not found.",
                    )

                today_kolkata = datetime.now(CLINIC_TIMEZONE).date().isoformat()
                if date < today_kolkata:
                    raise HTTPException(
                        status_code=400,
                        detail="Appointments cannot be rescheduled into the past.",
                    )

                if date != current["date"] or time != current["time"]:
                    booked = connection.execute(
                        """
                        SELECT 1
                        FROM appointments
                        WHERE date = %s
                          AND time = %s
                          AND id <> %s
                        """,
                        (date, time, appointment_id),
                    ).fetchone()

                    gcal_booked = GoogleCalendarService.get_booked_slots_for_date(date)

                    if (
                        time not in database.DEFAULT_SLOTS
                        or booked
                        or time in gcal_booked
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail="Requested slot is unavailable.",
                        )

                updated = connection.execute(
                    """
                    UPDATE appointments
                    SET date = %s,
                        time = %s,
                        calendar_sync_status = 'pending'
                    WHERE id = %s
                    RETURNING *
                    """,
                    (date, time, appointment_id),
                ).fetchone()

        except psycopg.errors.UniqueViolation:
            raise HTTPException(
                status_code=409,
                detail=(
                    "That appointment slot has just been booked. Please choose another."
                ),
            ) from None

        # Synchronize the already-updated appointment with Calendar.
        try:
            event_id = current.get("google_calendar_event_id")

            if event_id:
                result = GoogleCalendarService.update_event(
                    event_id=event_id,
                    name=updated["name"],
                    phone=updated["phone"],
                    appointment_date=date,
                    appointment_time=time,
                    service=updated["service"],
                )
                event_id = result["event_id"]
            else:
                event_id = GoogleCalendarService.create_event(
                    name=updated["name"],
                    phone=updated["phone"],
                    appointment_date=date,
                    appointment_time=time,
                    service=updated["service"],
                )

            with database.get_connection() as connection:
                connection.execute(
                    """
                    UPDATE appointments
                    SET google_calendar_event_id = %s,
                        calendar_sync_status = 'synced'
                    WHERE id = %s
                    """,
                    (event_id, appointment_id),
                )

        except Exception as exc:
            with contextlib.suppress(Exception):
                AppointmentService._mark_calendar_status(appointment_id, "failed")

            raise HTTPException(
                status_code=503,
                detail=(
                    "The appointment was updated in PostgreSQL, "
                    "but Google Calendar synchronization failed. "
                    f"Appointment ID: {appointment_id}. "
                    "Do not confirm the reschedule until synchronization "
                    f"is resolved. Error: {exc}"
                ),
            ) from exc

        return {
            "id": appointment_id,
            "name": updated["name"],
            "phone": updated["phone"],
            "date": date,
            "time": time,
            "service": updated["service"],
            "google_calendar_event_id": event_id,
            "calendar_sync_status": "synced",
        }

    @staticmethod
    def delete_appointment(appointment_id: str) -> dict:
        """Cancel an appointment and remove its Google Calendar event."""
        with database.get_connection() as connection:
            current = connection.execute(
                """
                SELECT *
                FROM appointments
                WHERE id = %s
                """,
                (appointment_id,),
            ).fetchone()

        if current is None:
            raise HTTPException(
                status_code=404,
                detail="Appointment not found.",
            )

        event_id = current.get("google_calendar_event_id")

        # Remove the calendar event first. If this fails, retain the
        # appointment row so the cancellation can be retried safely.
        if event_id:
            try:
                GoogleCalendarService.delete_event(event_id)
            except Exception as exc:
                with contextlib.suppress(Exception):
                    AppointmentService._mark_calendar_status(appointment_id, "failed")

                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The appointment was not removed from PostgreSQL "
                        "because Google Calendar cancellation failed. "
                        f"Appointment ID: {appointment_id}. Error: {exc}"
                    ),
                ) from exc

        with database.get_connection() as connection:
            deleted = connection.execute(
                """
                DELETE FROM appointments
                WHERE id = %s
                RETURNING id, name, phone, date, time, service
                """,
                (appointment_id,),
            ).fetchone()

        if deleted is None:
            raise HTTPException(
                status_code=404,
                detail="Appointment not found.",
            )

        return {
            "message": "Appointment cancelled successfully.",
            "appointment": dict(deleted),
        }

    @staticmethod
    def get_appointments_by_phone(phone: str) -> list[dict]:
        """Return appointments associated with a validated phone number."""
        normalized_phone = AppointmentService.validate_phone(phone)

        with database.get_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, name, phone, date, time, service,
                       google_calendar_event_id, calendar_sync_status
                FROM appointments
                WHERE phone = %s
                ORDER BY date, time
                """,
                (normalized_phone,),
            ).fetchall()

        return [dict(row) for row in rows]

    @staticmethod
    def get_appointment_by_id(appointment_id: str) -> dict:
        """Return a single appointment."""
        with database.get_connection() as connection:
            item = connection.execute(
                """
                SELECT id, name, phone, date, time, service,
                       google_calendar_event_id, calendar_sync_status
                FROM appointments
                WHERE id = %s
                """,
                (appointment_id,),
            ).fetchone()

        if item is None:
            raise HTTPException(
                status_code=404,
                detail="Appointment not found.",
            )

        return dict(item)
