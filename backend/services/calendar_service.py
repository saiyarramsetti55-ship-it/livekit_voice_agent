"""Appointment slot calculations backed by PostgreSQL and Google Calendar."""

import logging
from datetime import date as date_type
from datetime import datetime

from backend.database import DEFAULT_SLOTS, get_connection
from backend.services.google_calendar_service import (
    CLINIC_TIMEZONE,
    GoogleCalendarService,
)

logger = logging.getLogger("citycare-backend")


def validate_date(date: str) -> str:
    """Validate and normalize a date in YYYY-MM-DD format."""
    try:
        return date_type.fromisoformat(date.strip()).isoformat()
    except (ValueError, AttributeError):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Date must use YYYY-MM-DD format.",
        ) from None


def validate_time(time: str) -> str:
    """Validate and normalize a time in HH:MM format."""
    from fastapi import HTTPException

    try:
        return datetime.strptime(time.strip(), "%H:%M").strftime("%H:%M")
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Time must use HH:MM format.",
        ) from None


class CalendarService:
    """Read appointment availability from persistent records and Google Calendar."""

    @staticmethod
    def get_available_slots(date: str) -> list[str]:
        clean_date = validate_date(date)
        d = date_type.fromisoformat(clean_date)

        now_kolkata = datetime.now(CLINIC_TIMEZONE)
        today_kolkata = now_kolkata.date()

        # Past dates have no available slots
        if d < today_kolkata:
            return []

        # Clinic closed on Sundays
        if d.weekday() == 6:
            return []

        # Saturday clinic hours: 9:00 AM to 1:00 PM
        if d.weekday() == 5:
            applicable_slots = [s for s in DEFAULT_SLOTS if int(s.split(":")[0]) < 13]
        else:
            applicable_slots = DEFAULT_SLOTS.copy()

        # For today, filter out slots that have already passed
        if d == today_kolkata:
            current_time_str = now_kolkata.strftime("%H:%M")
            applicable_slots = [s for s in applicable_slots if s > current_time_str]

        # Query booked slots from PostgreSQL
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT time FROM appointments WHERE date = %s",
                (clean_date,),
            ).fetchall()

        pg_booked = {row["time"] for row in rows}

        # Query booked slots from Google Calendar
        gcal_booked = GoogleCalendarService.get_booked_slots_for_date(clean_date)

        all_booked = pg_booked | gcal_booked

        available = [slot for slot in applicable_slots if slot not in all_booked]
        logger.info(
            f"[AVAILABILITY] Date: {clean_date} | Applicable: {applicable_slots} | "
            f"PG Booked: {sorted(pg_booked)} | GCal Booked: {sorted(gcal_booked)} | Available: {available}"
        )
        return available

    @staticmethod
    def is_slot_available(date: str, time: str) -> bool:
        clean_date = validate_date(date)
        clean_time = validate_time(time)
        return clean_time in CalendarService.get_available_slots(clean_date)

    @staticmethod
    def get_default_slots() -> list[str]:
        return DEFAULT_SLOTS.copy()
