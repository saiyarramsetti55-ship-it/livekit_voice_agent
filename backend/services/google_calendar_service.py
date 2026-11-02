"""Google Calendar service for managing appointments and availability."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from backend.database import DEFAULT_SLOTS

ROOT_DIR = Path(__file__).resolve().parents[2]
CALENDAR_SCOPE = ["https://www.googleapis.com/auth/calendar"]
CLINIC_TIMEZONE = ZoneInfo("Asia/Kolkata")
DEFAULT_APPOINTMENT_DURATION_MINUTES = 30

logger = logging.getLogger("citycare-backend")


class GoogleCalendarService:
    """Create, update, delete, and query CityCare appointment events."""

    @staticmethod
    def _get_calendar_service():
        """Authenticate using a Google Cloud service-account JSON file."""
        credentials_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_FILE") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        if not credentials_path:
            raise RuntimeError(
                "Set GOOGLE_CALENDAR_CREDENTIALS_FILE in .env.local "
                "to the service-account JSON file path."
            )

        credentials_file = Path(credentials_path)

        if not credentials_file.is_absolute():
            credentials_file = ROOT_DIR / credentials_file

        if not credentials_file.is_file():
            raise RuntimeError(
                "Google Calendar service-account JSON file was not found. "
                "Check GOOGLE_CALENDAR_CREDENTIALS_FILE."
            )

        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_file),
            scopes=CALENDAR_SCOPE,
        )

        return build(
            "calendar",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )

    @staticmethod
    def _get_calendar_id() -> str:
        """Read the target calendar ID from the environment."""
        calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "").strip()

        if not calendar_id:
            raise RuntimeError("GOOGLE_CALENDAR_ID is missing from .env.local.")

        return calendar_id

    @staticmethod
    def _build_event(
        name: str,
        phone: str,
        appointment_date: str,
        appointment_time: str,
        service: str,
    ) -> dict:
        """Build a Google Calendar event in the clinic's local timezone."""
        start_naive = datetime.strptime(
            f"{appointment_date} {appointment_time}",
            "%Y-%m-%d %H:%M",
        )

        start_time = start_naive.replace(tzinfo=CLINIC_TIMEZONE)
        end_time = start_time + timedelta(minutes=DEFAULT_APPOINTMENT_DURATION_MINUTES)

        return {
            "summary": f"CityCare Clinic - {service}",
            "description": (
                f"Patient: {name}\n"
                f"Phone: {phone}\n"
                f"Service: {service}\n"
                "Created by the CityCare Clinic booking system."
            ),
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "Asia/Kolkata",
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": "Asia/Kolkata",
            },
            "extendedProperties": {
                "private": {
                    "clinic": "CityCare Clinic",
                    "patientPhone": phone,
                }
            },
        }

    @classmethod
    def get_booked_slots_for_date(cls, date_str: str) -> set[str]:
        """
        Query Google Calendar for events occurring on the given date (YYYY-MM-DD)
        and return the set of DEFAULT_SLOTS ("HH:MM") that overlap with existing events.
        """
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return set()

        start_of_day = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            0,
            0,
            0,
            tzinfo=CLINIC_TIMEZONE,
        )
        end_of_day = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            23,
            59,
            59,
            999999,
            tzinfo=CLINIC_TIMEZONE,
        )

        try:
            calendar_service = cls._get_calendar_service()
            calendar_id = cls._get_calendar_id()
            events_result = (
                calendar_service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=start_of_day.isoformat(),
                    timeMax=end_of_day.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            items = events_result.get("items", [])
        except Exception as exc:
            logger.warning(
                f"Failed to query Google Calendar events for {date_str}: {exc}"
            )
            return set()

        booked_slots: set[str] = set()

        for event in items:
            if event.get("status") == "cancelled":
                continue

            start_info = event.get("start", {})
            end_info = event.get("end", {})

            # All-day event blocks all slots
            if "date" in start_info:
                return set(DEFAULT_SLOTS)

            start_str = start_info.get("dateTime")
            end_str = end_info.get("dateTime")
            if not start_str or not end_str:
                continue

            try:
                ev_start = datetime.fromisoformat(start_str)
                ev_end = datetime.fromisoformat(end_str)
            except ValueError:
                continue

            for slot in DEFAULT_SLOTS:
                slot_time = datetime.strptime(
                    f"{date_str} {slot}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=CLINIC_TIMEZONE)
                slot_end = slot_time + timedelta(
                    minutes=DEFAULT_APPOINTMENT_DURATION_MINUTES
                )

                if max(slot_time, ev_start) < min(slot_end, ev_end):
                    booked_slots.add(slot)

        return booked_slots

    @classmethod
    def create_event(
        cls,
        name: str,
        phone: str,
        appointment_date: str,
        appointment_time: str,
        service: str,
    ) -> str:
        """Create an event and return its Google Calendar event ID."""
        calendar_service = cls._get_calendar_service()
        calendar_id = cls._get_calendar_id()

        event_body = cls._build_event(
            name=name,
            phone=phone,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            service=service,
        )

        try:
            event = (
                calendar_service.events()
                .insert(
                    calendarId=calendar_id,
                    body=event_body,
                )
                .execute()
            )
        except HttpError as exc:
            raise RuntimeError(
                f"Google Calendar could not create the event: {exc}"
            ) from exc

        event_id = event.get("id")

        if not event_id:
            raise RuntimeError("Google Calendar created no usable event ID.")

        return event_id

    @classmethod
    def update_event(
        cls,
        event_id: str,
        name: str,
        phone: str,
        appointment_date: str,
        appointment_time: str,
        service: str,
    ) -> dict:
        """Update an existing Google Calendar event."""
        calendar_service = cls._get_calendar_service()
        calendar_id = cls._get_calendar_id()

        event_body = cls._build_event(
            name=name,
            phone=phone,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            service=service,
        )

        try:
            event = (
                calendar_service.events()
                .patch(
                    calendarId=calendar_id,
                    eventId=event_id,
                    body=event_body,
                )
                .execute()
            )
        except HttpError as exc:
            raise RuntimeError(
                f"Google Calendar could not update the event: {exc}"
            ) from exc

        return {
            "event_id": event.get("id", event_id),
            "html_link": event.get("htmlLink"),
        }

    @classmethod
    def delete_event(cls, event_id: str) -> None:
        """Delete an existing Google Calendar event."""
        calendar_service = cls._get_calendar_service()
        calendar_id = cls._get_calendar_id()

        try:
            (
                calendar_service.events()
                .delete(
                    calendarId=calendar_id,
                    eventId=event_id,
                )
                .execute()
            )
        except HttpError as exc:
            # A 404 means the event is already absent.
            if exc.resp.status == 404:
                return

            raise RuntimeError(
                f"Google Calendar could not delete the event: {exc}"
            ) from exc
