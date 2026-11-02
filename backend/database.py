"""PostgreSQL database connection and schema management for CityCare Clinic."""

import contextlib
import os
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

CLINIC_SERVICES = {
    "general check-up",
    "blood tests",
    "vaccines",
    "children's doctor",
}

DEFAULT_SLOTS = ["09:00", "10:00", "11:30", "14:00", "15:30"]

SUPPORTED_PHARMACIES = [
    "CityCare In-House Pharmacy",
    "Walgreens Downtown",
    "CVS Main Street",
    "Apollo Pharmacy",
]

# Thread-safe connection pool for reusing active PostgreSQL TLS connections
_POOL_SIZE = 5
_connection_pool: queue.Queue[psycopg.Connection] = queue.Queue(maxsize=_POOL_SIZE)
_pool_lock = threading.Lock()


def _is_connection_healthy(conn: psycopg.Connection) -> bool:
    """Check if connection is open and ready for a query."""
    try:
        if hasattr(conn, "closed") and isinstance(conn.closed, bool) and conn.closed:
            return False
        return not (
            hasattr(conn, "broken") and isinstance(conn.broken, bool) and conn.broken
        )
    except Exception:
        return False


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Provide a pooled PostgreSQL connection and manage its transaction."""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is missing. Configure it in your environment.")

    conn: psycopg.Connection | None = None

    # Try to reuse an existing healthy connection from the pool
    while not _connection_pool.empty():
        try:
            candidate = _connection_pool.get_nowait()
            if _is_connection_healthy(candidate):
                conn = candidate
                break
            else:
                with contextlib.suppress(Exception):
                    candidate.close()
        except queue.Empty:
            break

    if conn is None:
        conn = psycopg.connect(database_url, row_factory=dict_row)

    try:
        yield conn
        conn.commit()
    except Exception:
        with contextlib.suppress(Exception):
            conn.rollback()
        raise
    finally:
        # Return connection to pool if it remains healthy
        if _is_connection_healthy(conn):
            try:
                _connection_pool.put_nowait(conn)
            except queue.Full:
                with contextlib.suppress(Exception):
                    conn.close()
        else:
            with contextlib.suppress(Exception):
                conn.close()


def initialize_database() -> None:
    """Create required tables and safely add columns to existing tables."""

    statements = [
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            service TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            google_calendar_event_id TEXT,
            calendar_sync_status TEXT NOT NULL DEFAULT 'not_synced',
            CONSTRAINT unique_appointment_slot UNIQUE (date, time)
        )
        """,
        # Migrations for existing PostgreSQL databases.
        """
        ALTER TABLE appointments
        ADD COLUMN IF NOT EXISTS google_calendar_event_id TEXT
        """,
        """
        ALTER TABLE appointments
        ADD COLUMN IF NOT EXISTS calendar_sync_status TEXT
        NOT NULL DEFAULT 'not_synced'
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_appointments_phone
        ON appointments (phone)
        """,
        """
        CREATE TABLE IF NOT EXISTS bills (
            phone TEXT PRIMARY KEY,
            amount TEXT NOT NULL,
            due_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Due'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prescriptions (
            id TEXT PRIMARY KEY,
            patient_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            medication TEXT NOT NULL,
            refills_remaining INTEGER NOT NULL DEFAULT 0
                CHECK (refills_remaining >= 0),
            prescribing_doctor TEXT NOT NULL,
            last_filled TEXT,
            status TEXT NOT NULL DEFAULT 'Active'
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_prescriptions_phone
        ON prescriptions (phone)
        """,
        """
        CREATE TABLE IF NOT EXISTS refill_records (
            refill_id TEXT PRIMARY KEY,
            patient_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            prescription_id TEXT NOT NULL
                REFERENCES prescriptions(id),
            medication TEXT NOT NULL,
            pharmacy TEXT NOT NULL,
            refills_remaining INTEGER NOT NULL,
            status TEXT NOT NULL,
            estimated_ready TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    ]

    with get_connection() as connection:
        for statement in statements:
            connection.execute(statement)
