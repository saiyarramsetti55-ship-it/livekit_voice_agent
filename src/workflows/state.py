from dataclasses import dataclass, field


@dataclass
class CallerData:
    """
    Stores information collected during the current call.

    Identity must be collected in this order:

        name
        phone
        date_of_birth
        verification
    """

    name: str | None = None
    phone: str | None = None
    date_of_birth: str | None = None

    verified: bool = False

    # Explicit identity state and confirmations.
    verification_state: str = "WAITING_FOR_NAME"
    pending_name: str | None = None
    name_collected: bool = False
    name_confirmed: bool = False

    phone_collected: bool = False
    pending_phone: str | None = None
    partial_phone: str | None = None
    phone_confirmed: bool = False

    pending_dob: str | None = None
    dob_collected: bool = False
    dob_confirmed: bool = False

    # Booking state.
    appointment_intent: str | None = None  # CHECK, BOOK, RESCHEDULE, CANCEL
    booking_active: bool = False
    booking_withdrawn: bool = False
    booking_confirmed: bool = False
    pending_service: str | None = None
    pending_date: str | None = None
    pending_time: str | None = None

    # Appointment selected for cancellation/change.
    selected_appointment_id: str | None = None
    appointment_ids: list[str] | None = None
    appointment_found: bool = False
    appointment_confirmed: bool = False

    # Capstone: Prescription refill state.
    prescription_active: bool = False
    prescription_medication: str | None = None
    prescription_pharmacy: str | None = None
    refilled_medications: list[str] = field(default_factory=list)

    # Verification retry tracking
    verification_attempts: int = 0
