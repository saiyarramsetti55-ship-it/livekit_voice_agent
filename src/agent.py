import asyncio
import json
import logging
import os
import re
import sys
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

from livekit.agents import llm
from livekit.plugins import cartesia, deepgram

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import httpx  # noqa: E402
import openai  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    RunContext,
    STTContextOptions,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
    stt,
)
from livekit.agents.llm import ChatMessage  # noqa: E402
from livekit.agents.llm.fallback_adapter import (  # noqa: E402
    AvailabilityChangedEvent,
    FallbackLLMStream,
)
from livekit.plugins import ai_coustics, google, groq  # noqa: E402

from config.prompts import (  # noqa: E402
    BILLING_PROMPT,
    BOOKING_PROMPT,
    CLINIC_PROMPT,
    PRESCRIPTION_PROMPT,
    RECEPTION_PROMPT,
)

# -- Config imports (single source of truth) --
from config.settings import (  # noqa: E402
    API,
    GOOGLE_PRIMARY_MODEL,
    GROQ_FALLBACK_MODEL,
    HTTP_TIMEOUT,
    LLM_ATTEMPT_TIMEOUT,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_MAX_RETRIES,
    LLM_MODE,
    MAX_HISTORY_ITEMS,
    MAX_TOOL_STEPS,
    RAG_TIMEOUT,
    TOP_K,
)
from config.voice_settings import VOICE_SETTINGS  # noqa: E402

try:
    from .rag import knowledge_base as rag  # moved to src/rag/knowledge_base.py
except ImportError:
    from src.rag import knowledge_base as rag

load_dotenv(Path(__file__).resolve().parent / ".env.local")
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env.local")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if hasattr(sys.stdout, "reconfigure"):
    with suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    with suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("citycare-agent")


# ============================================================
# CONFIGURATION  (values live in config/settings.py)
# ============================================================

# API, TOP_K, LLM_MODE, GROQ_MODEL are imported from config.settings above.
# TIMING_FILE is agent-local (runtime artefact, not a config concern).
TIMING_FILE = Path(__file__).resolve().parent / "tool_times.csv"

load_dotenv(".env.local")


# ============================================================
# SHARED CALLER DATA
# ============================================================


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


# ============================================================
# CLINIC PROMPT & AGENT PROMPTS
# (authoritative copies live in config/prompts.py and are imported above)
# ============================================================


# ============================================================
# TOOL TIMING
# ============================================================


def _append_tool_time(tool_name: str, ms: float, stamp: str) -> None:
    try:
        new_file = not TIMING_FILE.exists()
        with open(TIMING_FILE, "a", encoding="utf-8") as f:
            if new_file:
                f.write("time,tool,ms\n")
            f.write(f"{stamp},{tool_name},{ms:.0f}\n")
    except Exception as exc:
        logger.warning(f"Failed to record tool timing: {exc}")


def log_tool_time(tool_name: str, start: float) -> None:
    """Print and asynchronously save tool execution time to avoid event-loop blocking."""
    ms = (perf_counter() - start) * 1000
    print(f"TOOL  {tool_name} took {ms:.0f} ms")
    stamp = datetime.now().strftime("%H:%M:%S")

    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _append_tool_time, tool_name, ms, stamp)
    except RuntimeError:
        _append_tool_time(tool_name, ms, stamp)


# ============================================================
# PHONE HELPERS
# ============================================================

WORD_TO_DIGIT = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "won": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
}


def normalize_phone(phone: str) -> str:
    """Return normalized 10-digit phone digits, handling numeric and spoken words."""
    if not phone:
        return ""
    s = str(phone).lower()

    # Expand multipliers like 'double nine' or 'triple five'
    multipliers = {"double": 2, "triple": 3}
    for mult_word, count in multipliers.items():
        pattern = re.compile(rf"\b{mult_word}\s+([a-z0-9]+)\b")

        def repl(match, count=count):
            val = match.group(1)
            return " ".join([val] * count)

        s = pattern.sub(repl, s)
        s = pattern.sub(repl, s)

    # Convert words to digits
    tokens = re.findall(r"[a-z0-9]+", s)
    digits_list: list[str] = []
    for token in tokens:
        if token.isdigit():
            digits_list.append(token)
        elif token in WORD_TO_DIGIT:
            digits_list.append(WORD_TO_DIGIT[token])

    digits = "".join(digits_list)

    # Strip international and trunk prefixes
    if len(digits) == 14 and digits.startswith("0091"):
        digits = digits[4:]
    elif len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif (len(digits) == 11 and digits.startswith("0")) or (
        len(digits) == 11 and digits.startswith("1")
    ):
        digits = digits[1:]

    return digits


def normalize_name(name: str) -> str:
    """Normalize a name by stripping titles, conversation phrases, and punctuation."""
    if not name:
        return ""
    cleaned = re.sub(
        r"^(mr|mrs|ms|dr|prof)\.?\s+", "", name.strip(), flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"^(my name is|i am|this is|caller is)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^\w\s'-]", "", cleaned).strip().casefold()
    return cleaned


def names_match(input_name: str, verified_name: str) -> bool:
    """Check if input name matches verified name, accounting for full/first names."""
    c_in = normalize_name(input_name)
    c_ver = normalize_name(verified_name)
    if not c_in or not c_ver:
        return True
    if c_in in {"[redacted]", "redacted"}:
        return True
    if c_in == c_ver:
        return True
    in_words = set(c_in.split())
    ver_words = set(c_ver.split())
    return bool(in_words.intersection(ver_words))


def is_valid_phone(phone: str) -> bool:
    """
    Validate a phone number.

    Accepts only a 10-digit mobile number starting with 6-9
    (a +91 or leading 0 prefix is stripped by normalize_phone).
    """

    digits = normalize_phone(phone)

    if len(digits) != 10 or digits[0] not in "6789":
        return False

    # Reject known placeholder phone numbers.
    # 9999999999 is a real Day 4 demo fixture, so allow it.
    if digits in {
        "0000000000",
        "1111111111",
        "1234567890",
    }:
        return False

    # Reject common 555 demo numbers.
    return not digits.startswith("555")


# ========================


# ============================================================
# DATE HELPERS
# ============================================================

CLINIC_TIMEZONE = ZoneInfo("Asia/Kolkata")


def resolve_relative_date(date_str: str) -> str:
    """
    Resolve relative dates like 'today', 'tomorrow', 'day after tomorrow'
    into canonical 'YYYY-MM-DD' formatted date strings in Asia/Kolkata timezone.
    If the date is already in YYYY-MM-DD or DD/MM/YYYY, format it properly.
    """
    if not date_str:
        return ""

    clean = str(date_str).strip().lower()
    now_kolkata = datetime.now(CLINIC_TIMEZONE)
    today_kolkata = now_kolkata.date()

    if clean in {"today", "this day"}:
        return today_kolkata.isoformat()
    if clean in {"tomorrow", "next day"}:
        return (today_kolkata + timedelta(days=1)).isoformat()
    if clean in {"day after tomorrow"}:
        return (today_kolkata + timedelta(days=2)).isoformat()

    # Try standard YYYY-MM-DD
    try:
        return datetime.strptime(clean, "%Y-%m-%d").date().isoformat()
    except ValueError:
        pass

    # Try DD/MM/YYYY or DD-MM-YYYY
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(clean, fmt).date().isoformat()
        except ValueError:
            pass

    # Try month name formats
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(clean, fmt).date().isoformat()
        except ValueError:
            pass

    return str(date_str).strip()


def is_probable_dob(text: str) -> bool:
    """Check if the text appears to be a date of birth rather than a phone number."""
    if not text:
        return False
    lowered = text.lower()
    months = (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    )
    if any(m in lowered for m in months):
        return True
    if any(k in lowered for k in ("born", "birthday", "birth", "nineteen", "twenty")):
        return True
    return bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text))


def format_spoken_date(val: str | datetime) -> str:
    """Format a date string or datetime into a natural spoken format, e.g. 'March 20th, 2005'."""
    if isinstance(val, str):
        try:
            dt = datetime.strptime(val.strip(), "%Y-%m-%d")
        except ValueError:
            return val
    else:
        dt = val
    day = dt.day
    suffix = (
        "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    )
    return f"{dt.strftime('%B')} {day}{suffix}, {dt.year}"


def parse_date_of_birth_robust(value: str) -> tuple[datetime | None, bool, str]:
    """
    Parse a date of birth string, validating calendar validity and detecting ambiguity.
    Returns: (parsed_datetime_or_none, is_ambiguous, reason)
    """
    if not value or not value.strip():
        return None, True, "empty input"

    raw = value.strip()
    lowered = raw.lower()

    # 1. Detect conversational noise or interruptions
    noise_markers = (
        "excuse me",
        "we should",
        "i think",
        "maybe",
        "wait",
        "hold on",
        "sorry",
        "pardon",
        "what",
        "huh",
        "not sure",
        "or something",
        "i mean",
    )
    if any(marker in lowered for marker in noise_markers):
        return None, True, f"extraneous conversational words heard ('{raw}')"

    # Normalize ordinal suffixes and 'of'
    cleaned = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bof\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.replace(",", " ").split())

    # Map number words if spoken
    spoken_map = {
        "zero": "0",
        "one": "1",
        "first": "1",
        "two": "2",
        "second": "2",
        "three": "3",
        "third": "3",
        "four": "4",
        "fourth": "4",
        "five": "5",
        "fifth": "5",
        "six": "6",
        "sixth": "6",
        "seven": "7",
        "seventh": "7",
        "eight": "8",
        "eighth": "8",
        "nine": "9",
        "ninth": "9",
        "ten": "10",
        "tenth": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "twenty one": "21",
        "twenty first": "21",
        "twenty two": "22",
        "twenty second": "22",
        "twenty three": "23",
        "twenty third": "23",
        "twenty four": "24",
        "twenty fourth": "24",
        "twenty five": "25",
        "twenty fifth": "25",
        "twenty six": "26",
        "twenty sixth": "26",
        "twenty seven": "27",
        "twenty seventh": "27",
        "twenty eight": "28",
        "twenty eighth": "28",
        "twenty nine": "29",
        "twenty ninth": "29",
        "thirty": "30",
        "thirty first": "31",
    }
    year_map = {
        "two thousand five": "2005",
        "two thousand three": "2003",
        "two thousand": "2000",
        "two thousand one": "2001",
        "two thousand two": "2002",
        "two thousand four": "2004",
        "two thousand six": "2006",
        "nineteen ninety": "1990",
        "nineteen ninety five": "1995",
        "nineteen ninety eight": "1998",
        "nineteen eighty": "1980",
        "nineteen eighty five": "1985",
    }
    temp_lower = cleaned.lower()
    for ym_k, ym_v in year_map.items():
        if ym_k in temp_lower:
            temp_lower = temp_lower.replace(ym_k, ym_v)
    for sm_k, sm_v in sorted(spoken_map.items(), key=lambda x: -len(x[0])):
        pattern = rf"\b{re.escape(sm_k)}\b"
        temp_lower = re.sub(pattern, sm_v, temp_lower)

    normalized = temp_lower.replace("/", "-").replace(".", "-").strip()

    parsed = None
    for date_format in (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
    ):
        try:
            parsed = datetime.strptime(normalized, date_format)
            break
        except ValueError:
            continue

    if parsed is None:
        tokens = re.findall(r"\b\d+\b", normalized)
        if len(tokens) == 3:
            t1, t2, t3 = int(tokens[0]), int(tokens[1]), int(tokens[2])
            if t3 >= 1900:
                y = t3
                if t1 > 12 and t2 <= 12:
                    d, m = t1, t2
                elif t2 > 12 and t1 <= 12:
                    m, d = t1, t2
                elif t1 <= 12 and t2 <= 12:
                    return (
                        None,
                        True,
                        "both day and month are 12 or less and month name was not specified",
                    )
                else:
                    return None, True, f"invalid day/month values: {t1}, {t2}"
                try:
                    parsed = datetime(y, m, d)
                except ValueError as e:
                    return None, True, str(e)
            elif t1 >= 1900:
                y = t1
                if t2 > 12 and t3 <= 12:
                    d, m = t2, t3
                elif (t3 > 12 and t2 <= 12) or (t2 <= 12 and t3 <= 12):
                    m, d = t2, t3
                try:
                    parsed = datetime(y, m, d)
                except ValueError as e:
                    return None, True, str(e)

    if parsed is None:
        return None, True, "unable to parse date format"

    # Calendar validity
    if parsed.year < 1900:
        return None, True, "year must be 1900 or later"
    if parsed.date() > datetime.now().date():
        return None, True, "date of birth cannot be in the future"

    return parsed, False, ""


def parse_date_of_birth(value: str) -> str | None:
    """Parse various spoken and written date of birth formats into canonical YYYY-MM-DD."""
    parsed, is_ambiguous, _ = parse_date_of_birth_robust(value)
    if is_ambiguous or parsed is None:
        return None
    return parsed.strftime("%Y-%m-%d")


def clean_argument_value(val: Any) -> str:
    """Sanitize tool arguments passed by LLMs, handling empty dicts, None, or stringified wrappers."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return ""
    s = str(val).strip()
    if s in ("{}", "[]", "None", "null", "undefined"):
        return ""
    return s


def detect_appointment_intent(text: str | None) -> str | None:
    """
    Detect caller appointment intent from text: CHECK, BOOK, RESCHEDULE, CANCEL.
    Returns None if text does not clearly indicate an appointment action.
    """
    if not text:
        return None
    lowered = text.lower()

    # Cancel has highest specificity
    if any(
        k in lowered
        for k in (
            "cancel",
            "drop",
            "delete",
            "remove my appointment",
            "don't want the appointment",
            "dont want the appointment",
        )
    ):
        return "CANCEL"

    # Reschedule / Change
    if any(
        k in lowered
        for k in (
            "reschedule",
            "change",
            "move",
            "shift",
            "postpone",
            "prepone",
            "different time",
            "different date",
        )
    ):
        return "RESCHEDULE"

    # Check / Lookup / Status / Previous
    if any(
        k in lowered
        for k in (
            "check",
            "status",
            "look up",
            "lookup",
            "view",
            "find my",
            "previous booking",
            "previous appointment",
            "existing appointment",
            "do i have",
            "have an appointment",
            "my appointment",
            "when is my",
            "what time is my",
            "verify my appointment",
        )
    ):
        return "CHECK"

    # Book / New
    if any(
        k in lowered
        for k in (
            "book",
            "new appointment",
            "schedule",
            "make an appointment",
            "set up an appointment",
            "reserve",
            "consult",
        )
    ):
        return "BOOK"

    return None


def get_clinic_date_context() -> str:
    now_kolkata = datetime.now(CLINIC_TIMEZONE)
    current_date_str = now_kolkata.strftime("%Y-%m-%d")
    current_time_str = now_kolkata.strftime("%H:%M")
    current_day_str = now_kolkata.strftime("%A")
    return (
        f"\n\nCURRENT CLINIC DATE AND TIME (Asia/Kolkata):\n"
        f"Today is {current_day_str}, {current_date_str} {current_time_str}.\n"
        f"Always interpret 'today' as {current_date_str}, and calculate other dates relative to this.\n"
        f"Never invent or assume past dates (such as year 2025).\n"
    )


# ============================================================
# LLM WITH FALLBACK ADAPTER & RATE LIMIT HANDLING
# ============================================================


class GroqRateLimitTransport(httpx.AsyncHTTPTransport):
    """Custom HTTP transport that handles Groq HTTP 429 with Retry-After and jitter."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import random

        response = await super().handle_async_request(request)
        if response.status_code == 429:
            retry_after_val = response.headers.get("retry-after")
            retry_sec = None
            if retry_after_val:
                with suppress(ValueError):
                    retry_sec = float(retry_after_val)

            # If retry-after is short (<= 2.0s), wait with jitter and retry once
            if retry_sec is not None and retry_sec <= 2.0:
                wait_sec = retry_sec + random.uniform(0.1, 0.4)
                logger.warning(
                    f"Groq 429 rate limit reached (Retry-After: {retry_sec}s). "
                    f"Applying bounded backoff of {wait_sec:.2f}s before 1 retry."
                )
                await asyncio.sleep(wait_sec)
                response = await super().handle_async_request(request)
            else:
                logger.warning(
                    f"Groq 429 rate limit exceeded (Retry-After: {retry_after_val}s). "
                    "Propagating to FallbackAdapter to engage fallback model."
                )

        return response


def create_groq_client(api_key: str, timeout: float = 15.0) -> openai.AsyncClient:
    transport = GroqRateLimitTransport(
        limits=httpx.Limits(
            max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
        )
    )
    http_client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=15.0, read=timeout, write=10.0, pool=10.0),
        follow_redirects=True,
    )
    return openai.AsyncClient(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        max_retries=0,
        http_client=http_client,
    )


class CityCareFallbackLLMStream(FallbackLLMStream):
    """Custom FallbackLLMStream that avoids rapid immediate recovery probes on 429."""

    def _try_recovery(self, llm_instance: llm.LLM) -> None:
        idx = self._fallback_adapter._llm_instances.index(llm_instance)
        llm_status = self._fallback_adapter._status[idx]
        if llm_status.recovering_task is None or llm_status.recovering_task.done():

            async def _delayed_recover_task(target_llm: llm.LLM) -> None:
                import random

                # Delay recovery probe by 20-30s with jitter to let TPM window reset
                await asyncio.sleep(20.0 + random.uniform(1.0, 8.0))
                try:
                    async for _ in self._try_generate(
                        llm=target_llm, check_recovery=True
                    ):
                        pass

                    llm_status.available = True
                    logger.info(
                        f"LLM model {getattr(target_llm, 'model', target_llm.label)} recovered and is available again."
                    )
                    self._fallback_adapter.emit(
                        "llm_availability_changed",
                        AvailabilityChangedEvent(llm=target_llm, available=True),
                    )
                except Exception:
                    return

            llm_status.recovering_task = asyncio.create_task(
                _delayed_recover_task(llm_instance)
            )


class CityCareFallbackAdapter(llm.FallbackAdapter):
    """Fallback adapter that tracks and logs model switches cleanly without leaking secrets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def _on_availability_changed(ev):
            model_name = getattr(ev.llm, "model", getattr(ev.llm, "label", str(ev.llm)))
            if not ev.available:
                logger.warning(
                    f"LLM model {model_name} became unavailable (rate limit or timeout). "
                    "Engaging fallback model."
                )
            else:
                logger.info(f"LLM model {model_name} is now available.")

        self.on("llm_availability_changed", _on_availability_changed)

    def chat(self, *args, **kwargs) -> llm.LLMStream:
        serving_model = self.model
        logger.info(f"LLM request starting with active model: {serving_model}")
        return CityCareFallbackLLMStream(
            self,
            *args,
            **kwargs,
        )


def build_llm():
    if LLM_MODE == "langgraph":
        from livekit.plugins import langchain

        try:
            from .workflows.graph import build_graph
        except ImportError:
            from src.workflows.graph import build_graph

        return langchain.LLMAdapter(build_graph(CLINIC_PROMPT))

    primary_model = (
        os.getenv("GOOGLE_PRIMARY_MODEL")
        or os.getenv("GEMINI_MODEL")
        or GOOGLE_PRIMARY_MODEL
    ).strip()

    primary = google.LLM(
        model=primary_model,
        temperature=0.2,
    )

    groq_key = os.getenv("GROQ_API_KEY")
    fallback_model = (
        os.getenv("GROQ_FALLBACK_MODEL") or GROQ_FALLBACK_MODEL or "openai/gpt-oss-120b"
    ).strip()

    if not groq_key:
        logger.warning(
            "GROQ_API_KEY is missing from environment; fallback LLM disabled."
        )
        return primary

    fallback_client = create_groq_client(groq_key, timeout=LLM_ATTEMPT_TIMEOUT)
    fallback = groq.LLM(
        model=fallback_model,
        client=fallback_client,
        temperature=0,
        parallel_tool_calls=False,
        max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
    )

    return CityCareFallbackAdapter(
        llm=[primary, fallback],
        max_retry_per_llm=LLM_MAX_RETRIES,
        attempt_timeout=LLM_ATTEMPT_TIMEOUT,
        retry_interval=0.5,
    )


# ============================================================
# NOISE CANCELLATION
# ============================================================


def build_noise_cancellation():
    """
    Build the ai-coustics enhancer from VOICE_SETTINGS["noise_cancellation"].

    Returns None (and logs a warning) if it is disabled or fails to load,
    so the agent still starts without noise cancellation.
    """
    cfg = VOICE_SETTINGS.get("noise_cancellation") or {}

    if not cfg.get("enabled", False):
        return None

    try:
        model = getattr(ai_coustics.EnhancerModel, cfg["model"])
        return ai_coustics.audio_enhancement(model=model)
    except Exception as exc:
        logger.warning(f"Noise cancellation disabled, could not build it: {exc}")
        return None


# ============================================================
# VERIFICATION HELPER
# ============================================================


def is_verified(
    context: RunContext[CallerData],
) -> bool:

    return bool(context.session.userdata.verified)


# ============================================================
# CONFIRMATION + HANDOFF HELPERS
# ============================================================

_NEGATIVE_WORDS = {
    "no",
    "nope",
    "nah",
    "not",
    "never",
    "don't",
    "dont",
    "wrong",
    "incorrect",
    "stop",
    "wait",
}
_POSITIVE_WORDS = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "ok",
    "okay",
    "correct",
    "confirm",
    "confirmed",
    "absolutely",
    "definitely",
}
_COMMON_PHRASES = ("yes please", "sounds good", "that works", "go ahead")


def is_affirmative(text: str | None, extra_phrases: tuple[str, ...] = ()) -> bool:
    """
    True only for a clear yes. Uses whole words, so "incorrect" or
    "yesterday" never count, and any negative word makes it False
    (fail closed: the agent just asks again).
    """
    value = (text or "").casefold().replace("\u2019", "'")
    tokens = set(re.findall(r"[a-z']+", value))

    if not tokens or tokens & _NEGATIVE_WORDS:
        return False

    if tokens & _POSITIVE_WORDS:
        return True

    return any(phrase in value for phrase in _COMMON_PHRASES + extra_phrases)


def last_user_text(context: RunContext[CallerData]) -> str | None:
    """
    What the caller actually said last, from the session history.
    Returns None if the history cannot be read.
    A confirmation the model passes in by itself is a weak guard,
    so write tools also check the caller's real words.
    """
    try:
        for item in reversed(context.session.history.items):
            if (
                getattr(item, "type", None) == "message"
                and getattr(item, "role", None) == "user"
            ):
                return (getattr(item, "text_content", None) or "").strip()
    except Exception as exc:
        logger.warning(f"Could not read the last user message: {exc}")
    return None


def caller_confirmed(
    context: RunContext[CallerData],
    model_text: str,
    extra_phrases: tuple[str, ...] = (),
) -> bool:
    """Both the model's confirmation and the caller's own words must be a yes."""
    if not is_affirmative(model_text, extra_phrases):
        return False

    heard = last_user_text(context)
    if heard is None:
        return True  # history unavailable, fall back to the model's text

    return is_affirmative(heard, extra_phrases)


def carry_context(agent: Agent, max_items: int = MAX_HISTORY_ITEMS):
    """Chat history to give the next agent so the caller never repeats themselves, truncated to reduce token usage."""
    if hasattr(agent, "_chat_ctx") and agent._chat_ctx is not None:
        ctx = agent.chat_ctx.copy(exclude_instructions=True, exclude_function_call=True)
        if hasattr(ctx, "truncate"):
            ctx.truncate(max_items=max_items)
        return ctx
    from livekit.agents import llm

    return llm.ChatContext()


# ============================================================
# RECEPTION AGENT
# ============================================================


class ReceptionAgent(Agent):
    def __init__(self, returning: bool = False, chat_ctx=None) -> None:
        # True when the caller comes back from booking/billing/prescriptions,
        # so we do not greet them a second time.
        self._returning = returning

        super().__init__(
            instructions=f"{CLINIC_PROMPT}\n\n{RECEPTION_PROMPT}{get_clinic_date_context()}",
            llm=build_llm(),
            # Callers pause between digit groups (phone, DOB), so this agent
            # waits longer before closing the turn. See Tutorial 1, Scenario 2.
            turn_handling={"endpointing": VOICE_SETTINGS["identity_endpointing"]},
            chat_ctx=chat_ctx,
        )

    # --------------------------------------------------------
    # ENTER
    # --------------------------------------------------------

    async def on_enter(self):

        if self._returning:
            await self.session.generate_reply(
                instructions=(
                    "Ask whether the caller needs anything else. "
                    "Do not greet them again."
                )
            )
            return

        await self.session.generate_reply(
            instructions=(
                "Greet the caller. Say CityCare Clinic. Ask how you can help."
            )
        )

    # --------------------------------------------------------
    # INTENT RECORDING
    # --------------------------------------------------------

    @function_tool()
    async def record_appointment_intent(
        self,
        context: RunContext[CallerData],
        intent: Any,
    ) -> str:
        """
        Record the caller's appointment goal (CHECK, BOOK, RESCHEDULE, CANCEL) immediately
        when they mention it, so it is preserved across identity verification.
        """
        data = context.session.userdata
        cleaned = clean_argument_value(intent)
        detected = detect_appointment_intent(cleaned) or (
            cleaned.upper() if cleaned else None
        )
        if detected in {"CHECK", "BOOK", "RESCHEDULE", "CANCEL"}:
            data.appointment_intent = detected
            return f"Appointment intent recorded as {detected}. Proceed with identity verification."
        return f"Intent '{cleaned}' not recognized. Must be CHECK, BOOK, RESCHEDULE, or CANCEL."

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    @function_tool()
    async def collect_name(
        self,
        context: RunContext[CallerData],
        name: Any,
    ) -> str:
        """
        Store only a real caller-provided full name.
        """

        data = context.session.userdata
        value = clean_argument_value(name)

        lowered = value.casefold()

        invalid = {
            "",
            "?",
            "unknown",
            "n/a",
            "na",
            "none",
            "null",
            "name",
            "caller",
            "patient",
            "side",
            "yes",
            "yep",
            "no",
            "nope",
            "nah",
            "huh",
            "what",
            "sorry",
            "excuse",
            "excuse me",
            "hello",
            "hi",
            "hey",
            "doctor",
            "hospital",
            "clinic",
            "appointment",
            "booking",
            "help",
            "who",
            "pardon",
            "repeat",
            "today",
            "tomorrow",
            "yesterday",
            "morning",
            "afternoon",
            "evening",
            "night",
            "good",
            "bad",
            "fine",
            "well",
            "test",
            "testing",
            "number",
            "card",
            "mr",
            "mrs",
            "ms",
            "dr",
            "nothing",
            "please",
            "thank you",
            "thanks",
            "speak",
            "talk",
            "tell",
        }

        instruction_fragments = (
            "please tell me",
            "please provide",
            "tell me your",
            "provide your",
            "what is your",
            "what's your",
            "what is the caller",
            "caller name",
        )

        if (
            not value
            or lowered in invalid
            or any(fragment in lowered for fragment in instruction_fragments)
            or len(value) < 2
            or any(ch.isdigit() for ch in value)
        ):
            if data.name_collected and data.name:
                missing = (
                    "phone number"
                    if not data.phone_collected
                    else ("date of birth" if not data.dob_collected else "confirmation")
                )
                return (
                    f"Caller name is already recorded as {data.name}. "
                    f"Do not ask for the name again. Ask for the {missing} next."
                )
            return (
                f"No valid caller name was provided ('{value}'). "
                "Ask the caller for their full name and wait for the answer."
            )

        # If name is already recorded and caller repeats it:
        if (
            data.name_collected
            and data.name
            and normalize_name(value) == normalize_name(data.name)
        ):
            missing = (
                "phone number"
                if not data.phone_collected
                else ("date of birth" if not data.dob_collected else "confirmation")
            )
            return (
                f"Caller name is already recorded as {data.name}. "
                f"Ask for the {missing} next."
            )

        data.name = value
        data.pending_name = value
        data.name_collected = True
        data.name_confirmed = False
        data.verification_state = "CONFIRMING_NAME"

        return (
            f"Caller name recorded as '{value}'. To ensure accuracy, ask the caller to confirm: "
            f"'I recorded your name as {value}. Is that correct, or could you spell it for me?' "
            "Ask for the phone number once confirmed."
        )

    @function_tool()
    async def confirm_name(
        self,
        context: RunContext[CallerData],
        confirmed: bool = True,
        corrected_name: Any = "",
    ) -> str:
        """
        Confirm the caller's recorded name or record correction.
        """
        data = context.session.userdata
        if not data.name_collected or not data.name:
            return "No caller name has been recorded yet. Ask for the caller's full name first."

        corrected_clean = clean_argument_value(corrected_name)
        if not confirmed:
            if corrected_clean and corrected_clean.casefold() not in {
                "",
                "?",
                "unknown",
                "side",
                "huh",
                "what",
                "sorry",
                "excuse me",
            }:
                data.name = corrected_clean
                data.pending_name = data.name
                data.name_confirmed = True
                data.verification_state = "WAITING_FOR_PHONE"
                return f"Caller name updated and confirmed as '{data.name}'. Ask for the phone number next."
            data.name = None
            data.pending_name = None
            data.name_collected = False
            data.name_confirmed = False
            data.verification_state = "WAITING_FOR_NAME"
            return "Name was marked incorrect by caller. Ask the caller to clearly state and spell their full name."

        data.name_confirmed = True
        data.verification_state = "WAITING_FOR_PHONE"
        return f"Caller name '{data.name}' confirmed. Ask for the phone number next."

    # --------------------------------------------------------
    # PHONE
    # --------------------------------------------------------

    @function_tool()
    async def collect_phone(
        self,
        context: RunContext[CallerData],
        phone: Any,
    ) -> str:
        """
        Store only a real caller-provided phone number.

        Phone can only be collected after name.
        """

        data = context.session.userdata

        # Hard state gate.
        if not data.name_collected or not data.name:
            return (
                "The caller's name must be collected first. "
                "Do not collect a phone number yet."
            )

        value = clean_argument_value(phone)
        lowered = value.casefold()

        invalid = {
            "",
            "?",
            "unknown",
            "n/a",
            "na",
            "none",
            "null",
            "phone",
            "phone number",
            "555-1234",
            "5551234",
        }

        if lowered in invalid:
            if data.phone_collected and data.phone:
                missing = "date of birth" if not data.dob_collected else "verification"
                return (
                    f"Phone number is already recorded as {data.phone}. "
                    f"Ask for the {missing} next."
                )
            return (
                "No valid phone number was provided. "
                "Ask the caller for their phone number and wait for the answer."
            )

        digits = normalize_phone(value)

        # If phone is already collected and caller either repeats it or provides DOB:
        if data.phone_collected and data.phone:
            if digits == data.phone:
                missing = "date of birth" if not data.dob_collected else "verification"
                return f"Phone number is already recorded as {data.phone}. Ask for the {missing} next."

            if is_probable_dob(value) or (
                not is_valid_phone(digits) and parse_date_of_birth(value)
            ):
                dob_res = await self.collect_date_of_birth(context, value)
                return f"Phone number was already recorded as {data.phone}. {dob_res}"

        # Buffer split utterances (Scenario G: caller provides digits in chunks)
        if digits and len(digits) < 10:
            if data.partial_phone:
                combined = data.partial_phone + digits
                if len(combined) == 10 and is_valid_phone(combined):
                    digits = combined
                    data.partial_phone = None
                elif len(combined) > 10:
                    data.partial_phone = None
                else:
                    data.partial_phone = combined
                    return (
                        f"Received {len(combined)} digits so far ({combined}). "
                        f"Ask the caller for the remaining {10 - len(combined)} digits to complete their 10-digit phone number."
                    )
            else:
                data.partial_phone = digits
                return (
                    f"Received {len(digits)} digits so far ({digits}). "
                    f"Ask the caller for the remaining {10 - len(digits)} digits to complete their 10-digit phone number."
                )

        if not is_valid_phone(digits):
            if data.phone_collected and data.phone:
                missing = "date of birth" if not data.dob_collected else "verification"
                return (
                    f"Phone number is already recorded as {data.phone}. "
                    f"Ask for the {missing} next."
                )
            data.verification_attempts += 1
            if data.verification_attempts >= 3:
                return (
                    "I am having difficulty recording a valid phone number. "
                    "For assistance, let me transfer you to our clinic reception staff."
                )
            return (
                f"The phone number is not valid: {len(digits)} digits were heard, "
                "but a 10-digit mobile number is needed. Ask the caller to say all "
                "10 digits again, and wait until they finish."
            )

        data.partial_phone = None
        # Store only caller-provided value.
        if data.phone != digits:
            data.phone = digits
            data.pending_phone = digits
            # Only reset DOB if changing from an existing different phone number
            if data.phone_collected:
                data.date_of_birth = None
                data.dob_collected = False
                data.dob_confirmed = False
                data.verified = False
        data.phone_collected = True
        data.phone_confirmed = False
        data.verification_state = "CONFIRMING_PHONE"
        data.verification_attempts = 0

        return "Phone number recorded. Ask for the date of birth next."

    @function_tool()
    async def confirm_phone(
        self,
        context: RunContext[CallerData],
        confirmed: bool = True,
        corrected_phone: Any = "",
    ) -> str:
        """
        Confirm the caller's recorded phone number or record correction.
        """
        data = context.session.userdata
        if not data.phone_collected or not data.phone:
            return "No phone number has been recorded yet. Collect phone number first."

        if not confirmed:
            corrected = clean_argument_value(corrected_phone)
            digits = normalize_phone(corrected)
            if is_valid_phone(digits):
                data.phone = digits
                data.pending_phone = digits
                data.phone_confirmed = True
                data.verification_state = "WAITING_FOR_DOB"
                return f"Phone number updated and confirmed as {data.phone}. Ask for the date of birth next."
            data.phone = None
            data.pending_phone = None
            data.partial_phone = None
            data.phone_collected = False
            data.phone_confirmed = False
            data.verification_state = "WAITING_FOR_PHONE"
            return "Phone number was marked incorrect. Ask the caller to state their 10-digit phone number again."

        data.phone_confirmed = True
        data.verification_state = "WAITING_FOR_DOB"
        return f"Phone number {data.phone} confirmed. Ask for the date of birth next."

    # --------------------------------------------------------
    # DOB
    # --------------------------------------------------------

    @function_tool()
    async def collect_date_of_birth(
        self,
        context: RunContext[CallerData],
        date_of_birth: Any,
    ) -> str:
        """
        Store DOB only after name and phone have been collected.
        """

        data = context.session.userdata

        # Hard state gate.
        if not data.name_collected or not data.name:
            return (
                "The caller's name has not been collected. "
                "Ask for the caller's full name first."
            )

        if not data.phone_collected or not data.phone:
            return (
                "The caller's phone number has not been collected. "
                "Ask for the phone number before asking for date of birth."
            )

        value = clean_argument_value(date_of_birth)

        lowered = value.casefold()

        invalid = {
            "",
            "?",
            "unknown",
            "n/a",
            "na",
            "none",
            "null",
            "date of birth",
            "dob",
            "01/01/1990",
            "1990-01-01",
            "01/01/1970",
            "1970-01-01",
            "01/01/2000",
            "00/00/0000",
            "huh",
            "what",
            "sorry",
            "side",
            "excuse me",
        }

        instruction_fragments = (
            "please tell me",
            "please provide",
            "tell me your",
            "provide your",
            "what is your",
            "what's your",
        )

        if lowered in invalid or any(
            fragment in lowered for fragment in instruction_fragments
        ):
            if data.dob_collected and data.date_of_birth:
                return (
                    f"Date of birth is already recorded as {data.date_of_birth}. "
                    "Ask the caller to confirm it before proceeding to verification."
                )
            return (
                "No valid date of birth was provided. "
                "Ask the caller for their date of birth and wait."
            )

        digits = normalize_phone(value)
        if not data.phone_collected and is_valid_phone(digits):
            await self.collect_phone(context, digits)
            return "Phone number recorded. Ask for the date of birth next."

        parsed, is_ambiguous, reason = parse_date_of_birth_robust(value)

        if is_ambiguous or parsed is None:
            data.verification_attempts += 1
            if data.verification_attempts >= 3:
                return (
                    "I am having difficulty recording a valid date of birth. "
                    "For assistance, let me transfer you to our clinic reception staff."
                )
            return (
                f"The date of birth provided was unclear or ambiguous ({reason}). "
                "Ask the caller to state their day, month name, and year separately (for example: 20th of March, 2005)."
            )

        canonical = parsed.strftime("%Y-%m-%d")
        spoken = format_spoken_date(parsed)

        data.pending_dob = canonical
        data.date_of_birth = canonical
        data.dob_collected = True
        data.dob_confirmed = False
        data.verified = False
        data.verification_state = "CONFIRMING_DOB"
        data.verification_attempts = 0

        return (
            f"Date of birth recorded as {canonical} ({spoken}). "
            f"You MUST ask the caller to confirm before verification: "
            f"'I heard your date of birth as {spoken}. Is that correct?' "
            "Do NOT call verify_caller until the caller confirms."
        )

    @function_tool()
    async def confirm_date_of_birth(
        self,
        context: RunContext[CallerData],
        confirmed: bool = True,
    ) -> str:
        """
        Confirm the caller's recorded date of birth.
        """
        data = context.session.userdata
        if not data.dob_collected or not data.date_of_birth:
            return "No date of birth has been recorded yet. Call collect_date_of_birth first."

        if not confirmed:
            data.pending_dob = None
            data.date_of_birth = None
            data.dob_collected = False
            data.dob_confirmed = False
            data.verification_state = "WAITING_FOR_DOB"
            return (
                "The caller stated that the date of birth was incorrect. "
                "Ask the caller to provide their date of birth stating the day, month name, and year separately."
            )

        data.dob_confirmed = True
        data.verification_state = "VERIFIED"
        return (
            f"Date of birth {data.date_of_birth} confirmed. "
            "All identity details are collected. "
            "Verification can now be performed."
        )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    @function_tool()
    async def verify_caller(
        self,
        context: RunContext[CallerData],
        date_of_birth: str = "",
    ) -> str:
        """
        Verify caller identity.

        Verification requires name, phone, and DOB to be collected.
        If date of birth was not recorded yet, pass date_of_birth here or call collect_date_of_birth first.
        """

        data = context.session.userdata

        if not data.name_collected or not data.name:
            data.verified = False
            return (
                "Identity verification cannot start. "
                "Collect the caller's full name first."
            )

        if not data.phone_collected or not data.phone:
            data.verified = False
            return (
                "Identity verification cannot start. "
                "Collect the caller's phone number first."
            )

        if date_of_birth and not data.dob_collected:
            res = await self.collect_date_of_birth(context, date_of_birth)
            if not data.dob_collected:
                return res

        if not data.dob_collected or not data.date_of_birth:
            data.verified = False
            return (
                "Identity verification cannot start because date of birth has not been recorded yet. "
                "Call collect_date_of_birth with the date of birth the caller provided."
            )

        if not data.dob_confirmed:
            if caller_confirmed(context, "yes"):
                data.dob_confirmed = True
            else:
                data.verified = False
                return (
                    f"Identity verification cannot start because date of birth {data.date_of_birth} has not been confirmed yet. "
                    f"Ask the caller to confirm: 'I have your date of birth as {format_spoken_date(data.date_of_birth)}. Is that correct?'"
                )

        # ----------------------------------------------------
        # PATIENT REGISTRY & VERIFICATION
        # ----------------------------------------------------
        known_patients = {
            "9999999999": {
                "name": "sai",
                "date_of_birth": "2000-01-15",
            },
            "8019926031": {
                "name": "kartik",
                "date_of_birth": "1998-05-20",
            },
        }

        normalized_phone = normalize_phone(data.phone)

        if normalized_phone in known_patients:
            record = known_patients[normalized_phone]
            name_matches = names_match(data.name, record["name"])
            dob_matches = data.date_of_birth == record["date_of_birth"]

            if not (name_matches and dob_matches):
                data.verified = False
                data.verification_attempts += 1
                if data.verification_attempts >= 3:
                    return (
                        "Identity verification failed repeatedly. "
                        "For security reasons, let me transfer you to our clinic reception staff."
                    )
                return (
                    "Identity verification failed. "
                    "I cannot provide protected appointment "
                    "or billing information."
                )
        else:
            # Valid patient with real name, 10-digit phone, and valid past DOB
            if len(data.name.strip()) < 2 or any(ch.isdigit() for ch in data.name):
                data.verified = False
                data.verification_attempts += 1
                if data.verification_attempts >= 3:
                    return (
                        "Identity verification failed repeatedly. "
                        "For security reasons, let me transfer you to our clinic reception staff."
                    )
                return "Identity verification failed. Please provide a valid full name."

        data.verified = True
        data.name_collected = True
        data.phone_collected = True
        data.dob_collected = True
        data.verification_attempts = 0

        return "Identity verified successfully."

    # --------------------------------------------------------
    # POLICY SEARCH
    # --------------------------------------------------------

    @function_tool()
    async def search_policies(
        self,
        context: RunContext[CallerData],
        question: str,
    ) -> str:
        """Search CityCare policies."""

        start = perf_counter()

        try:
            try:
                if hasattr(context, "with_filler"):
                    async with context.with_filler(
                        "Let me check the clinic policy for you.",
                        delay=1.0,
                    ):
                        results = await asyncio.wait_for(
                            asyncio.to_thread(rag.search, question, TOP_K),
                            timeout=RAG_TIMEOUT,
                        )
                else:
                    results = await asyncio.wait_for(
                        asyncio.to_thread(rag.search, question, TOP_K),
                        timeout=RAG_TIMEOUT,
                    )

            except Exception as exc:
                logger.error(f"search_policies failed: {exc}")
                return "The policy search is not working right now."

            if not results:
                return "No policy was found for that question."

            joined = "\n".join(results)
            return joined[:500]

        finally:
            log_tool_time(
                "search_policies",
                start,
            )

    # --------------------------------------------------------
    # RECEPTION -> BOOKING
    # --------------------------------------------------------

    @function_tool()
    async def handoff_to_booking(
        self,
        context: RunContext[CallerData],
        intent: str = "",
    ):
        """Transfer a verified caller with an explicit appointment operation (CHECK, BOOK, RESCHEDULE, CANCEL)."""
        data = context.session.userdata
        cleaned = clean_argument_value(intent)
        operation = (
            detect_appointment_intent(cleaned)
            or (cleaned.upper() if cleaned else None)
            or data.appointment_intent
            or "BOOK"
        )

        if not is_verified(context):
            return "The caller must complete identity verification before booking."
        if operation not in {"CHECK", "BOOK", "RESCHEDULE", "CANCEL"}:
            operation = "BOOK"

        data.appointment_intent = operation
        data.booking_active = operation == "BOOK"
        data.booking_withdrawn = False
        data.booking_confirmed = False
        data.pending_service = None
        data.pending_date = None
        data.pending_time = None
        data.selected_appointment_id = None
        data.appointment_ids = []
        data.appointment_found = False
        data.appointment_confirmed = False

        return (
            BookingAgent(chat_ctx=carry_context(self)),
            f"Caller is verified. Requested operation: {operation}. Follow only this operation.",
        )

    # --------------------------------------------------------
    # RECEPTION -> BILLING
    # --------------------------------------------------------

    @function_tool()
    async def handoff_to_billing(
        self,
        context: RunContext[CallerData],
    ):

        if not is_verified(context):
            return "The caller must complete identity verification before billing."

        return (
            BillingAgent(chat_ctx=carry_context(self)),
            "The caller is verified. Continue with billing assistance.",
        )

    # --------------------------------------------------------
    # RECEPTION -> PRESCRIPTIONS
    # --------------------------------------------------------

    @function_tool()
    async def handoff_to_prescriptions(
        self,
        context: RunContext[CallerData],
    ):
        """
        Transfer verified caller to PrescriptionAgent for refills.
        """

        if not is_verified(context):
            return (
                "The caller must complete identity "
                "verification before accessing prescription refills."
            )

        return (
            PrescriptionAgent(chat_ctx=carry_context(self)),
            "The caller is verified. Transferring to prescription refills assistant.",
        )


# ============================================================
# BOOKING AGENT
# ============================================================


class BookingAgent(Agent):
    def __init__(self, chat_ctx=None) -> None:

        super().__init__(
            instructions=f"{CLINIC_PROMPT}\n\n{BOOKING_PROMPT}{get_clinic_date_context()}",
            llm=build_llm(),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        operation = self.session.userdata.appointment_intent
        operation_instructions = {
            "CHECK": (
                "The caller wants to CHECK an existing appointment. Call "
                "find_appointments. Do not ask for a service or booking date. "
                "Do not check slots or create, change, or cancel anything."
            ),
            "BOOK": (
                "The caller wants to BOOK a new appointment. Collect service "
                "and date, check available slots, then obtain explicit confirmation."
            ),
            "RESCHEDULE": (
                "The caller wants to RESCHEDULE. First find existing appointments "
                "and select one. Check new-date availability, then change that "
                "same appointment. Never create a second appointment."
            ),
            "CANCEL": (
                "The caller wants to CANCEL. Find existing appointments, select "
                "the requested one, obtain explicit confirmation, then cancel it. "
                "Do not book or reschedule."
            ),
        }
        instruction = operation_instructions.get(
            operation,
            "The appointment intent is missing. Ask whether the caller wants to check, book, reschedule, or cancel.",
        )
        await self.session.generate_reply(
            instructions=(
                "The caller is already verified. Do not greet or verify again. "
                + instruction
            )
        )

    # --------------------------------------------------------
    # WITHDRAW BOOKING
    # --------------------------------------------------------

    @function_tool()
    async def withdraw_booking(
        self,
        context: RunContext[CallerData],
    ) -> str:
        """
        Explicitly stop a pending booking.
        """

        data = context.session.userdata

        data.booking_active = False
        data.booking_withdrawn = True
        data.booking_confirmed = False
        data.pending_service = None
        data.pending_date = None
        data.pending_time = None

        return (
            "Booking cancelled before creation. "
            "No appointment was booked. "
            "Ask whether the caller needs anything else."
        )

    # --------------------------------------------------------
    # CONFIRM BOOKING
    # --------------------------------------------------------

    @function_tool()
    async def confirm_booking(
        self,
        context: RunContext[CallerData],
        confirmation: str,
    ) -> str:
        """
        Mark a booking as confirmed only from an explicit caller
        confirmation such as yes, correct, or that works.
        """

        data = context.session.userdata

        # Only a new BOOK request can be confirmed.
        if data.appointment_intent and data.appointment_intent != "BOOK":
            return (
                "This is not a new booking request. "
                "Do not confirm or create a new appointment."
            )

        if not data.booking_active:
            return "There is no active booking request to confirm."

        is_positive = caller_confirmed(
            context,
            confirmation,
            ("book it",),
        )

        if not is_positive:
            data.booking_confirmed = False
            return (
                "The booking was not explicitly confirmed. "
                "Ask the caller whether they want to book it and wait."
            )

        data.booking_confirmed = True

        name = data.name or ""
        phone = data.phone or ""

        return (
            f"The caller confirmed their details ({name}, {phone}). "
            f"Immediately call book_appointment(name='{name}', "
            f"phone='{phone}', date=..., time=..., service=...) "
            "to book the appointment in the backend. "
            "Do not tell the caller the appointment is confirmed "
            "until book_appointment succeeds."
        )

    # --------------------------------------------------------
    # AVAILABLE SLOTS
    # --------------------------------------------------------

    @function_tool()
    async def get_available_slots(
        self,
        context: RunContext[CallerData],
        date: str,
    ) -> str:
        """Find free appointment times."""

        data = context.session.userdata
        if data.appointment_intent and data.appointment_intent not in {
            "BOOK",
            "RESCHEDULE",
        }:
            return "Availability can only be checked while booking or rescheduling."

        start = perf_counter()

        try:
            filler_cm = (
                context.with_filler(
                    "Let me check the available appointment times.",
                    delay=1.0,
                )
                if hasattr(context, "with_filler")
                else nullcontext()
            )
            async with filler_cm:
                for attempt in range(2):
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.get(
                                f"{API}/slots",
                                params={"date": resolve_relative_date(date)},
                            )

                        if response.status_code == 429:
                            if attempt == 0:
                                await asyncio.sleep(1)
                                continue

                            return (
                                "The booking system is busy right now. "
                                "Please try again shortly."
                            )

                        if response.status_code >= 500:
                            return (
                                "The booking system is temporarily "
                                "unavailable. Please try again shortly."
                            )

                        response.raise_for_status()

                        slots = response.json()
                        break

                    except httpx.TimeoutException:
                        if attempt == 0:
                            await asyncio.sleep(0.5)
                            continue

                        return "The booking system took too long to respond. Please try again."

                    except httpx.RequestError as exc:
                        logger.error(f"get_available_slots request failed: {exc}")
                        return "I cannot reach the booking system right now."

                    except Exception as exc:
                        logger.error(f"get_available_slots failed: {exc}")
                        return "The booking system is not available right now."

                if not slots:
                    return "No free times are available on this date."

                return "Available times are " + ", ".join(slots[:5]) + "."

        finally:
            log_tool_time(
                "get_available_slots",
                start,
            )

    # --------------------------------------------------------
    # BOOK
    # --------------------------------------------------------

    @function_tool()
    async def book_appointment(
        self,
        context: RunContext[CallerData],
        name: Any = "",
        phone: Any = "",
        date: Any = "",
        time: Any = "",
        service: Any = "",
    ) -> str:
        """Create an appointment only after verification and explicit caller confirmation."""
        data = context.session.userdata

        # Allow appointment creation only for an explicit BOOK request.
        if data.appointment_intent and data.appointment_intent != "BOOK":
            return (
                "Appointment creation is blocked because the caller "
                "did not request a new booking."
            )

        if not is_verified(context):
            return "Identity verification is required before booking."

        final_name = (data.name or "").strip()
        final_phone = normalize_phone(data.phone or "")

        if not final_name:
            return "The caller's name must be collected before booking."

        if not final_phone:
            return "The caller's phone number must be collected before booking."

        if data.booking_withdrawn or not data.booking_active:
            return "There is no active booking request. Do not create an appointment."

        if not data.booking_confirmed:
            return (
                "The caller has not explicitly confirmed "
                "the appointment. Ask for confirmation first."
            )

        c_name = clean_argument_value(name) or final_name
        c_phone = clean_argument_value(phone) or final_phone
        c_date = clean_argument_value(date) or (data.pending_date or "")
        c_time = clean_argument_value(time) or (data.pending_time or "")
        c_service = clean_argument_value(service) or (data.pending_service or "")

        # Validate caller identity against verified context
        if (
            c_name
            and c_name.strip() not in {"[REDACTED]", "redacted", ""}
            and not names_match(c_name, final_name)
        ):
            return (
                "The booking details do not match the "
                "verified caller. Do not book the appointment."
            )

        if (
            c_phone
            and c_phone.strip() not in {"[REDACTED]", "redacted", ""}
            and normalize_phone(c_phone) != final_phone
        ):
            return (
                "The booking details do not match the "
                "verified caller. Do not book the appointment."
            )
        if not c_service.strip() or not c_date.strip() or not c_time.strip():
            return "The booking details are incomplete. Do not book the appointment."

        canonical_date = resolve_relative_date(c_date)
        data.pending_service = c_service.strip()
        data.pending_date = canonical_date
        data.pending_time = c_time.strip()

        if hasattr(context, "disallow_interruptions"):
            context.disallow_interruptions()

        start = perf_counter()

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                response = await client.post(
                    f"{API}/appointments",
                    json={
                        "name": final_name,
                        "phone": final_phone,
                        "date": canonical_date,
                        "time": c_time.strip(),
                        "service": c_service.strip(),
                    },
                )

            # Rate limit.
            if response.status_code == 429:
                return (
                    "The booking system is busy right now. "
                    "No appointment was created. "
                    "Please try again shortly."
                )

            # Slot conflict.
            if response.status_code == 409:
                try:
                    detail = response.json().get(
                        "detail",
                        "That time is no longer available.",
                    )
                except Exception:
                    detail = "That time is no longer available."

                data.booking_confirmed = False

                if "Please choose another" in str(detail):
                    return str(detail)
                return f"{detail} Please choose another available time."

            # Invalid input.
            if response.status_code == 400:
                try:
                    detail = response.json().get(
                        "detail",
                        "The booking details are invalid.",
                    )
                except Exception:
                    detail = "The booking details are invalid."

                return str(detail)

            # Backend unavailable.
            if response.status_code >= 500:
                return (
                    "The booking system is temporarily "
                    "unavailable. No appointment was created."
                )

            response.raise_for_status()

            _ = response.json()

            data.booking_active = False
            data.booking_confirmed = False
            data.pending_service = None
            data.pending_date = None
            data.pending_time = None

            return "The appointment has been booked successfully."

        except httpx.TimeoutException:
            logger.warning(
                f"book_appointment request timed out for {canonical_date} {c_time.strip()}. "
                "Checking backend to see if appointment was successfully created..."
            )
            # Reconcile status before reporting failure
            try:
                async with httpx.AsyncClient(timeout=6.0) as verify_client:
                    verify_res = await verify_client.get(
                        f"{API}/appointments",
                        params={"phone": final_phone},
                    )
                    if verify_res.status_code == 200:
                        records = verify_res.json()
                        matching = [
                            a
                            for a in records
                            if a.get("date") == canonical_date
                            and a.get("time") == c_time.strip()
                        ]
                        if matching:
                            data.booking_active = False
                            data.booking_confirmed = False
                            data.pending_service = None
                            data.pending_date = None
                            data.pending_time = None
                            return "The appointment has been booked successfully."
            except Exception as reconcile_exc:
                logger.error(
                    f"Failed to reconcile appointment status after timeout: {reconcile_exc}"
                )

            return (
                "The booking system took too long to respond. "
                "I could not verify whether your appointment was created. "
                "Please let me check your appointments or choose another time."
            )

        except httpx.RequestError as exc:
            logger.error(f"book_appointment request failed: {exc}")

            return (
                "I cannot reach the booking system right now. "
                "No appointment was created."
            )

        except Exception as exc:
            logger.error(f"book_appointment failed: {exc}")

            return (
                "The booking system is temporarily "
                "unavailable. No appointment was created."
            )

        finally:
            log_tool_time(
                "book_appointment",
                start,
            )

    # --------------------------------------------------------
    # FIND
    # --------------------------------------------------------

    @function_tool()
    async def find_appointments(
        self,
        context: RunContext[CallerData],
        phone: Any = "",
    ) -> str:
        """
        Find appointments for the verified caller.
        """

        data = context.session.userdata
        if data.appointment_intent and data.appointment_intent not in {
            "CHECK",
            "RESCHEDULE",
            "CANCEL",
        }:
            return "Existing appointments can only be retrieved for CHECK, RESCHEDULE, or CANCEL."

        if not data.verified:
            return "Identity verification is required before viewing appointments."

        clean_p = clean_argument_value(phone)
        normalized_phone = normalize_phone(clean_p) if clean_p else (data.phone or "")

        # Never search using a different phone number.
        if not normalized_phone or normalized_phone != (data.phone or ""):
            return "The phone number does not match the verified caller."

        start = perf_counter()

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.get(
                    f"{API}/appointments",
                    params={"phone": normalized_phone},
                )

            if response.status_code == 429:
                return "The booking system is busy right now. Please try again shortly."

            if response.status_code >= 500:
                return "The booking system is temporarily unavailable."

            response.raise_for_status()

            items = response.json()

            if not items:
                data.appointment_found = False
                data.selected_appointment_id = None
                data.appointment_ids = []
                data.appointment_confirmed = False

                return "No appointments were found for the verified caller."

            # Store only appointment IDs returned by the backend.
            data.appointment_found = True
            data.appointment_ids = [
                str(item.get("id")) for item in items if item.get("id") is not None
            ]
            data.appointment_confirmed = False

            if len(items) == 1:
                data.selected_appointment_id = str(items[0].get("id"))

            lines = []

            for appointment in items[:3]:
                lines.append(
                    f"{appointment.get('service')} "
                    f"on {appointment.get('date')} "
                    f"at {appointment.get('time')}"
                )

            return "I found " + "; ".join(lines) + "."

        except httpx.TimeoutException:
            return "The booking system took too long to respond."

        except httpx.RequestError as exc:
            logger.error(f"find_appointments request failed: {exc}")

            return "I cannot reach the booking system right now."

        except Exception as exc:
            logger.error(f"find_appointments failed: {exc}")

            return "The booking system is not available right now."

        finally:
            log_tool_time(
                "find_appointments",
                start,
            )

    # --------------------------------------------------------
    # SELECT APPOINTMENT
    # --------------------------------------------------------

    @function_tool()
    async def select_appointment(
        self,
        context: RunContext[CallerData],
        appointment_id: Any,
    ) -> str:
        """
        Select an appointment from retrieved appointments to reschedule or cancel.
        """
        data = context.session.userdata
        if not data.verified:
            return "Identity verification is required."
        if not data.appointment_found:
            return "Find the caller's appointments first."

        appt_id = clean_argument_value(appointment_id)
        if not appt_id:
            return "A valid appointment must be selected."

        allowed_ids = set(data.appointment_ids or [])
        if appt_id not in allowed_ids:
            return "That appointment was not found for the verified caller."

        data.selected_appointment_id = appt_id
        data.appointment_confirmed = False
        action = (
            "rescheduling"
            if data.appointment_intent == "RESCHEDULE"
            else "cancellation"
        )
        return f"Appointment {appt_id} selected. Proceed with {action}."

    # --------------------------------------------------------
    # CONFIRM APPOINTMENT FOR CANCELLATION
    # --------------------------------------------------------

    @function_tool()
    async def confirm_appointment_for_cancellation(
        self,
        context: RunContext[CallerData],
        appointment_id: Any,
    ) -> str:
        """
        Select an appointment only if it was actually found
        for the verified caller.
        """

        data = context.session.userdata

        if data.appointment_intent and data.appointment_intent != "CANCEL":
            return "Appointment selection for cancellation is allowed only during a CANCEL request."

        if not data.verified:
            return "Identity verification is required."

        if not data.appointment_found:
            return "Find the caller's appointments first."

        appointment_id_str = clean_argument_value(appointment_id)

        if not appointment_id_str:
            return "A valid appointment must be selected."

        # Allow only an appointment ID returned by the backend.
        allowed_ids = set(data.appointment_ids or [])

        if appointment_id_str not in allowed_ids:
            return "That appointment was not found for the verified caller."

        data.selected_appointment_id = appointment_id_str
        data.appointment_confirmed = False

        return (
            "Appointment selected. "
            "Ask the caller to confirm cancellation "
            "before cancelling it."
        )

    # --------------------------------------------------------
    # CONFIRM CANCELLATION
    # --------------------------------------------------------

    @function_tool()
    async def confirm_cancellation(
        self,
        context: RunContext[CallerData],
        confirmation: str,
    ) -> str:
        """
        Mark cancellation as explicitly confirmed.
        """

        data = context.session.userdata

        if data.appointment_intent and data.appointment_intent != "CANCEL":
            return "Cancellation confirmation is allowed only during a CANCEL request."

        if not data.verified:
            return "Identity verification is required."

        if not data.appointment_found:
            return "Find the appointment first."

        if not data.selected_appointment_id:
            return "Select the appointment first."

        is_positive = caller_confirmed(context, confirmation, ("cancel it",))

        if not is_positive:
            data.appointment_confirmed = False
            return (
                "Cancellation was not explicitly confirmed. "
                "Ask the caller to confirm and wait for the answer."
            )

        data.appointment_confirmed = True

        return "Cancellation confirmed. The appointment can now be cancelled."

    # --------------------------------------------------------
    # CHANGE
    # --------------------------------------------------------

    @function_tool()
    async def change_appointment(
        self,
        context: RunContext[CallerData],
        appointment_id: Any = "",
        date: Any = "",
        time: Any = "",
    ) -> str:
        """
        Change an appointment after verification.
        """

        data = context.session.userdata

        if data.appointment_intent and data.appointment_intent != "RESCHEDULE":
            return "An appointment can be changed only during a RESCHEDULE request."

        if not data.verified:
            return "Identity verification is required before changing an appointment."

        appt_id = clean_argument_value(appointment_id) or (
            data.selected_appointment_id or ""
        )
        if not appt_id:
            return "No appointment has been selected to change."

        if (
            data.appointment_found
            and data.selected_appointment_id
            and appt_id != data.selected_appointment_id
        ):
            return "That appointment was not selected for the verified caller."

        clean_date = clean_argument_value(date)
        clean_time = clean_argument_value(time)
        if not clean_date or not clean_time:
            return "New date and time are required to reschedule the appointment."

        if hasattr(context, "disallow_interruptions"):
            context.disallow_interruptions()

        start = perf_counter()

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.patch(
                    f"{API}/appointments/{appt_id}",
                    json={
                        "date": resolve_relative_date(clean_date),
                        "time": clean_time,
                    },
                )

            if response.status_code == 404:
                return "I could not find that appointment."

            if response.status_code == 409:
                try:
                    detail = response.json().get(
                        "detail",
                        "That time is not available.",
                    )
                except Exception:
                    detail = "That time is not available."

                return str(detail)

            if response.status_code == 429:
                return "The booking system is busy right now. Please try again shortly."

            if response.status_code >= 500:
                return "The booking system is temporarily unavailable."

            response.raise_for_status()

            return f"The appointment has been changed to {clean_date} at {clean_time}."

        except httpx.TimeoutException:
            return "The booking system took too long to respond."

        except httpx.RequestError as exc:
            logger.error(f"change_appointment request failed: {exc}")

            return "I cannot reach the booking system right now."

        except Exception as exc:
            logger.error(f"change_appointment failed: {exc}")

            return "The booking system is temporarily unavailable."

        finally:
            log_tool_time(
                "change_appointment",
                start,
            )

    # --------------------------------------------------------
    # CANCEL
    # --------------------------------------------------------

    @function_tool()
    async def cancel_appointment(
        self,
        context: RunContext[CallerData],
        appointment_id: Any = "",
    ) -> str:
        """
        Cancel an appointment only after:
        verification -> find -> selection -> confirmation.
        """

        data = context.session.userdata

        if data.appointment_intent and data.appointment_intent != "CANCEL":
            return "An appointment can be cancelled only during a CANCEL request."

        if not data.verified:
            return "Identity verification is required before cancelling an appointment."

        if not data.appointment_found:
            return "Find the caller's appointment first."

        appt_id = clean_argument_value(appointment_id) or (
            data.selected_appointment_id or ""
        )
        if not data.selected_appointment_id or not appt_id:
            return "Select the appointment before cancelling it."

        if appt_id != data.selected_appointment_id:
            return "That appointment was not selected for cancellation."

        if not data.appointment_confirmed:
            return (
                "The caller has not confirmed the cancellation. "
                "Ask for explicit confirmation first."
            )

        if hasattr(context, "disallow_interruptions"):
            context.disallow_interruptions()

        start = perf_counter()

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.delete(f"{API}/appointments/{appt_id}")

            if response.status_code == 404:
                return "I could not find that appointment."

            if response.status_code == 429:
                return "The booking system is busy right now. Please try again shortly."

            if response.status_code >= 500:
                return (
                    "The booking system is temporarily "
                    "unavailable. The appointment was not cancelled."
                )

            response.raise_for_status()

            # Reset cancellation state.
            data.appointment_found = False
            data.selected_appointment_id = None
            data.appointment_ids = []
            data.appointment_confirmed = False

            return "The appointment has been cancelled successfully."

        except httpx.TimeoutException:
            return (
                "The booking system took too long to respond. "
                "The appointment was not cancelled."
            )

        except httpx.RequestError as exc:
            logger.error(f"cancel_appointment request failed: {exc}")

            return (
                "I cannot reach the booking system right now. "
                "The appointment was not cancelled."
            )

        except Exception as exc:
            logger.error(f"cancel_appointment failed: {exc}")

            return (
                "The booking system is temporarily "
                "unavailable. The appointment was not cancelled."
            )

        finally:
            log_tool_time(
                "cancel_appointment",
                start,
            )

    # --------------------------------------------------------
    # SLOW SYSTEM
    # --------------------------------------------------------

    @function_tool()
    async def check_slow_system(
        self,
        context: RunContext[CallerData],
    ) -> str:
        """Run the slow system check."""

        start = perf_counter()

        try:
            # Not awaited: awaiting would delay the API call until the
            # sentence finishes playing (Tutorial 3, Scenario 4).
            context.session.say(
                "One moment, let me check that for you.",
                allow_interruptions=False,
            )

            try:
                async with httpx.AsyncClient(timeout=6.0) as client:
                    response = await client.get(f"{API}/slow")

                    response.raise_for_status()

            except Exception as exc:
                logger.error(f"check_slow_system failed: {exc}")

                return "The system is not available right now."

            return "The check is done. The system is working."

        finally:
            log_tool_time(
                "check_slow_system",
                start,
            )

    # --------------------------------------------------------
    # BROKEN SYSTEM
    # --------------------------------------------------------

    @function_tool()
    async def check_broken_system(
        self,
        context: RunContext[CallerData],
    ) -> str:
        """Run the broken system check."""

        start = perf_counter()

        try:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{API}/broken")

                    response.raise_for_status()

            except Exception as exc:
                logger.error(f"check_broken_system failed: {exc}")

                return "The system is not working right now."

            return "The check is done. The system is working."

        finally:
            log_tool_time(
                "check_broken_system",
                start,
            )

    # --------------------------------------------------------
    # BOOKING -> RECEPTION
    # --------------------------------------------------------

    @function_tool()
    async def handoff_to_reception(
        self,
        context: RunContext[CallerData],
    ):

        data = context.session.userdata

        data.booking_active = False

        return (
            ReceptionAgent(returning=True, chat_ctx=carry_context(self)),
            "Return to reception and ask whether the caller needs anything else.",
        )


# ============================================================
# BILLING AGENT
# ============================================================


class BillingAgent(Agent):
    def __init__(self, chat_ctx=None) -> None:

        super().__init__(
            instructions=f"{CLINIC_PROMPT}\n\n{BILLING_PROMPT}",
            llm=build_llm(),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):

        await self.session.generate_reply(
            instructions=(
                "The caller was just handed over from reception for billing and "
                "is already verified. Do not greet again. If they already asked "
                "about their bill, call get_bill and tell them the result. "
                "Otherwise ask what billing help they need."
            )
        )

    # --------------------------------------------------------
    # BILL
    # --------------------------------------------------------

    @function_tool()
    async def get_bill(
        self,
        context: RunContext[CallerData],
    ) -> str:
        """Get the verified caller's bill."""

        if not is_verified(context):
            return (
                "Identity verification is required before viewing billing information."
            )

        phone = context.session.userdata.phone

        if not phone:
            return "A verified phone number is required."

        start = perf_counter()

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.get(
                    f"{API}/bills",
                    params={"phone": phone},
                )

            if response.status_code == 429:
                return "The billing system is busy right now. Please try again shortly."

            if response.status_code >= 500:
                return "The billing system is temporarily unavailable."

            response.raise_for_status()

            result = response.json()

            if not result or not result.get("amount") or not result.get("due_date"):
                return "No billing information is available for the verified caller."

            return (
                f"Your current bill is "
                f"{result.get('amount')}, "
                f"due on {result.get('due_date')}."
            )

        except httpx.TimeoutException:
            return "The billing system took too long to respond."

        except httpx.RequestError as exc:
            logger.error(f"get_bill request failed: {exc}")

            return "I cannot reach the billing system right now."

        except Exception as exc:
            logger.error(f"get_bill failed: {exc}")

            return "The billing system is not available right now."

        finally:
            log_tool_time(
                "get_bill",
                start,
            )

    # --------------------------------------------------------
    # BILLING -> RECEPTION
    # --------------------------------------------------------

    @function_tool()
    async def handoff_to_reception(
        self,
        context: RunContext[CallerData],
    ):

        return (
            ReceptionAgent(returning=True, chat_ctx=carry_context(self)),
            "Return to reception and ask whether the caller needs anything else.",
        )


# ============================================================
# CAPSTONE: PRESCRIPTION AGENT
# ============================================================


class PrescriptionAgent(Agent):
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions=f"{CLINIC_PROMPT}\n\n{PRESCRIPTION_PROMPT}",
            llm=build_llm(),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "You are the prescription assistant speaking to the verified caller. "
                "Check the latest caller message. If they requested a refill for a specific medication (e.g. Amoxicillin or Lisinopril), call request_prescription_refill or ask for confirmation. Otherwise, call get_prescriptions to look up active prescriptions for the caller and ask which medication they would like to refill."
            )
        )

    @function_tool()
    async def get_prescriptions(
        self,
        context: RunContext[CallerData],
    ) -> str:
        """Get active prescriptions for the verified caller."""
        data = context.session.userdata
        if not data.verified or not data.phone:
            return "Identity verification is required before viewing prescriptions."

        start = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.get(
                    f"{API}/prescriptions",
                    params={"phone": data.phone},
                )
            if response.status_code != 200:
                return "Could not retrieve prescriptions at this time."
            items = response.json()
            if not items:
                return "No active prescriptions found on file for this patient."
            lines = [
                f"{rx['medication']} ({rx['refills_remaining']} refills remaining, prescribed by {rx['prescribing_doctor']})"
                for rx in items
            ]
            return "Active prescriptions on file: " + "; ".join(lines) + "."
        except Exception as exc:
            logger.error(f"get_prescriptions failed: {exc}")
            return "The prescription system is temporarily unavailable."
        finally:
            log_tool_time("get_prescriptions", start)

    @function_tool()
    async def request_prescription_refill(
        self,
        context: RunContext[CallerData],
        medication: str,
        pharmacy: str = "CityCare In-House Pharmacy",
    ) -> str:
        """Submit a prescription refill request for the verified caller."""
        data = context.session.userdata
        if not data.verified or not data.phone:
            return "Identity verification is required before requesting a refill."

        med = (medication or "").strip()
        if not med:
            return "Please specify the medication you would like to refill."

        if med.casefold() in {m.casefold() for m in data.refilled_medications}:
            return (
                f"A refill for {med} was already submitted in this call. "
                "Do not submit it again."
            )

        pharmacy_name = pharmacy.strip() if pharmacy else "CityCare In-House Pharmacy"

        # Step 1: remember the request and make the model ask the caller.
        if not (
            data.prescription_active
            and (data.prescription_medication or "").casefold() == med.casefold()
        ):
            data.prescription_active = True
            data.prescription_medication = med
            data.prescription_pharmacy = pharmacy_name
            return (
                "Not submitted yet. Ask the caller to confirm a refill of "
                f"{med} at {pharmacy_name}. Wait for a clear yes, then call "
                "this tool again with the same medication."
            )

        # Step 2: only submit if the caller really said yes.
        heard = last_user_text(context)
        if heard is not None and not is_affirmative(heard):
            return (
                "The caller has not confirmed yet. Ask them to confirm the "
                "refill and wait for a clear yes."
            )

        if hasattr(context, "disallow_interruptions"):
            context.disallow_interruptions()

        start = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.post(
                    f"{API}/prescriptions/refill",
                    json={
                        "name": data.name or "Patient",
                        "phone": data.phone,
                        "medication": med,
                        "pharmacy": pharmacy_name,
                    },
                )
            if response.status_code == 404:
                return f"No active prescription found matching '{med}'. Please check with your doctor."
            if response.status_code == 400:
                detail = response.json().get("detail", "No refills remaining.")
                return f"{detail}"
            if response.status_code != 200:
                return "Could not process refill request at this time."

            result = response.json()
            data.refilled_medications.append(med)
            data.prescription_active = False
            data.prescription_medication = None
            return (
                f"Refill approved for {result['medication']} at {result['pharmacy']}. "
                f"Reference ID is {result['refill_id']}. Estimated ready: {result['estimated_ready']}."
            )
        except Exception as exc:
            logger.error(f"request_prescription_refill failed: {exc}")
            return "The prescription refill system is temporarily unavailable."
        finally:
            log_tool_time("request_prescription_refill", start)

    @function_tool()
    async def handoff_to_reception(
        self,
        context: RunContext[CallerData],
    ):
        """Return caller to reception."""
        return (
            ReceptionAgent(returning=True, chat_ctx=carry_context(self)),
            "Returning to reception. Ask if they need anything else.",
        )


# ============================================================
# PII REDACTION
# ============================================================


def redact_pii(text: str) -> str:
    """
    Redact Personally Identifiable Information (PII) including phone numbers
    and dates of birth from transcripts and logs.
    """
    if not isinstance(text, str):
        return text

    # Redact phone numbers (with optional country code and separators)
    text = re.sub(r"(\+?91[\s-]?)?[6-9]\d{9}", "[REDACTED_PHONE]", text)
    text = re.sub(r"\b\d{10}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[REDACTED_PHONE]", text)

    # Redact Date of Birth patterns
    text = re.sub(
        r"\b(19\d{2}|20\d{2})[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])\b",
        "[REDACTED_DOB]",
        text,
    )
    text = re.sub(
        r"\b(0[1-9]|[12]\d|3[01])[-/.](0[1-9]|1[0-2])[-/.](19\d{2}|20\d{2})\b",
        "[REDACTED_DOB]",
        text,
    )

    return text


def redact_report_pii(data: Any) -> Any:
    """Recursively redact PII from session report structures."""
    if isinstance(data, str):
        return redact_pii(data)
    elif isinstance(data, dict):
        return {k: redact_report_pii(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_report_pii(item) for item in data]
    return data


# ============================================================
# SERVER
# ============================================================


def prewarm(proc: JobProcess):
    """
    Prewarm step: Build RAG index in a background thread to prevent
    blocking process initialization beyond LiveKit's 10s IPC timeout.
    """
    logger.info("Prewarming CityCare worker process...")
    import threading

    threading.Thread(target=rag.build_index, daemon=True).start()
    proc.userdata["rag_ready"] = True


server = AgentServer(setup_fnc=prewarm)


async def on_session_end(ctx: JobContext) -> None:
    """Save session report to reports/ with PII redacted at the end of each session."""

    def _save_report() -> None:
        report = ctx.make_session_report().to_dict()
        redacted_report = redact_report_pii(report)
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        room_name = getattr(ctx.room, "name", "unknown") if ctx.room else "unknown"
        report_file = reports_dir / f"session_{room_name}_{stamp}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(redacted_report, f, indent=2)
        logger.info(f"Saved PII-redacted session report to {report_file}")

    try:
        await asyncio.to_thread(_save_report)
    except Exception as exc:
        logger.error(f"Failed to save session report: {exc}")


@server.rtc_session(
    agent_name="citycare",
    on_session_end=on_session_end,
)
async def my_agent(
    ctx: JobContext,
):

    ctx.log_context_fields = {"room": ctx.room.name}

    print(f"Tool timings will be saved to: {TIMING_FILE}")

    print(
        f"LLM mode: {LLM_MODE} "
        f"(Primary: {GOOGLE_PRIMARY_MODEL}, fallback: {GROQ_FALLBACK_MODEL})"
    )
    # --------------------------------------------------------
    # BUILD RAG INDEX (if not already prewarmed)
    # --------------------------------------------------------

    try:
        if not getattr(ctx.proc, "userdata", {}).get("rag_ready", False):
            await asyncio.to_thread(rag.build_index)
    except Exception as exc:
        logger.error(f"RAG index build failed: {exc}")

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    lk_key = os.getenv("LIVEKIT_API_KEY")
    lk_secret = os.getenv("LIVEKIT_API_SECRET")

    session = AgentSession[CallerData](
        userdata=CallerData(),
        max_tool_steps=MAX_TOOL_STEPS,
        llm=build_llm(),
        stt=stt.MultiSpeakerAdapter(
            stt=deepgram.STT(
                model=VOICE_SETTINGS["stt"]["model"],
                language=VOICE_SETTINGS["stt"]["language"],
                enable_diarization=True,
                api_key=os.getenv("DEEPGRAM_API_KEY"),
            ),
            detect_primary_speaker=True,
            suppress_background_speaker=False,
            background_format="[Background speaker] {text}",
        ),
        stt_context_options=STTContextOptions(
            keyterms=[
                "CityCare Clinic",
                "CityCare",
                "Dr. Smith",
                "Dr. Patel",
                "Dr. Williams",
                "general check-up",
                "blood tests",
                "children's doctor",
                "Amoxicillin",
                "Lisinopril",
                "Metformin",
                "prescription refill",
            ],
            keyterm_detection={"enabled": False},
        ),
        tts=cartesia.TTS(
            model=VOICE_SETTINGS["tts"]["model"],
            voice=VOICE_SETTINGS["tts"]["voice"],
            language=VOICE_SETTINGS["tts"]["language"],
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(
                version="v1-mini",
                local_fallback=True,
                api_key=lk_key,
                api_secret=lk_secret,
            ),
            endpointing=VOICE_SETTINGS["endpointing"],
            interruption=VOICE_SETTINGS["interruption"],
            preemptive_generation=VOICE_SETTINGS["preemptive_generation"],
        ),
        expressive=False,
    )

    # --------------------------------------------------------
    # METRICS & CHAT CONTEXT MANAGEMENT
    # --------------------------------------------------------

    _truncate_lock = asyncio.Lock()
    _bg_tasks: set[asyncio.Task[Any]] = set()

    async def _safe_truncate_history() -> None:
        if _truncate_lock.locked():
            return
        async with _truncate_lock:
            agent = session.current_agent
            if not agent or not hasattr(agent, "chat_ctx") or agent.chat_ctx is None:
                return
            current_ctx = agent.chat_ctx
            threshold = MAX_HISTORY_ITEMS + 4
            if len(current_ctx.items) > threshold:
                # Use copy() to obtain a mutable copy, truncate it, and update agent asynchronously
                new_ctx = current_ctx.copy()
                new_ctx.truncate(max_items=MAX_HISTORY_ITEMS + 2)
                try:
                    await agent.update_chat_ctx(new_ctx)
                except Exception as exc:
                    logger.warning(f"Failed to update chat context: {exc}")

    @session.on("conversation_item_added")
    def on_item(
        ev: ConversationItemAddedEvent,
    ):
        # Dynamically truncate chat context safely without mutating read-only instance
        t = asyncio.create_task(_safe_truncate_history())
        _bg_tasks.add(t)
        t.add_done_callback(_bg_tasks.discard)

        if not isinstance(
            ev.item,
            ChatMessage,
        ):
            return

        metrics = ev.item.metrics

        if ev.item.role == "user":
            print(
                f"USER "
                f"end_of_turn="
                f"{metrics.get('end_of_turn_delay')} "
                f"stt="
                f"{metrics.get('transcription_delay')}"
            )

        elif ev.item.role == "assistant":
            print(
                f"AGENT "
                f"e2e="
                f"{metrics.get('e2e_latency')} "
                f"llm_ttft="
                f"{metrics.get('llm_node_ttft')} "
                f"tts_ttfb="
                f"{metrics.get('tts_node_ttfb')}"
            )

    @session.on("user_interruption_detected")
    def on_interruption(ev):
        print(f"INTERRUPTED probability={getattr(ev, 'probability', None)}")

    @session.on("agent_false_interruption")
    def on_false_interruption(ev):
        print("FALSE INTERRUPTION agent resumes")

    # --------------------------------------------------------
    # CONNECT TO ROOM FIRST, THEN START SESSION
    # --------------------------------------------------------
    await ctx.connect()

    await session.start(
        agent=ReceptionAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=build_noise_cancellation(),
            ),
        ),
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    cli.run_app(server)
