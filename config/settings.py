import os

from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv(".env")

API = os.getenv("API_URL", "http://127.0.0.1:8010").strip()

TOP_K = 2

# LLM configuration
LLM_MODE = os.getenv("LLM_MODE", "plain").strip()

GOOGLE_PRIMARY_MODEL = (
    os.getenv("GOOGLE_PRIMARY_MODEL")
    or os.getenv("GEMINI_MODEL")
    or "gemini-3.5-flash-lite"
).strip()

GROQ_FALLBACK_MODEL = os.getenv(
    "GROQ_FALLBACK_MODEL",
    "openai/gpt-oss-120b",
).strip()

# Compatibility aliases
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", GOOGLE_PRIMARY_MODEL).strip()
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", GROQ_FALLBACK_MODEL).strip()
GROQ_PRIMARY_MODEL = PRIMARY_LLM_MODEL

LLM_ATTEMPT_TIMEOUT = float(os.getenv("LLM_ATTEMPT_TIMEOUT", "10.0"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "0"))
LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "250"))
MAX_HISTORY_ITEMS = int(os.getenv("MAX_HISTORY_ITEMS", "6"))

# Timeouts
HTTP_TIMEOUT = 5.0
RAG_TIMEOUT = 5.0

# Tool configuration
MAX_TOOL_STEPS = 5
