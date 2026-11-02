import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python scripts/analyze_logs.py <log-file>")
    raise SystemExit(1)
log_file = Path(sys.argv[1])
if not log_file.is_file():
    print(f"Log file not found: {log_file}")
    raise SystemExit(1)
text = log_file.read_text(encoding="utf-8-sig", errors="replace")
lines = text.splitlines()
blocked = [
    line for line in lines if re.search(r"event loop blocked for \d+ms", line, re.I)
]
vad = [line for line in lines if "slower than realtime" in line.lower()]
errors = [line for line in lines if re.search(r"\b(error|critical)\b", line, re.I)]
exceptions = [line for line in lines if "exception" in line.lower()]
booking = [line for line in lines if "book_appointment" in line.lower()]
print("\n========== CITYCARE PERFORMANCE REPORT ==========")
print(f"Log file: {log_file.resolve()}")
print(f"Total lines: {len(lines)}")
print(f"\nEvent-loop blocked warnings: {len(blocked)}")
for line in blocked:
    duration = re.search(r"blocked for (\d+)ms", line, re.I)
    value = f"{duration.group(1)} ms" if duration else "duration unknown"
    print(f"  [{value}] {line[:220]}")
print(f"\nVAD slower than realtime: {len(vad)}")
print(f"Error/critical lines: {len(errors)}")
print(f"Exception mentions: {len(exceptions)}")
print(f"Booking tool mentions: {len(booking)}")
for label, pattern in {
    "End-to-end response": r"e2e.*?\d+(?:\.\d+)?\s*s",
    "LLM first token": r"TTFT.*?\d+(?:\.\d+)?\s*s",
    "TTS first audio": r"TTS TTFB.*?\d+(?:\.\d+)?\s*s",
}.items():
    matches = [line.strip() for line in lines if re.search(pattern, line, re.I)]
    print(f"\n{label} timing entries: {len(matches)}")
    for line in matches[-5:]:
        print("  " + line[:220])
print("\n=================================================\n")
