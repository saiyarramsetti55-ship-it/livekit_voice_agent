"""
Stage 2 — Baseline Latency Logger

Reads tool_times.csv and any session JSON reports, prints a
latency summary table for: end_of_turn_delay, llm_node_ttft,
tts_node_ttfb, and tool durations.

Usage:
    uv run python latency_baseline.py
"""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ── Tool durations from tool_times.csv ────────────────────────────────────────


def tool_stats() -> None:
    csv_file = ROOT / "src" / "tool_times.csv"
    if not csv_file.exists():
        print("tool_times.csv not found — run the agent and make some calls first.\n")
        return

    rows: dict[str, list[float]] = {}
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tool = row["tool"]
            ms = float(row["ms"])
            rows.setdefault(tool, []).append(ms)

    print("── Tool Durations (ms) ──────────────────────────────────")
    print(f"{'Tool':<30} {'N':>4} {'Avg':>8} {'P50':>8} {'P95':>8}")
    print("-" * 60)
    for tool, times in sorted(rows.items()):
        times.sort()
        avg = statistics.mean(times)
        p50 = statistics.median(times)
        p95 = times[int(len(times) * 0.95)] if len(times) >= 2 else times[-1]
        print(f"{tool:<30} {len(times):>4} {avg:>8.0f} {p50:>8.0f} {p95:>8.0f}")
    print()


# ── Session-level metrics from reports/*.json ──────────────────────────────────


def session_stats() -> None:
    reports_dir = ROOT / "reports"
    if not reports_dir.exists():
        print("reports/ directory not found.\n")
        return

    e2e: list[float] = []
    llm_ttft: list[float] = []
    tts_ttfb: list[float] = []
    eot: list[float] = []

    for report_file in sorted(reports_dir.glob("*.json"))[-10:]:  # last 10 sessions
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        for item in data.get("conversation_items", []):
            m = item.get("metrics") or {}
            if item.get("role") == "assistant":
                if m.get("e2e_latency"):
                    e2e.append(float(m["e2e_latency"]))
                if m.get("llm_node_ttft"):
                    llm_ttft.append(float(m["llm_node_ttft"]))
                if m.get("tts_node_ttfb"):
                    tts_ttfb.append(float(m["tts_node_ttfb"]))
            elif item.get("role") == "user":
                if m.get("end_of_turn_delay"):
                    eot.append(float(m["end_of_turn_delay"]))

    def fmt(label: str, values: list[float]) -> None:
        if not values:
            print(f"  {label:<25} no data")
            return
        avg = statistics.mean(values)
        p50 = statistics.median(values)
        p95 = (
            sorted(values)[int(len(values) * 0.95)] if len(values) >= 2 else values[-1]
        )
        print(
            f"  {label:<25} n={len(values):>4}  avg={avg * 1000:>6.0f}ms  p50={p50 * 1000:>6.0f}ms  p95={p95 * 1000:>6.0f}ms"
        )

    print("── Session Latency Metrics (last 10 sessions) ───────────")
    fmt("end_of_turn_delay", eot)
    fmt("llm_node_ttft", llm_ttft)
    fmt("tts_node_ttfb", tts_ttfb)
    fmt("e2e_latency", e2e)
    print()


if __name__ == "__main__":
    print("\n=== CityCare Clinic — Latency Baseline ===\n")
    tool_stats()
    session_stats()
    print(
        "Tip: run `uv run python latency_baseline.py` again after voice tuning to compare.\n"
    )
