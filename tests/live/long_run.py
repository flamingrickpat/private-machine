"""Long-run harness: a simulated user texts the companion across N virtual hours.

Virtual time is a ``tests.support.ManualClock``. Each tick advances it, steps
every bus module, and lets the simulated user write when awake. The sim user is
one model call per message, driven by a plain-text style card. Nothing here
judges quality; three mechanical checks come out:

* overnight messages: companion lines emitted while the sim user is asleep
* memory growth: process RSS slope in MB per virtual hour
* repeats: exact or near-identical companion lines

Run from the repo root::

    python -m tests.live.long_run --hours 24 --card tests/live/style_cards/example.txt
    python -m tests.live.long_run --stub --hours 1      # no model, dummy cognition

Model endpoint: the repo's own config (``PM_MODEL_CONFIG``, else
``config.local.yaml``, else ``config.example.yaml``). Writes
``<out>/transcript.jsonl`` and ``<out>/report.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

from tests.support import ManualClock, Recorder, fresh_ghost, load_seed_state
from workflows.companion import build_app

DEFAULT_CARD = Path(__file__).resolve().parent / "style_cards" / "example.txt"

SIM_SYSTEM = (
    "You play one real person texting a companion character on a phone. You are not an assistant. "
    "Write exactly like the style card says: length, voice, habits. Improvise on what the "
    "conversation gives you. You see only the companion's messages. "
    "Reply with the message text only, no quotes, no stage directions."
)


# ------------------------------------------------------------------ helpers

def rss_mb() -> float:
    """Current process resident memory in MB. psutil, else /proc, else peak RSS."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (OSError, ValueError, AttributeError):
        pass
    import resource  # Unix; ru_maxrss is a peak, so it only ever rises
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1e6 if sys.platform == "darwin" else 1e3)


def is_asleep(moment: datetime, sleep_from: int, wake_at: int) -> bool:
    """Asleep from ``sleep_from`` o'clock to ``wake_at`` o'clock (wraps midnight)."""
    hour = moment.hour
    if sleep_from <= wake_at:
        return sleep_from <= hour < wake_at
    return hour >= sleep_from or hour < wake_at


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def repeat_stats(lines: list[str], near: float = 0.9, window: int = 50) -> dict:
    """Count companion lines that repeat one of the previous ``window`` lines."""
    exact = near_count = 0
    seen: list[str] = []
    examples: list[str] = []
    for line in lines:
        norm = _norm(line)
        if not norm:
            continue
        recent = seen[-window:]
        if norm in recent:
            exact += 1
            examples.append(line[:80])
        elif any(SequenceMatcher(None, norm, old).ratio() >= near for old in recent):
            near_count += 1
            examples.append(line[:80])
        seen.append(norm)
    total = len(seen)
    return {"lines": total, "exact": exact, "near": near_count,
            "fraction": (exact + near_count) / total if total else 0.0, "examples": examples[:5]}


def rss_slope(samples: list[tuple[float, float]], warmup_hours: float) -> float:
    """Least-squares MB per virtual hour over samples at or after the warm-up."""
    pts = [(h, mb) for h, mb in samples if h >= warmup_hours] or samples
    if len(pts) < 2:
        return 0.0
    mean_h = sum(h for h, _ in pts) / len(pts)
    mean_m = sum(m for _, m in pts) / len(pts)
    var = sum((h - mean_h) ** 2 for h, _ in pts)
    return sum((h - mean_h) * (m - mean_m) for h, m in pts) / var if var else 0.0


# --------------------------------------------------------------- sim user

class SimUser:
    """One model call per message. ``complete`` is a callable(messages, **gen) -> str."""

    def __init__(self, card: str, complete):
        self.card, self.complete = card, complete

    def write(self, transcript: list[dict], now: datetime) -> str:
        recent = "\n".join(
            f'{"me" if t["who"] == "user" else "her"}: {t["text"][:400]}' for t in transcript[-16:]
        ) or "(nothing said yet; you start)"
        prompt = (f"STYLE CARD:\n{self.card}\n\nNOW: {now:%A %H:%M}\n\n"
                  f"CONVERSATION:\n{recent}\n\nWrite your next message.")
        text = (self.complete([{"role": "system", "content": SIM_SYSTEM},
                               {"role": "user", "content": prompt}], temperature=0.8) or "").strip()
        return text or "hey"


class StubUser:
    """No-model sim user for the dry run."""

    def __init__(self):
        self.count = 0

    def write(self, transcript: list[dict], now: datetime) -> str:
        self.count += 1
        return f"stub message {self.count}"


# ------------------------------------------------------------------ driver

@dataclass
class Params:
    hours: float = 24.0
    start_hour: int = 8            # virtual wall-clock hour the run starts at
    sleep_from: int = 23           # sim user asleep from this hour ...
    wake_at: int = 7               # ... until this hour
    tick_seconds: float = 60.0     # virtual seconds per driver tick
    thought_period: float = 300.0  # virtual seconds between companion inner-thought ticks
    min_gap_minutes: float = 20.0  # sim user writes every min..max minutes while awake
    max_gap_minutes: float = 120.0
    seed: int = 1
    stub: bool = False
    card: Path = DEFAULT_CARD
    out: Path | None = None
    # thresholds
    max_overnight: int = 0
    max_rss_mb_per_hour: float = 20.0
    warmup_hours: float = 1.0
    max_repeat_fraction: float = 0.10
    near_ratio: float = 0.9
    min_lines_for_repeat_check: int = 8


@dataclass
class Result:
    transcript: list[dict] = field(default_factory=list)
    rss: list[tuple[float, float]] = field(default_factory=list)
    overnight: int = 0
    companion_lines: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def _model_complete():
    from pm.llm.activation import ModelActivation
    from pm.llm.config import load_config, resolve_connection
    from pm.llm.stage_model import LazyStageModel
    default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
    connection = resolve_connection(load_config(os.environ.get("PM_MODEL_CONFIG", default)))
    stage_model = LazyStageModel(connection, ModelActivation(), live_test=True)
    return stage_model, stage_model.complete


def run(params: Params) -> Result:
    rng = random.Random(params.seed)
    out = params.out or Path(tempfile.mkdtemp(prefix="pm-longrun-"))
    out.mkdir(parents=True, exist_ok=True)
    result = Result()
    clock = ManualClock()
    start = datetime(2030, 1, 7, params.start_hour, 0, 0)  # a Monday; the date is arbitrary

    ghost = fresh_ghost(load_seed_state())
    ghost.checkpoint = out / "ghost.pickle"
    if params.stub:
        app = build_app(None, out / "runtime", thought_period=params.thought_period,
                        clock=clock, ghost=ghost, emit_reply=lambda _t: None)
        user = StubUser()
    else:
        stage_model, complete = _model_complete()
        app = build_app(None, out / "runtime", thought_period=params.thought_period, clock=clock,
                        ghost=ghost, cognition="real", stage_model=stage_model,
                        emit_reply=lambda _t: None)
        user = SimUser(params.card.read_text(encoding="utf-8"), complete)

    replies = Recorder(app.bus, "dialogue.reply")
    seen_replies = 0
    log = (out / "transcript.jsonl").open("w", encoding="utf-8")

    def emit(who: str, now: datetime, text: str, **extra) -> None:
        row = {"ts": now.isoformat(timespec="seconds"), "who": who, "text": text, **extra}
        result.transcript.append(row)
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()

    def drain() -> None:
        for _ in range(50):  # bounded: modules hand work to each other until idle
            if not any([module.step() for module in app.modules]):
                return
        raise AssertionError("Modules did not go idle within 50 passes.")

    total_ticks = int(params.hours * 3600 / params.tick_seconds)
    next_write = 0.0  # virtual seconds; the sim user opens the chat when first awake
    next_rss = 0.0
    result.rss.append((0.0, rss_mb()))
    try:
        for _ in range(total_ticks):
            clock.advance(params.tick_seconds)
            now = start + timedelta(seconds=clock.value)
            asleep = is_asleep(now, params.sleep_from, params.wake_at)
            if not asleep and clock.value >= next_write:
                text = user.write(result.transcript, now)
                emit("user", now, text)
                app.user_input.submit(text)
                next_write = clock.value + rng.uniform(params.min_gap_minutes,
                                                       params.max_gap_minutes) * 60
            drain()
            app.raise_if_error()
            for event in replies.events[seen_replies:]:
                text = str(ghost.get_knoxel_by_id(event.payload["knoxel_id"]).content)
                emit("companion", now, text, asleep=asleep)
                result.companion_lines.append(text)
                if asleep:
                    result.overnight += 1
            seen_replies = len(replies.events)
            if clock.value >= next_rss + 3600:
                next_rss = clock.value
                result.rss.append((clock.value / 3600, rss_mb()))
    finally:
        log.close()
        app.shutdown()
        supervisor = getattr(app, "supervisor", None)
        if supervisor is not None:
            for run_id, receipt in supervisor.receipts().items():
                if receipt["state"] in {"accepted", "running", "waiting_for_input"}:
                    supervisor.terminate(run_id)
            supervisor.join_all()

    slope = rss_slope(result.rss, params.warmup_hours)
    repeats = repeat_stats(result.companion_lines, params.near_ratio)
    result.report = {
        "virtual_hours": params.hours, "ticks": total_ticks,
        "user_messages": sum(1 for r in result.transcript if r["who"] == "user"),
        "companion_messages": len(result.companion_lines),
        "overnight": {"count": result.overnight, "max": params.max_overnight},
        "memory": {"mb_per_hour": round(slope, 3), "max": params.max_rss_mb_per_hour,
                   "start_mb": round(result.rss[0][1], 1), "end_mb": round(result.rss[-1][1], 1),
                   "samples": len(result.rss)},
        "repeats": {**repeats, "max_fraction": params.max_repeat_fraction},
    }
    if result.overnight > params.max_overnight:
        result.failures.append(
            f"overnight: {result.overnight} companion messages while the user slept "
            f"({params.sleep_from:02d}:00-{params.wake_at:02d}:00), max {params.max_overnight}")
    if slope > params.max_rss_mb_per_hour:
        result.failures.append(f"memory: {slope:.2f} MB/h RSS growth, max {params.max_rss_mb_per_hour}")
    if (repeats["lines"] >= params.min_lines_for_repeat_check
            and repeats["fraction"] > params.max_repeat_fraction):
        result.failures.append(
            f"repeats: {repeats['fraction']:.0%} of {repeats['lines']} companion lines repeat "
            f"(max {params.max_repeat_fraction:.0%}); e.g. {repeats['examples'][:2]}")
    (out / "report.json").write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    d = Params()
    p.add_argument("--hours", type=float, default=d.hours)
    p.add_argument("--card", type=Path, default=d.card)
    p.add_argument("--out", type=Path)
    p.add_argument("--stub", action="store_true", help="dummy cognition and scripted user; no model")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--start-hour", type=int, default=d.start_hour)
    p.add_argument("--sleep-from", type=int, default=d.sleep_from)
    p.add_argument("--wake-at", type=int, default=d.wake_at)
    p.add_argument("--tick-seconds", type=float, default=d.tick_seconds)
    p.add_argument("--thought-period", type=float, default=d.thought_period)
    p.add_argument("--max-overnight", type=int, default=d.max_overnight)
    p.add_argument("--max-rss-mb-per-hour", type=float, default=d.max_rss_mb_per_hour)
    p.add_argument("--warmup-hours", type=float, default=d.warmup_hours)
    p.add_argument("--max-repeat-fraction", type=float, default=d.max_repeat_fraction)
    a = p.parse_args(argv)
    params = Params(hours=a.hours, card=a.card, out=a.out, stub=a.stub, seed=a.seed,
                    start_hour=a.start_hour, sleep_from=a.sleep_from, wake_at=a.wake_at,
                    tick_seconds=a.tick_seconds, thought_period=a.thought_period,
                    max_overnight=a.max_overnight, max_rss_mb_per_hour=a.max_rss_mb_per_hour,
                    warmup_hours=a.warmup_hours, max_repeat_fraction=a.max_repeat_fraction)
    result = run(params)
    print(json.dumps(result.report, indent=2))
    for failure in result.failures:
        print("FAIL", failure)
    print("PASS" if not result.failures else "FAILED")
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
