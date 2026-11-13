"""Record and replay deterministic cognitive situations.

Commands::

    python -m tests.replay list
    python -m tests.replay record --scenario private-continuity --samples 3
    python -m tests.replay replay --snapshot .runtime/replay/private-continuity-20261114
    python -m tests.replay compare --snapshot .runtime/replay/private-continuity-20261114

``record`` spends provider money: every sample is one independent live run of
the scenario. ``replay`` spends nothing and needs no reachable endpoint; it
serves the recorded answers over the recorded checkpoint.

A snapshot directory holds one subdirectory per sample. Each sample holds
``calls.jsonl`` (the recorded model calls), ``manifest.json`` (what produced
them), ``ghost.pickle`` (the situation the run started from), and
``outcome.json`` (what the run did).

Replay writes ``replay-<sample>.json`` beside the recording and never modifies
the recording itself.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pm import vclock
from pm.memory.store import MemoryStore
from pm.utils.operating_system import OperatingSystem

from tests.replay.determinism import deterministic_identities
from tests.replay.model import ReplayDivergence
from tests.replay.runtime import ROOT, ScenarioRuntime, open_runtime
from tests.replay.scenarios import SCENARIOS, Scenario, ScenarioOutcome, scenario_named
from tests.replay.script import RecordingManifest


DEFAULT_ROOT = ROOT / ".runtime/replay"

#: The situation modules whose content decides whether a recording is still
#: valid for the current code. A recording made against different ones is not
#: evidence about today's behaviour.
SOURCE_PATHS = (
    "workflows/inner_thought",
    "workflows/cognitive_control",
    "workflows/focus_management",
    "workflows/input_triage",
)


class SampleRecord(BaseModel):
    """One sample's manifest, retained beside its recording."""

    model_config = ConfigDict(extra="forbid")

    sample: int = Field(ge=1, description="One-based sample index within the snapshot.")
    directory: str = Field(description="Sample directory, relative to the snapshot.")
    manifest: RecordingManifest = Field(description="What produced this sample.")


class ReplayResult(BaseModel):
    """What one replay of one sample observed.

    The full typed outcome is written beside this record as
    ``replay-outcome-<sample>.json``. This model keeps the comparable summary
    and the failure receipt, which is all a caller needs to judge the sample.
    """

    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(description="Scenario name.")
    sample: int = Field(ge=1, description="Sample index that was replayed.")
    ok: bool = Field(description="True when the replay completed without divergence or error.")
    summary: tuple[str, ...] = Field(description="Ordered human-readable facts the replay observed.")
    unasked_calls: int = Field(ge=0, description="Recorded calls the replay never asked for.")
    error_type: str = Field(description="Empty on success.")
    error: str = Field(description="Empty on success.")
    traceback_text: str = Field(description="Empty on success.")


def source_hashes() -> dict[str, str]:
    """Return SHA-256 for every module the replay harness depends on."""
    paths: list[Path] = []
    for relative in SOURCE_PATHS:
        target = ROOT / relative
        if target.is_dir():
            paths.extend(sorted(path for path in target.rglob("*")
                                if path.is_file() and path.suffix in {".py", ".md", ".yaml"}))
        else:
            paths.append(target)
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _revision() -> str:
    """Return the git revision, marked dirty when the tree has local changes."""
    head = OperatingSystem.output_text(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).strip()
    status = OperatingSystem.output_text(["git", "status", "--porcelain"], cwd=ROOT).strip()
    return head + ("+dirty" if status else "")


def record(scenario: Scenario, samples: int, root: Path) -> Path:
    """Record ``samples`` independent live runs of one scenario."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = root / f"{scenario.name}-{stamp}"
    snapshot.mkdir(parents=True, exist_ok=False)
    print(f"Recording {scenario.name} x{samples} into {snapshot}", flush=True)
    for sample in range(1, samples + 1):
        directory = snapshot / f"sample-{sample:02d}"
        directory.mkdir(parents=True)
        started = datetime.now(timezone.utc)
        clock_start = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
        vclock.install(vclock.VirtualClock(clock_start))
        # Records store naive UTC; the virtual clock is the authority for both.
        # The store commits into ``situation.pickle`` as it runs, so the starting
        # point is copied to ``ghost.pickle`` and never written again. A replay
        # always begins exactly where the recording began.
        situation = scenario.situation(vclock.utcnow(), directory / "situation.pickle")
        situation.store.save()
        shutil.copyfile(directory / "situation.pickle", directory / "ghost.pickle")
        # The recording already keeps every exact request and response, so the
        # separate prompt log would only add files and draw identity values the
        # replay does not draw. Turn it off for both sides.
        OperatingSystem.environment["PM_PROMPT_LOG"] = "0"
        outcome: ScenarioOutcome | None = None
        failure: str | None = None
        runtime: ScenarioRuntime | None = None
        try:
            # Assembly draws identities too, so it runs inside the same seed.
            with deterministic_identities(f"{scenario.name}:{sample}"):
                runtime = open_runtime(scenario.name, situation.store, directory, mode="record",
                                       clock_start=clock_start)
                outcome = scenario.run(runtime)
        except BaseException as error:
            failure = traceback.format_exc()
            print(f"sample {sample} failed: {type(error).__name__}: {error}", flush=True)
        finally:
            if runtime is None:
                raise RuntimeError(f"Sample {sample} failed before the runtime was assembled.")
            requests = runtime.provider_requests()
            calls = runtime.model.recorded_calls
            manifest = RecordingManifest(
                scenario=scenario.name, sample=sample, recorded_at_utc=started, revision=_revision(),
                model=runtime.model.connection.model, endpoint=runtime.model.connection.base_url,
                clock_start_utc=clock_start, call_count=calls, request_count=requests,
                source_hashes=source_hashes(), notes=(scenario.description,),
                replay_blockers=() if outcome is None else outcome.replay_blockers)
            (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            runtime.close()
        if outcome is not None:
            (directory / "outcome.json").write_text(outcome.model_dump_json(indent=2), encoding="utf-8")
            print(f"sample {sample}: {calls} calls, {requests} requests", flush=True)
            for line in outcome.summary:
                print(f"  {line}", flush=True)
        if failure is not None:
            # Keep the partial recording: it is how the failure gets diagnosed
            # and replayed without paying for the same calls again.
            (directory / "record-failure.txt").write_text(failure, encoding="utf-8")
            raise RuntimeError(f"Recording sample {sample} failed; partial calls retained in {directory}")
    return snapshot


def _sample_directories(snapshot: Path) -> list[Path]:
    """Return the sample directories of a snapshot in index order."""
    directories = sorted(path for path in snapshot.iterdir() if path.is_dir() and path.name.startswith("sample-"))
    if not directories:
        raise FileNotFoundError(f"No sample directories in {snapshot}")
    return directories


def replay_one(directory: Path) -> ReplayResult:
    """Replay one recorded sample against the current code."""
    manifest = RecordingManifest.model_validate_json((directory / "manifest.json").read_text(encoding="utf-8"))
    scenario = scenario_named(manifest.scenario)
    if manifest.replay_blockers:
        # A partial trace cannot be replayed. Reporting it as a divergence would
        # blame the current code for a recording that never held a full run.
        return ReplayResult(scenario=scenario.name, sample=manifest.sample, ok=False, summary=(),
                            unasked_calls=0, error_type="NotReplayable",
                            error="; ".join(manifest.replay_blockers), traceback_text="")
    vclock.install(vclock.VirtualClock(manifest.clock_start_utc))
    # Every commit writes the store's own checkpoint, so replay works on a copy.
    # The recorded situation must stay byte-identical for later samples.
    scratch = directory / f"replay-{manifest.sample:02d}.pickle"
    shutil.copyfile(directory / "ghost.pickle", scratch)
    store = MemoryStore.from_checkpoint(scratch)
    # Replay draws identities from the same seed the recording used, into its
    # own prompt directory, so both runs consume the sequence identically.
    OperatingSystem.environment["PM_PROMPT_LOG"] = "0"
    # One identity sequence covers assembly and the run, exactly as recording
    # does. Two separate sequences would restart the counter and diverge.
    runtime: ScenarioRuntime | None = None
    try:
        with deterministic_identities(f"{scenario.name}:{manifest.sample}"):
            runtime = open_runtime(scenario.name, store, directory, mode="replay",
                                   clock_start=manifest.clock_start_utc)
            outcome = scenario.run(runtime)
            unasked = runtime.model.unasked_calls()
        (directory / f"replay-outcome-{manifest.sample:02d}.json").write_text(
            outcome.model_dump_json(indent=2), encoding="utf-8")
        return ReplayResult(scenario=scenario.name, sample=manifest.sample, ok=True,
                            summary=outcome.summary, unasked_calls=unasked,
                            error_type="", error="", traceback_text="")
    except (ReplayDivergence, AssertionError, RuntimeError, ValueError, KeyError) as error:
        if isinstance(error, ReplayDivergence) and error.expected and error.actual:
            # Keep both full prompts: the message only shows one window.
            prefix = directory / f"divergence-{manifest.sample:02d}-{error.index:03d}"
            prefix.with_name(prefix.name + "-recorded.txt").write_text(error.expected, encoding="utf-8")
            prefix.with_name(prefix.name + "-replayed.txt").write_text(error.actual, encoding="utf-8")
        return ReplayResult(scenario=scenario.name, sample=manifest.sample, ok=False, summary=(),
                            unasked_calls=0, error_type=type(error).__name__, error=str(error),
                            traceback_text=traceback.format_exc())
    finally:
        if runtime is not None:
            runtime.close()


def replay(snapshot: Path) -> list[ReplayResult]:
    """Replay every sample of a snapshot and write one result file per sample."""
    results: list[ReplayResult] = []
    for directory in _sample_directories(snapshot):
        result = replay_one(directory)
        results.append(result)
        (directory / f"replay-{result.sample:02d}.json").write_text(result.model_dump_json(indent=2),
                                                                   encoding="utf-8")
        status = "ok" if result.ok else f"FAILED {result.error_type}: {result.error}"
        print(f"sample {result.sample}: {status}", flush=True)
        for line in result.summary:
            print(f"  {line}", flush=True)
        if result.unasked_calls:
            print(f"  recorded calls the replay never asked for: {result.unasked_calls}", flush=True)
    return results


def compare(snapshot: Path) -> None:
    """Print the recorded outcomes and any replays beside each other."""
    for directory in _sample_directories(snapshot):
        manifest = RecordingManifest.model_validate_json((directory / "manifest.json").read_text(encoding="utf-8"))
        scenario = scenario_named(manifest.scenario)
        recorded = scenario.outcome_type.model_validate_json(
            (directory / "outcome.json").read_text(encoding="utf-8"))
        print(f"sample {manifest.sample} ({manifest.scenario}, model {manifest.model}, "
              f"revision {manifest.revision})")
        print("  recorded:")
        for line in recorded.summary:
            print(f"    {line}")
        replay_path = directory / f"replay-{manifest.sample:02d}.json"
        if not replay_path.exists():
            print("  replayed: (not yet replayed)")
            continue
        replayed = ReplayResult.model_validate_json(replay_path.read_text(encoding="utf-8"))
        if not replayed.ok:
            print(f"  replayed: FAILED {replayed.error_type}: {replayed.error}")
            continue
        print("  replayed:")
        for line in replayed.summary:
            print(f"    {line}")
        print(f"  identical summary: {replayed.summary == recorded.summary}")


def _require_stable_hash_seed(argv: Sequence[str]) -> None:
    """Restart once under a fixed hash seed.

    Python randomizes string hashing per process, so a set of names prints in a
    different order in each run. Some of that text reaches prompts, for example
    a validation message naming missing properties. A replay can only compare
    identical prompt text if the search order is fixed, so the harness runs
    under ``PYTHONHASHSEED=0``. The seed is read at interpreter start, hence the
    restart rather than an environment assignment.
    """
    if OperatingSystem.environment.get("PYTHONHASHSEED") == "0":
        return
    print("Rerunning with PYTHONHASHSEED=0 so prompt text is reproducible.", flush=True)
    environment = {**OperatingSystem.copy_environment(), "PYTHONHASHSEED": "0"}
    completed = OperatingSystem.run_text([OperatingSystem.python_executable(), "-m", "tests.replay", *argv],
                                         env=environment, capture_output=False)
    raise SystemExit(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the replay CLI."""
    arguments_in = list(sys.argv[1:] if argv is None else argv)
    _require_stable_hash_seed(arguments_in)
    parser = argparse.ArgumentParser(prog="python -m tests.replay", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List the scenarios this harness can run.")
    record_parser = subparsers.add_parser("record", help="Run a scenario live and keep its answers.")
    record_parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    record_parser.add_argument("--samples", type=int, default=1, help="Independent live runs of the same situation.")
    record_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    for name, help_text in (("replay", "Replay every sample of a snapshot."),
                            ("compare", "Show recorded and replayed outcomes side by side.")):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--snapshot", type=Path, required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "list":
        for scenario in SCENARIOS.values():
            print(f"{scenario.name}: {scenario.description}")
        return 0
    if arguments.command == "record":
        if arguments.samples < 1:
            raise ValueError("--samples must be at least 1.")
        snapshot = record(scenario_named(arguments.scenario), arguments.samples, arguments.root)
        print(f"Recorded {snapshot}", flush=True)
        return 0
    if arguments.command == "replay":
        results = replay(arguments.snapshot)
        return 0 if all(result.ok for result in results) else 1
    compare(arguments.snapshot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
