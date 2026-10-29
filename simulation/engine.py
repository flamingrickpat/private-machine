"""Durable scenario execution and harness-owned participant routing.

One ``RunEngine`` owns one run folder: the controller document ``state.json``,
the per-attempt artifacts, and the participant processes it launches. It is the
authority for scenario progress and for the transit ledger; the browser only
reads detached copies through ``view``.

Delivery contract: a participant's public output is written to the controller
document before routing, and every routed message keeps a ``delivery_id`` that
the receiving process de-duplicates. A crash between the save and the send
retries the same delivery instead of duplicating it. Messages still in transit
belong to the coordinated checkpoint, so a restore returns them.

The package overview is in ``simulation/__init__.py``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Literal
from uuid import uuid4
from contextlib import closing

from pydantic import JsonValue

from pm.host.durable_files import ProcessLock, atomic_json
from pm.memory.checkpoint import read_checkpoint
from simulation.agents import Agents
from simulation.client import PMProcess
from simulation.evidence import capture_participant
from simulation.personas import fork_home
from simulation.plans import ScenarioPlan, StageJudgment
from simulation.source import freeze_source


class RunEngine:
    """Run an editable scenario with real participants and a durable transit queue.

    Pause and stop first fence new routing, then drain runnable work. Messages
    still in transit remain in the coordinated checkpoint. Resume retains them.
    """

    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.state_path: Path = root / "state.json"
        self._lock = threading.RLock()
        # The controller document is mutable browser state, not a fixed record:
        # its rows grow with the run. ``view`` documents the keys.
        self.state: dict[str, JsonValue] = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.plan: ScenarioPlan = ScenarioPlan.model_validate_json((root / "plan.json").read_text(encoding="utf-8"))
        self.processes: dict[str, PMProcess] = {}
        self.thread: threading.Thread | None = None
        self.request: threading.Event = threading.Event()
        self.requested_status: Literal["paused", "stopped"] = "paused"
        # Set by ``start`` for the lifetime of one attempt thread.
        self._run_lock: ProcessLock | None = None

    @classmethod
    def create(cls, root: Path, plan: ScenarioPlan, participants: Sequence[Mapping[str, str]],
               edges: Sequence[Mapping[str, str]], config: str) -> RunEngine:
        """Fork every source home, then publish the new run as ready.

        ``participants`` are selections that each name run-local ``id`` and the
        ``home`` folder to fork from. ``edges`` are directed speech connections
        with ``source`` and ``target`` participant IDs and an optional ``name``
        that the receiver shows as the speaker. ``config`` names the model
        configuration the participants start with.

        The run folder, ``plan.json``, ``state.json``, and the coordinated
        baseline are all durable before this returns, so no caller sees a half
        created run. Raises ``ValueError`` for duplicate or malformed selections
        and ``FileExistsError`` when the run folder already exists.
        """
        if not participants or len({item["id"] for item in participants}) != len(participants):
            raise ValueError("A run needs participants with unique IDs.")
        ids = {item["id"] for item in participants}
        if any(step.target not in ids for step in plan.steps):
            raise ValueError("Every scenario target must name a participant.")
        if any(edge["source"] not in ids or edge["target"] not in ids or edge["source"] == edge["target"] for edge in edges):
            raise ValueError("Each connection must join distinct registered participants.")
        if any(not item["id"].replace("-", "").replace("_", "").isalnum() for item in participants):
            raise ValueError("Participant IDs must contain letters, numbers, underscores, or hyphens.")
        root.mkdir(parents=True, exist_ok=False)
        prepared: list[dict[str, JsonValue]] = []
        for item in participants:
            home = fork_home(Path(item["home"]), root / "participants" / item["id"])
            data = read_checkpoint(home.ghost_pickle)
            history = data.causal_history
            last = max(history.transactions.values(), key=lambda row: row.sequence, default=None) if history else None
            cursor = f"{last.sequence}:{len(last.created_ids)}" if last else f"0:{len(data.state.knoxels)}"
            prepared.append({"id": item["id"], "source_home": item["home"], "home": str(home.root.resolve()),
                             "first_record_id": data.current_knoxel_id + 1, "cursor": cursor})
        atomic_json(root / "plan.json", plan.model_dump(mode="json"))
        initial_state: dict[str, JsonValue] = {
            "id": root.name, "title": plan.title, "status": "ready",
            "created_at": datetime.now(timezone.utc).isoformat(), "config": str(Path(config).resolve()),
            "participants": prepared, "edges": list(edges), "step_index": 0, "turn": 0, "attempt": 0,
            "clock": plan.start_time.isoformat(), "timeline": [], "transit": [], "facts": {},
            "pending_input": None, "reports": [], "failures": []}
        atomic_json(root / "state.json", initial_state)
        engine = cls(root)
        engine._initial_checkpoint()
        return engine

    def _initial_checkpoint(self) -> None:
        """Keep a coordinated baseline before any participant can exchange speech."""
        from pm.vclock import VirtualClock
        snapshots: dict[str, dict[str, JsonValue]] = {}
        for item in self.state["participants"]:
            home = Path(item["home"])
            snapshot = home / "runtime/snapshots/initial.pickle"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            # Copy the ghost and its clock receipt together, so the baseline
            # snapshot and the timers that resume from it describe one instant.
            shutil.copyfile(home / "ghost.pickle", snapshot)
            clock_path = home / "runtime/clock.json"
            saved = json.loads(clock_path.read_text(encoding="utf-8")) if clock_path.exists() else None
            clock: dict[str, JsonValue] = (
                saved if saved is not None and datetime.fromisoformat(saved["clock"]["now"]) == self.plan.start_time
                else {"clock": VirtualClock(self.plan.start_time).state(), "timers": {},
                      "environment": [], "carry": [], "output_cursor": item["cursor"]})
            atomic_json(snapshot.with_suffix(".json"), clock)
            snapshots[item["id"]] = {"path": str(snapshot.resolve())}
        atomic_json(self.root / "checkpoints/initial.json", {"clock": self.state["clock"],
                    "controller": self.view(), "snapshots": snapshots})

    def save(self) -> None:
        """Publish a complete controller receipt under the run mutex."""
        with self._lock:
            atomic_json(self.state_path, self.state)

    def view(self) -> dict[str, JsonValue]:
        """Return the controller document as a detached JSON copy for the browser.

        The document is mutable state rather than a fixed record. It carries
        ``id``, ``title``, ``status``, ``created_at``, ``config``,
        ``participants``, ``edges``, ``step_index``, ``turn``, ``attempt``,
        ``clock``, ``timeline``, ``transit``, ``facts``, ``pending_input``,
        ``reports``, ``failures``, and ``requested_status``. Timeline, transit,
        and fact rows accumulate for the life of the run, which is why this
        stays a JSON mapping instead of a rigid typed record.

        The JSON round trip detaches the reader from engine-owned state. With no
        attempt thread running, the file on disk is authoritative and is
        reloaded first, because another browser process may have written it.
        """
        with self._lock:
            if self.thread is None:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            return json.loads(json.dumps(self.state))

    def start(self) -> None:
        """Start or resume in the background. A duplicate start raises ValueError."""
        with self._lock:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError("This run is already active.")
            self._run_lock = ProcessLock(self.root / ".run.lock").acquire()
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
                self._finish_restore()
                if self.state["status"] == "failed" and len(self.state["participants"]) > 1:
                    barriers = list((self.root / "attempts").glob("*/coordinated-checkpoint.json"))
                    checkpoint = (max(barriers, key=lambda path: int(path.parent.name)) if barriers
                                  else self.root / "checkpoints/initial.json")
                    self._prepare_restore(checkpoint)
                if self.state["status"] == "completed":
                    raise ValueError("Fork a completed run to start a different scenario.")
            except BaseException:
                self._run_lock.close()
                raise
            self.request.clear()
            self.state["status"] = "starting"
            self.state["requested_status"] = None
            self.save()
            self.thread = threading.Thread(target=self._run, name=f"simulation-{self.root.name}", daemon=False)
            self.thread.start()

    def restore(self, attempt: int) -> None:
        """Restore all participants and routing state from one completed barrier.

        Preserve the later controller and ghost files. A durable restore intent
        lets startup finish a process interruption between file replacements.
        """
        with self._lock, closing(ProcessLock(self.root / ".run.lock").acquire()):
            if self.thread is not None and self.thread.is_alive():
                raise ValueError("Pause the run before restoring a coordinated checkpoint.")
            checkpoint = (self.root / "checkpoints/initial.json" if attempt == 0 else
                          self.root / "attempts" / str(int(attempt)) / "coordinated-checkpoint.json")
            self._prepare_restore(checkpoint)

    def _prepare_restore(self, checkpoint: Path) -> None:
        """Record a whole-world restore while the caller holds the run lock."""
        receipt = json.loads(checkpoint.read_text(encoding="utf-8"))
        token = uuid4().hex
        atomic_json(self.root / "restore-history" / f"{token}.json", self.view())
        atomic_json(self.root / "restore.json", {"status": "prepared", "token": token, "checkpoint": receipt,
            "attempt": self.state["attempt"], "reports": self.state["reports"], "failures": self.state["failures"]})
        self._finish_restore()

    def _finish_restore(self) -> None:
        """Finish an interrupted coordinated restore before any process starts."""
        intent_path = self.root / "restore.json"
        if not intent_path.exists():
            return
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        if intent["status"] == "completed":
            return
        receipt = intent["checkpoint"]
        targets: list[tuple[Path, Path, dict[str, JsonValue]]] = []
        for participant in receipt["controller"]["participants"]:
            home = Path(participant["home"]).resolve()
            snapshot = Path(receipt["snapshots"][participant["id"]]["path"]).resolve()
            if not home.is_relative_to((self.root / "participants").resolve()) or not snapshot.is_relative_to(home / "runtime" / "snapshots"):
                raise ValueError("A coordinated checkpoint escaped its participant home.")
            clock = json.loads(snapshot.with_suffix(".json").read_text(encoding="utf-8"))
            if not snapshot.is_file():
                raise ValueError("A coordinated participant snapshot is missing.")
            targets.append((home, snapshot, clock))
        from contextlib import ExitStack
        with ExitStack() as owners:
            for home, snapshot, clock in targets:
                lock = ProcessLock(home / ".simulation-process.lock").acquire()
                owners.callback(lock.close)
            for home, snapshot, clock in targets:
                # Keep the superseded ghost, then stage the restored one and
                # replace it in one rename. An interrupted restore leaves either
                # the old ghost or the restored ghost, never a partial file.
                archive = home / f"ghost.before-restore-{intent['token']}.pickle"
                if not archive.exists():
                    shutil.copyfile(home / "ghost.pickle", archive)
                staged = home / f"ghost.restoring-{intent['token']}.tmp"
                shutil.copyfile(snapshot, staged)
                staged.replace(home / "ghost.pickle")
                atomic_json(home / "runtime" / "clock.json", clock)
        self.state = {**receipt["controller"], "status": "paused", "requested_status": None,
                      "attempt": intent["attempt"], "reports": intent["reports"], "failures": intent["failures"]}
        self.save()
        atomic_json(intent_path, {**intent, "status": "completed"})

    def pause(self, *, stop: bool = False) -> None:
        """Request a boundary after current work finishes. Stop also permits resume."""
        with self._lock:
            if self.state["status"] not in {"starting", "running"}:
                raise ValueError("Only an active attempt can pause or stop.")
            if self.thread is None or not self.thread.is_alive():
                raise ValueError("This browser process does not own the active attempt.")
            self.requested_status = "stopped" if stop else "paused"
            self.request.set()
            self.state["requested_status"] = self.requested_status
            self.save()

    def collect(self, *, route: bool) -> None:
        """Record public outputs before routing, and persist each delivery before sending.

        The ledger is saved before a target is asked to accept a message, so a
        crash repeats the same ``delivery_id`` instead of producing a second one.
        ``route`` is false at a barrier: speech is still recorded and queued, but
        nothing is delivered until the run is allowed to advance.
        """
        with self._lock:
            for participant in self.state["participants"]:
                process = self.processes[participant["id"]]
                while True:
                    try:
                        event = process.events.get_nowait()
                    except queue.Empty:
                        break
                    if any(row.get("source") == process.id and row.get("output_id") == event["output_id"]
                           for row in self.state["timeline"]):
                        continue
                    self.state["timeline"].append({**event, "source": process.id, "kind": "speech"})
                    participant["cursor"] = event["cursor"]
                    for edge in self.state["edges"]:
                        if edge["source"] == process.id:
                            delivery_id = f"{self.root.name}:{process.id}:{event['output_id']}:{edge['target']}"
                            self.state["transit"].append({"delivery_id": delivery_id, "source": process.id,
                                "target": edge["target"], "name": edge.get("name", "Other Spamuel"),
                                "content": event["content"], "output_id": event["output_id"], "status": "pending"})
                    self.save()
            for delivery in self.state["transit"]:
                if delivery["status"] == "accepted":
                    continue
                target = self.processes[delivery["target"]]
                status = target.post("/control/delivery", {"delivery_id": delivery["delivery_id"]})
                if status["accepted"]:
                    delivery["status"] = "accepted"
                    self.save()
                    continue
                if route and not self.request.is_set() and not status["present"]:
                    ack = target.post("/v1/chat/completions", {
                        "messages": [{"role": "user", "content": delivery["content"]}],
                        "user": delivery["name"], "entity_ref": f"participant:{delivery['source']}",
                        "delivery_id": delivery["delivery_id"]})
                    delivery["ack"] = ack
                    delivery["status"] = "delivered"
                    self.save()

    def settle(self, *, target: str | None = None, route: bool = False) -> dict[str, dict[str, JsonValue]]:
        """Wait for runnable work without a deadline. Parked workflows remain resumable.

        Returns the last ``/control/state`` receipt of every participant, keyed by
        participant ID, or only ``target``'s receipt when one is named. A quiet
        window of 0.15 seconds counts as settled; polling continues until then so
        a late public output is not missed.
        """
        quiet_since: float | None = None
        while True:
            self.collect(route=route)
            states = {key: item.state() for key, item in self.processes.items()}
            selected = [states[target]] if target is not None else list(states.values())
            if all(item["idle"] for item in selected):
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since >= .15:
                    self.collect(route=False)
                    return states
            else:
                quiet_since = None
            time.sleep(.05)

    def advance(self, target: datetime) -> None:
        """Advance every participant through each semantic timer deadline.

        Poll timers do not force extra inference. Required wake and private
        thought deadlines still run. Routing stays buffered during the barrier.
        """
        while True:
            states = self.settle()
            now = max(datetime.fromisoformat(item["clock"]) for item in states.values())
            if target < now:
                raise ValueError("A scenario cannot move behind its current coordinated clock.")
            if now >= target or self.request.is_set():
                return
            stops = [target]
            stops.extend(datetime.fromisoformat(item["next_due"]) for item in states.values()
                         if item.get("next_due") and now < datetime.fromisoformat(item["next_due"]) <= target)
            stop = min(stops)
            for process in self.processes.values():
                process.post("/control/advance-step", {"to": stop.isoformat()})
            self.state["clock"] = stop.isoformat()
            self.save()

    def _stage(self, agents: Agents) -> None:
        """Run one conversational turn and review actual recorded responses."""
        step = self.plan.steps[self.state["step_index"]]
        self.advance(step.at)
        if self.request.is_set():
            return
        context = {"scenario": self.plan.model_dump(mode="json"), "stage": step.model_dump(mode="json"),
                   "conversation": self.state["timeline"], "delivered_facts": self.state["facts"]}
        pending = self.state["pending_input"]
        if pending is None:
            actor = agents.actor(context)
            if not isinstance(actor.get("content"), str) or not actor["content"].strip():
                raise ValueError("The scenario actor returned no message.")
            pending = {"content": actor["content"], "target": step.target, "speaker": step.speaker,
                       "delivery_id": f"{self.root.name}:stage:{step.id}:turn:{self.state['turn']}"}
            self.state["pending_input"] = pending
            self.save()
        process = self.processes[pending["target"]]
        delivered = process.post("/control/delivery", {"delivery_id": pending["delivery_id"]})
        if not delivered["present"]:
            process.post("/v1/chat/completions", {"messages": [{"role": "user", "content": pending["content"]}],
                "user": pending["speaker"], "delivery_id": pending["delivery_id"]})
        if not any(item.get("delivery_id") == pending["delivery_id"] for item in self.state["timeline"]):
            self.state["timeline"].append({**pending, "kind": "actor", "timestamp": self.state["clock"]})
            self.save()
        self.settle(target=step.target, route=True)
        delivered = process.post("/control/delivery", {"delivery_id": pending["delivery_id"]})
        if not delivered["accepted"]:
            raise RuntimeError("Main cognition did not admit the delivered scenario input.")
        judgment = agents.call("stage_director", "Review this stage against the actual delivered messages and PM replies. "
            "Return {step_complete:boolean,supported_fact_ids:[string],fact_evidence:{fact_id:[delivery_id]},reason:string,failure:string|null}. "
            "A fact needs explicit evidence in the delivered conversation. A missing reply alone is not failure. "
            "Only cite fact IDs assigned to this stage or already delivered earlier. "
            "Cite the exact delivery_id of the actor message that supports each fact. Companion speech is not actor evidence. "
            "Never skip mandatory facts. Request more conversation by returning step_complete:false. "
            "Companion disagreement is allowed; the actor must retain the scenario's commitments. "
            "Failure means the actor contradicts the required facts or an engineering failure prevents execution. "
            "A timed future event must not be introduced before its stage time. "
            "Do not rate companion quality; human review remains separate.",
            {**context, "conversation": self.state["timeline"]}, schema=StageJudgment)
        if judgment.get("failure"):
            raise RuntimeError(f"Scenario director reported failure: {judgment['failure']}")
        supported = set(judgment.get("supported_fact_ids", []))
        if not supported.issubset({item.id for item in self.plan.facts}):
            raise ValueError("The director cited an unknown mandatory fact.")
        if not supported.issubset(set(step.fact_ids) | set(self.state["facts"])):
            raise ValueError("The director cited a mandatory fact outside the current stage.")
        actor_messages = {row["delivery_id"]: row for row in self.state["timeline"] if row.get("kind") == "actor"}
        facts = {item.id: item for item in self.plan.facts}
        for fact_id in supported - set(self.state["facts"]):
            evidence = judgment["fact_evidence"][fact_id]
            for delivery_id in evidence:
                message = actor_messages.get(delivery_id)
                if message is None or message["target"] != step.target:
                    raise ValueError("The director cited an unavailable actor delivery or the wrong recipient.")
                at = facts[fact_id].at
                if at is not None and datetime.fromisoformat(message["timestamp"]) != at:
                    raise ValueError("The director's fact evidence does not retain its required time.")
            self.state["facts"][fact_id] = {"delivery_id": evidence[0], "delivery_ids": evidence, "reason": judgment["reason"]}
        if judgment.get("step_complete") and set(step.fact_ids).issubset(self.state["facts"]):
            self.state["step_index"] += 1
        self.state["turn"] += 1
        self.state["pending_input"] = None
        self.save()

    def _run(self) -> None:
        """Own execution, coordinated evidence, and cleanup of one attempt.

        The attempt folder exists before any participant starts, so a failure has
        a durable place for its receipt. ``outcome`` starts as ``failed`` and is
        replaced only by an explicit pause, stop, or completion, so an unexpected
        exit can never look like a finished scenario.
        """
        with closing(self._run_lock):
            self.state["attempt"] += 1
            attempt = self.root / "attempts" / str(self.state["attempt"])
            attempt.mkdir(parents=True, exist_ok=False)
            self.state["status"] = "starting"
            self.save()
            outcome: str = "failed"
            try:
                source = freeze_source(attempt / "source", Path(self.state["config"]))
                for item in self.state["participants"]:
                    process = PMProcess(item["id"], Path(item["home"]), source["config"],
                                        attempt / f"{item['id']}-endpoint.json", source_root=Path(source["source"]))
                    self.processes[item["id"]] = process
                    process.launch(self.state["clock"])
                    process.listen(item["cursor"])
                self.state["status"] = "running"
                self.save()
                agents = Agents(source["config"], attempt)
                while self.state["step_index"] < len(self.plan.steps) and not self.request.is_set():
                    self._stage(agents)
                outcome = self.requested_status if self.request.is_set() else "completed"
                if outcome == "completed" and set(self.state["facts"]) != {item.id for item in self.plan.facts}:
                    raise RuntimeError("The scenario ended without every mandatory fact.")
                self.settle()
                self._checkpoint(attempt)
            except BaseException as error:
                outcome = "failed"
                self.state["failures"].append({"attempt": self.state["attempt"], "error": repr(error)})
                atomic_json(attempt / "failure.json", {"error": repr(error)})
            finally:
                for process in self.processes.values():
                    try:
                        process.close()
                    except BaseException as error:
                        outcome = "failed"
                        self.state["failures"].append({"attempt": self.state["attempt"], "cleanup_error": repr(error)})
                self.processes = {}
                self.state["status"] = outcome
                self.state["requested_status"] = None
                self.save()
                self._report(attempt)

    def _checkpoint(self, attempt: Path) -> None:
        """Freeze every participant and the transit ledger at one clock barrier.

        Every participant is frozen and settled before any snapshot is taken, so
        the stored ghosts, clocks, and controller document describe one instant
        and a restore can resume the whole world without replaying speech.
        """
        for process in self.processes.values():
            process.post("/control/freeze", {})
        self.settle()
        snapshots: dict[str, dict[str, JsonValue]] = {}
        for participant in self.state["participants"]:
            process = self.processes[participant["id"]]
            snapshots[participant["id"]] = process.post("/control/snapshot", {"name": f"attempt-{self.state['attempt']}"})
        self.collect(route=False)
        atomic_json(attempt / "coordinated-checkpoint.json", {"clock": self.state["clock"],
            "snapshots": snapshots, "controller": self.view()})

    def _report(self, attempt: Path) -> None:
        """Retain failed attempts and capture actual records after process cleanup.

        A capture failure downgrades the run to ``failed`` but keeps every other
        participant's evidence, so a partial capture never claims a clean result.
        """
        captured: dict[str, dict[str, JsonValue]] = {}
        for item in self.state["participants"]:
            try:
                captured[item["id"]] = capture_participant(Path(item["home"]), attempt / "evidence" / item["id"], item["first_record_id"])
            except BaseException as error:
                self.state["failures"].append({"attempt": self.state["attempt"], "capture_error": repr(error)})
                self.state["status"] = "failed"
        report: dict[str, JsonValue] = {
            "schema": "pm.simulation-report.v1", "run": self.view(),
            "plan": self.plan.model_dump(mode="json"), "participants": captured,
            "human_review": "pending" if self.state["status"] == "completed" else "not_ready",
            "coordinated_checkpoint": "coordinated-checkpoint.json" if (attempt / "coordinated-checkpoint.json").exists() else None,
            "source": "source.json" if (attempt / "source.json").exists() else None,
            "automated_fact_checks": self.state["facts"]}
        atomic_json(attempt / "report.json", report)
        self.state["reports"].append((attempt / "report.json").relative_to(self.root).as_posix())
        self.state["requested_status"] = None
        self.save()
