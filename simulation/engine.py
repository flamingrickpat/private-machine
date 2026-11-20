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

from asyncio import CancelledError
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback
from typing import Literal
from uuid import uuid4
from contextlib import closing

from pydantic import BaseModel, JsonValue

from pm.host.durable_files import ProcessLock, atomic_json, runtime_status
from pm.memory.checkpoint import read_checkpoint
from simulation.agents import ActorContext, Agents, StageContext
from simulation.actors import ActorProfile, ConversationTurn, DeliveryEvidence, identity, load_profile
from simulation.client import PMProcess
from simulation.evidence import capture_participant

#: How often virtual time is pushed to the participants. One push per second
#: keeps the clock within a second of real time without flooding the controls.
ADVANCE_TICK_SECONDS = 1.0

def actor_turn_latency(delivery_id: str, pacing: ActorPacing) -> float:
    """Return Jesse's thinking time for one turn, derived from the turn identity.

    Deterministic on purpose: a resumed run keeps the same rhythm, and a failure
    receipt can be replayed against the same pacing.
    """
    digest = hashlib.sha256(delivery_id.encode("utf-8")).digest()
    choice = int.from_bytes(digest[:8], "big") / 2 ** 64
    if choice < pacing.ignored_probability:
        return pacing.ignored_seconds
    if choice < pacing.ignored_probability + pacing.delayed_probability:
        fraction = int.from_bytes(digest[8:16], "big") / 2 ** 64
        return pacing.delay_min_seconds + (pacing.delay_max_seconds - pacing.delay_min_seconds) * fraction
    return 0.0


def replied_after(rows: Sequence[Mapping[str, JsonValue]], delivery_id: str, source: str) -> bool:
    """Return whether ``source`` spoke after one delivered message."""
    seen_actor = False
    for row in rows:
        if row.get("delivery_id") == delivery_id:
            seen_actor = True
            continue
        if seen_actor and row.get("kind") == "speech" and row.get("source") == source:
            return True
    return False
from simulation.personas import fork_home
from simulation.plans import ActorPacing, ScenarioPlan, SimulationTiming, facts_for_scene
from simulation.source import freeze_source
from simulation.analysis import AnalysisSettings, analyze_run


class BetweenSceneReply(BaseModel):
    target: str
    speaker: str
    content: str
    delivery_id: str
    output_id: int


@dataclass(frozen=True, slots=True)
class ParticipantSelection:
    """One run-local participant and the body endpoint it may hold.

    ``home`` is the source character home that run creation forks. The source
    is never changed. ``minecraft_mcp_url`` names the one Minecraft body this
    participant owns for the whole run, or ``None`` for a participant that
    starts without a body. Two participants never share one endpoint, because
    a body lease belongs to exactly one avatar.
    """

    id: str
    home: str
    minecraft_mcp_url: str | None = None


def participant_body_endpoint(participant: Mapping[str, JsonValue]) -> str | None:
    """Return the body endpoint a controller participant declares.

    A run created before harness bodies existed has no such field, and a
    participant without a body keeps it null. Both read as "no body", so an old
    run folder still starts and resumes. A field that is present but not text is
    our own broken storage, and raises.
    """
    value = participant.get("minecraft_mcp_url")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("A controller participant body endpoint must be text or null.")
    return value


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
        self.time_multiplier: float = SimulationTiming.model_validate(self.state).time_multiplier
        self.actor_pacing: ActorPacing = SimulationTiming.model_validate(self.state).actor_pacing
        self.plan: ScenarioPlan = ScenarioPlan.model_validate_json((root / "plan.json").read_text(encoding="utf-8"))
        self.processes: dict[str, PMProcess] = {}
        self.thread: threading.Thread | None = None
        self.request: threading.Event = threading.Event()
        self.requested_status: Literal["paused", "stopped"] = "paused"
        # Set by ``start`` for the lifetime of one attempt thread.
        self._run_lock: ProcessLock | None = None

    @classmethod
    def create(cls, root: Path, plan: ScenarioPlan, participants: Sequence[ParticipantSelection],
               edges: Sequence[Mapping[str, str]], config: str, *, time_multiplier: float = 1.0,
               actor_pacing: ActorPacing = ActorPacing()) -> RunEngine:
        """Fork every source home, then publish the new run as ready.

        ``participants`` are selections that each name run-local ``id``, the
        ``home`` folder to fork from, and an optional ``minecraft_mcp_url`` for
        the one body that participant holds during the run. ``edges`` are
        directed speech connections with ``source`` and ``target`` participant
        IDs and an optional ``name`` that the receiver shows as the speaker.
        ``config`` names the model configuration the participants start with.

        The run folder, ``plan.json``, ``state.json``, and the coordinated
        baseline are all durable before this returns, so no caller sees a half
        created run. Raises ``ValueError`` for duplicate or malformed selections
        and ``FileExistsError`` when the run folder already exists.
        """
        if not participants or len({item.id for item in participants}) != len(participants):
            raise ValueError("A run needs participants with unique IDs.")
        body_endpoints = [item.minecraft_mcp_url for item in participants if item.minecraft_mcp_url is not None]
        if len(body_endpoints) != len(set(body_endpoints)):
            raise ValueError("Each participant needs its own Minecraft body endpoint.")
        ids = {item.id for item in participants}
        if any(step.target not in ids for step in plan.steps):
            raise ValueError("Every scenario target must name a participant.")
        if any(edge["source"] not in ids or edge["target"] not in ids or edge["source"] == edge["target"] for edge in edges):
            raise ValueError("Each connection must join distinct registered participants.")
        if any(not item.id.replace("-", "").replace("_", "").isalnum() for item in participants):
            raise ValueError("Participant IDs must contain letters, numbers, underscores, or hyphens.")
        timing = SimulationTiming(time_multiplier=time_multiplier, actor_pacing=actor_pacing)
        profiles: dict[str, ActorProfile] = {}
        for item in participants:
            profile = load_profile(Path(item.home))
            if plan.start_time < profile.continuity_at:
                raise ValueError("The scenario starts before the selected character's saved history ends.")
            if any(step.speaker != profile.speaker_name for step in plan.steps if step.target == item.id):
                raise ValueError("The scenario speaker must match the selected character's user name.")
            profiles[item.id] = profile
        root.mkdir(parents=True, exist_ok=False)
        prepared: list[dict[str, JsonValue]] = []
        for item in participants:
            home = fork_home(Path(item.home), root / "participants" / item.id)
            atomic_json(root / "actors" / f"{item.id}.json", profiles[item.id].model_dump(mode="json"))
            data = read_checkpoint(home.ghost_pickle)
            history = data.causal_history
            last = max(history.transactions.values(), key=lambda row: row.sequence, default=None) if history else None
            cursor = f"{last.sequence}:{len(last.created_ids)}" if last else f"0:{len(data.state.knoxels)}"
            prepared.append({"id": item.id, "source_home": item.home, "home": str(home.root.resolve()),
                             "minecraft_mcp_url": item.minecraft_mcp_url,
                             "first_record_id": data.current_knoxel_id + 1, "cursor": cursor})
        atomic_json(root / "plan.json", plan.model_dump(mode="json"))
        atomic_json(root / "analysis-settings.json", AnalysisSettings().model_dump(mode="json"))
        initial_state: dict[str, JsonValue] = {
            "id": root.name, "title": plan.title, "time_multiplier": timing.time_multiplier,
            "actor_pacing": timing.actor_pacing.model_dump(mode="json"), "status": "ready",
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
        live attempt thread, the file on disk is authoritative and is reloaded
        first, because another browser process may have written it. A finished
        thread still names itself here; only ``is_alive`` proves ownership.
        """
        with self._lock:
            if self.thread is None or not self.thread.is_alive():
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            result = json.loads(json.dumps(self.state))
            result["status"] = runtime_status(self.root, self.state["status"])
            result["last_saved_at"] = datetime.fromtimestamp(self.state_path.stat().st_mtime, timezone.utc).isoformat()
            result["owns_attempt"] = self.thread is not None and self.thread.is_alive()
            return result

    def has_body(self) -> bool:
        """Return whether any participant of this run owns a Minecraft body."""
        return any(participant_body_endpoint(item) is not None for item in self.state["participants"])

    def _refuse_body_restore(self) -> None:
        """Refuse to restore a run whose Minecraft world kept its later state.

        A coordinated checkpoint restores stores, clocks, and the transit
        ledger. It cannot restore a live Minecraft world, so replaying a failed
        attempt would put both characters into a world that already advanced
        past that attempt. A body run that fails needs a new run id.
        """
        if self.has_body():
            raise ValueError(
                "This run owns a Minecraft body, so its world cannot be restored with the other state. "
                "Create a new run instead of resuming this one.")

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
                    self._refuse_body_restore()
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
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._refuse_body_restore()
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
        """Request a boundary after current work finishes. Stop also permits resume.

        Any dashboard may read any run, but only the process that runs an
        attempt may pause it. When this process owns no live attempt, the
        durable document is read first and the refusal names the real cause,
        so a second page cannot report a stale status as if it were live.
        """
        with self._lock:
            if self.thread is None or not self.thread.is_alive():
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["status"] not in {"starting", "running"}:
                raise ValueError("Only an active attempt can pause or stop.")
            if self.thread is None or not self.thread.is_alive():
                raise ValueError(
                    "Another dashboard process owns the active attempt. Stop it from the page that started it.")
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
                    self.state["timeline"].append({**event, "source": process.id, "kind": "speech", "wall_at": datetime.now(timezone.utc).isoformat()})
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

    def deliver_between_scene_reply(self, reply: BetweenSceneReply) -> None:
        process = self.processes[reply.target]
        delivery = process.post("/control/delivery", {"delivery_id": reply.delivery_id})
        if not delivery["present"]:
            process.post("/v1/chat/completions", {"messages": [{"role": "user", "content": reply.content}],
                         "user": reply.speaker, "delivery_id": reply.delivery_id})
        self.state["timeline"].append({"kind": "actor", "wall_at": datetime.now(timezone.utc).isoformat(), "timestamp": self.state["clock"],
            "target": reply.target, "speaker": reply.speaker, "content": reply.content,
            "delivery_id": reply.delivery_id})
        self.state["pending_reaction"] = None
        self.save()

    def advance(self, target: datetime, agents: Agents) -> None:
        """Collect speech until the scene opens. Persist a reaction before sending it."""
        pending = self.state.get("pending_reaction")
        if pending is not None:
            self.deliver_between_scene_reply(BetweenSceneReply.model_validate(pending))
        states = {key: item.state() for key, item in self.processes.items()}
        base = max(datetime.fromisoformat(item["clock"]) for item in states.values())
        if target <= base:
            return
        started = time.monotonic()
        while not self.request.is_set():
            now = min(target, base + timedelta(seconds=(time.monotonic() - started) * self.time_multiplier))
            for process in self.processes.values():
                process.post("/control/advance-step", {"to": now.isoformat()})
            self.state["clock"] = now.isoformat()
            self.collect(route=True)
            self.save()
            if now >= target:
                return
            for participant in self.state["participants"]:
                target_id = participant["id"]
                evidence = [DeliveryEvidence.model_validate(row) for row in self.state["timeline"]
                            if row.get("target") == target_id or row.get("source") == target_id]
                if not evidence:
                    continue
                latest = evidence[-1]
                if (latest.kind != "speech" or latest.addressee_entity_id != states[target_id]["user_entity_id"]
                        or latest.output_id == participant.get("last_reaction_output_id")):
                    continue
                if latest.output_id is None:
                    raise ValueError("Public speech has no output ID.")
                delivery_id = f"{self.root.name}:reaction:{target_id}:{latest.output_id}"
                latency = actor_turn_latency(delivery_id, self.actor_pacing)
                if now < latest.timestamp + timedelta(seconds=latency):
                    continue
                profile = ActorProfile.model_validate_json(
                    (self.root / "actors" / f"{target_id}.json").read_text(encoding="utf-8"))
                context = ActorContext(identity=identity(profile), at=now,
                    scene="Between scheduled events, the companion has sent a message.",
                    intent="React naturally to the latest message in light of the conversation so far.",
                    current_facts=[fact for fact in self.plan.facts if fact.id in self.state["facts"]],
                    conversation=[ConversationTurn(speaker=row.speaker, at=row.timestamp.isoformat(),
                                                  text=row.content) for row in evidence],
                    guidance=self.plan.director.actor_guidance)
                if agents.wants_to_reply(context):
                    draft = agents.actor(context, [fact for fact in self.plan.facts if fact.at and fact.at > now],
                                         profile=profile, home=Path(participant["home"]))
                    reply = BetweenSceneReply(target=target_id, speaker=profile.speaker_name, content=draft.content,
                                              delivery_id=delivery_id, output_id=latest.output_id)
                    self.state["pending_reaction"] = reply.model_dump(mode="json")
                    participant["last_reaction_output_id"] = latest.output_id
                    self.save()
                    self.deliver_between_scene_reply(reply)
                else:
                    participant["last_reaction_output_id"] = latest.output_id
                    self.save()
            time.sleep(ADVANCE_TICK_SECONDS)

    def wait_actor_turn(self, delivery_id: str, source: str) -> None:
        """Let Jesse write her next line without ever waiting on Spamuel.

        Her cadence belongs to her, not to the companion. If Spamuel answered she
        writes at her own pace; if he said nothing she still writes again after a
        real pause of at least :data:`ACTOR_UNANSWERED_WAIT_SECONDS`. An ignored
        message is therefore a measured outcome, never a stalled run, and neither
        actor blocks the other. Speech arriving during the wait is still
        collected and routed.
        """
        sampled = actor_turn_latency(delivery_id, self.actor_pacing)
        states = {key: item.state() for key, item in self.processes.items()}
        base = max(datetime.fromisoformat(item["clock"]) for item in states.values())
        started = time.monotonic()
        answered_at: float | None = None

        while not self.request.is_set():
            now = time.monotonic()
            elapsed = (now - started) * self.time_multiplier
            clock = base + timedelta(seconds=elapsed)
            for process in self.processes.values():
                process.post("/control/advance-step", {"to": clock.isoformat()})
            self.state["clock"] = clock.isoformat()
            self.collect(route=True)
            self.save()
            answered = replied_after(self.state["timeline"], delivery_id, source)
            if answered:
                if answered_at is None:
                    answered_at = now
                if (now - answered_at) * self.time_multiplier >= sampled:
                    return
            elif elapsed >= max(sampled, self.actor_pacing.unanswered_wait_seconds):
                return
            time.sleep(ADVANCE_TICK_SECONDS)

    def _stage(self, agents: Agents) -> None:
        """Run one conversational turn and review actual recorded responses."""
        step = self.plan.steps[self.state["step_index"]]
        self.advance(step.at, agents)
        if self.request.is_set():
            return
        profile = ActorProfile.model_validate_json(
            (self.root / "actors" / f"{step.target}.json").read_text(encoding="utf-8"))
        conversation: list[ConversationTurn] = []
        for row in self.state["timeline"]:
            evidence = DeliveryEvidence.model_validate(row)
            if evidence.target == step.target or evidence.source == step.target:
                conversation.append(ConversationTurn(speaker=evidence.speaker,
                    at=evidence.timestamp.isoformat(), text=evidence.content))
        context = ActorContext(identity=identity(profile), at=step.at, scene=step.instruction, intent=step.actor_intent,
            current_facts=facts_for_scene(self.plan, step),
            conversation=conversation, guidance=self.plan.director.actor_guidance)
        pending = self.state["pending_input"]
        if pending is None:
            # ActorTurn rejects a blank message inside the call, so the actor is asked again.
            home = next(Path(item["home"]) for item in self.state["participants"] if item["id"] == step.target)
            actor = agents.actor(context, [fact for fact in self.plan.facts if fact.at is not None and fact.at > step.at],
                                 profile=profile, home=home)
            pending = {"content": actor.content, "target": step.target, "speaker": step.speaker,
                       "delivery_id": f"{self.root.name}:stage:{step.id}:turn:{self.state['turn']}"}
            self.state["pending_input"] = pending
            self.save()
        process = self.processes[pending["target"]]
        if self.request.is_set():
            return
        delivered = process.post("/control/delivery", {"delivery_id": pending["delivery_id"]})
        if not delivered["present"]:
            process.post("/v1/chat/completions", {"messages": [{"role": "user", "content": pending["content"]}],
                "user": pending["speaker"], "delivery_id": pending["delivery_id"]})
        if not any(item.get("delivery_id") == pending["delivery_id"] for item in self.state["timeline"]):
            self.state["timeline"].append({**pending, "kind": "actor", "wall_at": datetime.now(timezone.utc).isoformat(), "timestamp": self.state["clock"]})
            self.save()
        self.wait_actor_turn(pending["delivery_id"], step.target)
        delivered = process.post("/control/delivery", {"delivery_id": pending["delivery_id"]})
        if not delivered["accepted"]:
            raise RuntimeError("Main cognition did not admit the delivered scenario input.")
        delivered_facts = set(self.state["facts"])

        judgment = agents.judge(StageContext(scenario=self.plan, stage=step,
            companion_name=profile.companion_name,
            conversation=[DeliveryEvidence.model_validate(row) for row in self.state["timeline"]],
            delivered_fact_ids=list(self.state["facts"])))
        if judgment.failure:
            raise RuntimeError(f"Scenario director reported failure: {judgment.failure}")
        supported = set(judgment.supported_fact_ids)
        for fact_id in supported - delivered_facts:
            evidence = judgment.fact_evidence[fact_id]
            self.state["facts"][fact_id] = {"delivery_id": evidence[0], "delivery_ids": evidence, "reason": judgment.reason}
        scene_prefix = f"{self.root.name}:stage:{step.id}:turn:"
        scene_turns = sum(1 for row in self.state["timeline"]
                          if row.get("kind") == "actor" and row["delivery_id"].startswith(scene_prefix))
        if judgment.step_complete and set(step.fact_ids).issubset(self.state["facts"]) and scene_turns >= step.minimum_turns:
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
                                        attempt / f"{item['id']}-endpoint.json", source_root=Path(source["source"]),
                                        minecraft_mcp_url=participant_body_endpoint(item))
                    self.processes[item["id"]] = process
                    process.launch(self.state["clock"])
                    process.listen(item["cursor"])
                self.state["status"] = "running"
                self.save()
                agents = Agents(source["config"], attempt, cancel=self.request)
                while self.state["step_index"] < len(self.plan.steps) and not self.request.is_set():
                    try:
                        self._stage(agents)
                    except CancelledError:
                        if not self.request.is_set():
                            raise
                        break
                outcome = self.requested_status if self.request.is_set() else "completed"
                if outcome == "completed" and set(self.state["facts"]) != {item.id for item in self.plan.facts}:
                    raise RuntimeError("The scenario ended without every mandatory fact.")
                self.settle()
                self._checkpoint(attempt)
            except BaseException as error:
                outcome = "failed"
                self.state["failures"].append({"attempt": self.state["attempt"], "error": repr(error)})
                atomic_json(attempt / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
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
            settings_path = self.root / "analysis-settings.json"
            if self.state["status"] == "completed" and settings_path.exists():
                settings = AnalysisSettings.model_validate_json(settings_path.read_text(encoding="utf-8"))
                if settings.automatic:
                    try:
                        analyze_run(self.root, source["config"])
                    except Exception as error:
                        # Rating failure stays visible and never reruns completed companion work.
                        atomic_json(self.root / "analysis-failure.json", {
                            "error": repr(error), "traceback": traceback.format_exc()})

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
