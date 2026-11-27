"""Run scenario 5 against a real participant and verify each offline boundary."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
import queue
import re
import shutil
import time

from pydantic import AliasPath, BaseModel, Field, JsonValue, TypeAdapter

from pm.app.control import DeliveryState
from pm.app.lifecycle_records import CLEAN_END, last_completed_shutdown, last_known_activity
from pm.app.transport import StimulusAccepted
from pm.host.durable_files import ProcessLock, atomic_json, read_json
from pm.memory.store import MemoryStore
from pm.records.base import RecordBase
from pm.records.enums import FeatureType, InterlocusType
from pm.records.memory_records import Feature
from pm.vclock import VirtualClockState
from simulation.client import PMProcess, ParticipantExit
from simulation.observer import Observation, ObservationEvidence, ObservationTrigger, RunObserver, runtime_evidence
from simulation.participant import MemoryWorkStatus, PreparedMemoryStatus, SourceCycleStatus
from simulation.personas import DEFAULT_RUNS, canon_home, fork_home
from simulation.source import SimulationConfigOverrides, copy_simulation_config, freeze_source


TIME_MULTIPLIER = 6
REPLY_DEADLINE_SECONDS = 45 * 60
MEMORY_DEADLINE_SECONDS = 45 * 60
BEDTIME_WAIT_SECONDS = 10 * 60
OBSERVATION_INTERVAL_SECONDS = 10 * 60
GOODBYE = "I'm shutting down my PC for the night now. Goodnight, Spamuel."
GOODBYE_ON = "I'm heading to bed. I'll leave the PC running tonight. Goodnight, Spamuel."
SLEEP = "I'm heading to bed and leaving the PC running. Please sleep on our earlier conversations and use that time to sort your memories. Goodnight, Spamuel."


@dataclass(frozen=True, slots=True)
class Evening:
    """Jesse's public lines and the next morning's recall question."""

    dialogue: tuple[str, ...]
    morning: str
    sleep: bool
    leave_online: bool = False


EVENINGS = (
    Evening((
        "I found a little brass compass at the flea market today. The needle works, but the lid sticks.",
        "The seller said it belonged to a surveyor. There are three tiny dots scratched inside the lid.",
        "I put it in the green tea tin on my desk so it won't get lost again.",
        "I'm tempted to clean it, but I kind of like that you can see where someone's thumb wore the brass smooth.",
        "I used to collect smooth pebbles as a kid. They all ended up jammed in my coat pockets.",
        "Would you leave the compass as it is? I think the scratches make it nicer.",
        "I wonder why I always want little objects like that to have a story.",
        "I think I'll take it on a walk this weekend. Digital maps make me look down at the wrong thing.",
    ), "Morning! I'm looking for that flea-market compass. Do you remember where I put it and what was scratched inside?", False),
    Evening((
        "I ended up taking the long way home today. There's a bakery round the corner I somehow never noticed.",
        "They had one lemon bun left. The baker told me her brother calls them little suns.",
        "It was still warm when I opened the paper bag on the bus.",
        "I tried bringing one home once and ate it before I got to my stop. Apparently nothing has changed.",
        "If you had a tiny bakery, what would you make too much of on purpose?",
        "I like shops where the people behind the counter obviously care about very small things.",
    ), "Morning. I might go back to that bakery. Do you remember what the baker's brother called those buns?", True, leave_online=True),
    Evening((
        "I finally dug out my old watercolor box. The lid still has a dried daub of ultramarine on it.",
        "I painted the view from my kitchen window, except I made the roofs purple and left the sky empty.",
        "I wrote 'The Violet Roofs' on the back because it sounded like a mystery novel.",
        "It turned out much better when I quit trying to get the windows straight.",
        "I'm giving it to my sister. She kept a picture I drew when I was eight and it still hangs by her front door.",
        "Do you have a favorite color for the bits of a picture that are meant to look like shadow?",
        "I'm hanging it by the kettle while it dries. I keep seeing little things I want to change.",
        "I think I'll leave it alone now. It's nice to finish something small.",
    ), "Morning! I'm wrapping that little painting for my sister. Do you remember its title and where I left it to dry?", False),
)


class Speech(BaseModel):
    output_id: int
    content: str
    timestamp: datetime
    cursor: str
    correlation_id: str
    addressee_entity_id: int


class ReplyLink(StrEnum):
    Correlation = "correlation"
    Stimulus = "responding_to_ids"
    Frontier = "generation_user_frontier"


class SpeechSources(BaseModel):
    responding_to_ids: list[int] = Field(default_factory=list[int])
    generation_user_frontier: int | None = None

    def reply_link(self, input_id: int, *, require_reply: bool) -> ReplyLink | None:
        if input_id in self.responding_to_ids:
            return ReplyLink.Stimulus
        if not require_reply and self.generation_user_frontier == input_id:
            return ReplyLink.Frontier
        return None


class SpeechTrace(BaseModel):
    sources: SpeechSources = Field(validation_alias=AliasPath("record", "data", "metadata"))


class Exchange(BaseModel):
    delivery_id: str
    text: str
    world_time: datetime
    stimulus_id: str
    input_record_id: int
    reply: Speech | None = None
    reply_link: ReplyLink | None = None


class ChatJournal(BaseModel):
    turns: list[Exchange]


class DialogueRecord(BaseModel):
    record_id: int
    content: str
    speaker: str
    timestamp: datetime
    delivery_id: str
    correlation_id: str
    addressee_entity_id: int
    sources: SpeechSources


type DialogueById = dict[int, DialogueRecord]
type MemoryById = dict[int, RecordBase]


class OvernightRun:
    def __init__(self, root: Path, source: Path, config: Path, start: datetime) -> None:
        self.root: Path = root
        self.source: Path = source
        self.home: Path = root / "spamuel"
        self.config: Path = config
        self.clock: datetime = start
        self.online_since: float = 0
        self.boot_clock: datetime = start
        self.exchanges: list[Exchange] = []
        self.outputs: set[int] = set()
        self.user_entity_id: int | None = None
        self.observer = RunObserver(config, root / "observer")
        self.phase: str = "starting evening"
        self.process_log: Path | None = None
        self.current_process: PMProcess | None = None

    def status(self, phase: str, night: int) -> None:
        self.phase = phase
        atomic_json(self.root / "status.json", {"phase": phase, "night": night,  # pm-lint: allow PM005 -- status receipt on disk
                    "world_time": self.clock.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()})
        print(f"Night {night} | {self.clock.isoformat()} | {phase}", flush=True)

    def boot(self, folder: Path) -> PMProcess:
        process = PMProcess("spamuel", self.home, str(self.config), folder / "ready.json", source_root=self.source)
        self.current_process = process
        self.process_log = folder / "spamuel-process.log"
        process.launch(self.clock.isoformat(), record_lifecycle=True)
        state = process.state()
        self.user_entity_id = TypeAdapter(int).validate_python(state["user_entity_id"])
        process.listen(str(state["output_cursor"]))
        self.online_since = time.monotonic()
        self.boot_clock = self.clock
        return process

    def advance_online(self, process: PMProcess) -> None:
        target = self.boot_clock + timedelta(seconds=int((time.monotonic() - self.online_since) * TIME_MULTIPLIER))
        if target >= self.clock + timedelta(seconds=6):
            process.post("/control/advance-step", {"to": target.isoformat()})  # pm-lint: allow PM005 -- HTTP request
            self.clock = target

    def submit(self, process: PMProcess, text: str) -> Exchange:
        self.advance_online(process)
        delivery = f"{self.root.name}:{len(self.exchanges) + 1}"
        reply = process.post("/v1/chat/completions", {  # pm-lint: allow PM005 -- HTTP request
            "user": "Jesse", "delivery_id": delivery, "messages": [{"role": "user", "content": text}],
        })
        receipt = StimulusAccepted.model_validate(reply)
        delivery_state = TypeAdapter(DeliveryState).validate_python(
            process.post("/control/delivery", {"delivery_id": delivery}))  # pm-lint: allow PM005 -- HTTP request
        input_record_id = delivery_state["record_id"]
        if input_record_id is None:
            raise RuntimeError(f"PM acknowledged {delivery} without a stored public input.")
        exchange = Exchange(delivery_id=delivery, text=text, world_time=self.clock, stimulus_id=receipt.stimulus_id,
                            input_record_id=input_record_id)
        self.exchanges.append(exchange)
        self.save_chat()
        return exchange

    def save_chat(self) -> None:
        atomic_json(self.root / "chat.json", {  # pm-lint: allow PM005 -- conversation receipt on disk
            "turns": [turn.model_dump(mode="json") for turn in self.exchanges],  # pm-lint: allow PM007 -- conversation evidence export
        })

    def next_speech(self, process: PMProcess, timeout: float) -> Speech | None:
        try:
            event = Speech.model_validate(process.events.get(timeout=timeout))
        except queue.Empty:
            process.check_process()
            return None
        if event.output_id in self.outputs:
            raise RuntimeError(f"Public speech replayed output {event.output_id}.")
        self.outputs.add(event.output_id)
        with (self.root / "speech.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")  # pm-lint: allow PM007 -- public event serialization
        return event

    def observe_runtime(self, trigger: ObservationTrigger, goal: str, evidence: list[ObservationEvidence]) -> None:
        request = Observation(trigger=trigger, situation=self.phase, goal=goal, world_time=self.clock,
                              evidence=[*evidence, *runtime_evidence(self.home, self.process_log)])
        self.observer.observe(request)

    def link_speech(self, process: PMProcess, exchange: Exchange, speech: Speech,
                    *, require_reply: bool) -> bool:
        link = None
        if speech.correlation_id == exchange.stimulus_id:
            link = ReplyLink.Correlation
        elif (not speech.correlation_id and speech.addressee_entity_id == self.user_entity_id
              and speech.timestamp.replace(tzinfo=timezone.utc) >= exchange.world_time):
            trace = SpeechTrace.model_validate(
                process.post(f"/control/trace/{speech.output_id}", {}))  # pm-lint: allow PM005 -- HTTP request
            link = trace.sources.reply_link(exchange.input_record_id, require_reply=require_reply)
        if link is None:
            return False
        exchange.reply = speech
        exchange.reply_link = link
        self.save_chat()
        print(f"{exchange.delivery_id} | speech {speech.output_id} | {link}", flush=True)
        return True

    def talk(self, process: PMProcess, text: str, *, require_reply: bool = False) -> Exchange:
        exchange = self.submit(process, text)
        started = time.monotonic()
        deadline = started + REPLY_DEADLINE_SECONDS
        next_observation = started + OBSERVATION_INTERVAL_SECONDS
        input_evidence = ObservationEvidence(id=str(exchange.input_record_id), text=f"Jesse: {exchange.text}")
        recent_speech: list[ObservationEvidence] = []
        while time.monotonic() < deadline:
            speech = self.next_speech(process, 1)
            if speech is not None:
                candidate = ObservationEvidence(id=str(speech.output_id), text=f"Spamuel: {speech.content}")
                recent_speech.append(candidate)
                recent_speech = recent_speech[-3:]
                if self.link_speech(process, exchange, speech, require_reply=require_reply):
                    return exchange
            if time.monotonic() >= next_observation:
                self.observe_runtime(ObservationTrigger.Waiting, "Receive a public response to Jesse's latest message.",
                                     [input_evidence, *recent_speech])
                next_observation = time.monotonic() + OBSERVATION_INTERVAL_SECONDS
            self.advance_online(process)
        raise TimeoutError(f"No matching public speech for {exchange.delivery_id}.")

    def say_goodnight(self, process: PMProcess, folder: Path, text: str) -> Exchange:
        exchange = self.submit(process, text)
        deadline = time.monotonic() + BEDTIME_WAIT_SECONDS
        cycle = SourceCycleStatus()
        while time.monotonic() < deadline:
            speech = self.next_speech(process, 1)
            if speech is not None and self.link_speech(process, exchange, speech, require_reply=True):
                break
            cycle = SourceCycleStatus.model_validate(process.post(
                "/control/source-cycle", {"source_id": exchange.input_record_id}))  # pm-lint: allow PM005 -- HTTP request
            if cycle.complete:
                break
            self.advance_online(process)
        atomic_json(folder / "goodbye.json", {  # pm-lint: allow PM005 -- bedtime departure receipt
            "delivery_id": exchange.delivery_id, "input_record_id": exchange.input_record_id,
            "reply_id": exchange.reply.output_id if exchange.reply is not None else None,
            "cycle": cycle.model_dump(mode="json"),  # pm-lint: allow PM007 -- cycle evidence export
            "wait_expired": exchange.reply is None and not cycle.complete,
        })
        return exchange

    def wait_for_memory(self, process: PMProcess, folder: Path) -> list[PreparedMemoryStatus]:
        self.submit(process, SLEEP)
        deadline = time.monotonic() + MEMORY_DEADLINE_SECONDS
        next_observation = time.monotonic() + OBSERVATION_INTERVAL_SECONDS
        saw_sleep = False
        previous: MemoryWorkStatus | None = None
        while time.monotonic() < deadline:
            status = MemoryWorkStatus.model_validate(process.post("/control/memory-work", {}))  # pm-lint: allow PM005 -- HTTP request
            if status != previous:
                with (folder / "memory-work.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(status.model_dump_json() + "\n")  # pm-lint: allow PM007 -- memory status serialization
                previous = status  # pm-lint: allow PM003 -- the parsed status record
            saw_sleep |= status.sleep_action_id is not None
            if saw_sleep and status.ready:
                return status.ready
            if time.monotonic() >= next_observation:
                self.observe_runtime(ObservationTrigger.Waiting, "See memory sleep prepare a complete unit before shutdown.",
                    [ObservationEvidence(id="memory-work", text=status.model_dump_json())])  # pm-lint: allow PM007 -- live memory receipt
                next_observation = time.monotonic() + OBSERVATION_INTERVAL_SECONDS
            self.next_speech(process, 2)
            self.advance_online(process)
        self.observe_runtime(ObservationTrigger.Failure, "Prepare a complete memory unit before Jesse goes to bed.",
            [ObservationEvidence(id="memory-deadline", text=f"No prepared memory unit at deadline. Sleep observed: {saw_sleep}.")])
        return []

    def stop_pm(self, process: PMProcess) -> ParticipantExit:
        request = process.request_shutdown()
        while True:
            stopped = process.wait_for_exit(request, timeout_seconds=OBSERVATION_INTERVAL_SECONDS)
            if stopped is not None:
                return stopped
            self.observe_runtime(ObservationTrigger.Waiting, "PM must finish its graceful drain and exit.",
                [ObservationEvidence(id="shutdown-ack", text=request.model_dump_json())])  # pm-lint: allow PM007 -- shutdown acknowledgment

    def leave_running(self, process: PMProcess, target: datetime) -> None:
        while self.clock < target:
            self.next_speech(process, 4)
            self.advance_online(process)

    def memory_status(self, process: PMProcess) -> MemoryWorkStatus:
        return MemoryWorkStatus.model_validate(process.post("/control/memory-work", {}))  # pm-lint: allow PM005 -- HTTP request

    def finish_run(self, night: int) -> None:
        process = self.current_process
        if process is None:
            return
        if process.process is not None and process.process.poll() is None and process.url:
            memory = self.memory_status(process)
            if memory.busy:
                self.status("letting memory work finish before final shutdown", night)
            next_observation = time.monotonic() + OBSERVATION_INTERVAL_SECONDS
            while memory.busy:
                self.next_speech(process, 4)
                self.advance_online(process)
                memory = self.memory_status(process)
                if time.monotonic() >= next_observation:
                    self.observe_runtime(ObservationTrigger.Waiting, "Let active memory work finish before stopping PM.",
                        [ObservationEvidence(id="memory-work", text=memory.model_dump_json())])  # pm-lint: allow PM007 -- live memory receipt
                    next_observation = time.monotonic() + OBSERVATION_INTERVAL_SECONDS
            self.stop_pm(process)
        process.close()

    def restart_at_morning(self, process: PMProcess, folder: Path, night: int, goodbye: Exchange,
                           before: MemoryStore, previous_units: set[str]) -> tuple[PMProcess, MemoryStore]:
        self.status("draining PM", night)
        stopped = self.stop_pm(process)
        atomic_json(folder / "shutdown.json", stopped.model_dump(mode="json"))  # pm-lint: allow PM007 -- shutdown receipt export
        process.close()
        checkpoint = folder / "shutdown.pickle"
        shutil.copyfile(self.home / "ghost.pickle", checkpoint)
        after = MemoryStore.from_checkpoint(checkpoint)
        saved = dialogue(after)
        verify_dialogue(dialogue(before), saved)
        verify_exchanges(saved, self.exchanges)
        stop = last_completed_shutdown(after)
        stop_time = datetime.fromisoformat(stopped.clock)
        if stop is None or stop.timestamp_world_begin != stop_time.replace(tzinfo=None):
            raise RuntimeError("The final checkpoint has no clean-stop record at the acknowledged shutdown time.")
        changed = memory_changes(before, after)
        for unit_id in previous_units:
            if unit_id not in after.consolidations:
                raise RuntimeError(f"Shutdown lost an earlier night's reconsolidation receipt {unit_id}.")
        goodbye_reply = next((row.record_id for row in saved.values() if row.correlation_id == goodbye.stimulus_id), None)
        atomic_json(folder / "checkpoint.json", {  # pm-lint: allow PM005 -- checkpoint receipt on disk
            "clean_stop_id": stop.id, "boot_id": str(stop.metadata["boot_id"]),
            "goodbye_delivery_id": goodbye.delivery_id,
            "goodbye_reply_id": goodbye.reply.output_id if goodbye.reply else goodbye_reply,
            "dialogue_ids": [key for key in sorted(saved)], "changed_memory_ids": [key for key in sorted(changed)],
            "retained_prior_units": [unit_id for unit_id in sorted(previous_units)],
        })
        self.clock = next_morning(stop_time)
        self.status("restarting from shutdown checkpoint", night)
        process = self.boot(folder / "morning")
        reloaded = MemoryStore.from_checkpoint(self.home / "ghost.pickle")
        if saved != dialogue(reloaded):
            raise RuntimeError("Reload changed the accepted public dialogue before Jesse's morning message.")
        verify_memory(after, reloaded, set(changed))
        for unit_id in previous_units:
            if reloaded.consolidations.get(unit_id) != after.consolidations[unit_id]:
                raise RuntimeError(f"Reload lost the consolidation receipt {unit_id}.")
        startup = verify_boot(after, reloaded, stop, self.clock)
        atomic_json(folder / "restart.json", {"checkpoint": str(checkpoint), "startup_id": startup.id,  # pm-lint: allow PM005 -- restart receipt on disk
                    "startup_text": startup.content, "startup_metadata": startup.metadata,
                    "morning_clock": self.clock.isoformat(), "dialogue_count": len(saved)})
        return process, after

    def play(self, evenings: tuple[Evening, ...]) -> int:
        folder = self.root / "night-1"
        self.status("starting evening", 1)
        before = MemoryStore.from_checkpoint(self.home / "ghost.pickle")
        night = 1
        shutdowns = 0
        previous_units: set[str] = set()
        prior_dialogue = len(self.exchanges)
        try:
            process = self.boot(folder / "evening")
            for night, evening in enumerate(evenings, 1):
                folder = self.root / f"night-{night}"
                self.status("public conversation", night)
                for line in evening.dialogue[prior_dialogue:]:
                    self.talk(process, line)
                prior_dialogue = 0
                prepared: list[PreparedMemoryStatus] = []
                if evening.sleep:
                    self.status("preparing memory sleep", night)
                    prepared = self.wait_for_memory(process, folder)
                self.status("Jesse is getting ready for bed", night)
                memory = self.memory_status(process)
                can_stop = not evening.leave_online and not memory.busy
                goodbye = None
                if not evening.sleep and not memory.busy:
                    goodbye = self.say_goodnight(process, folder, GOODBYE if can_stop else GOODBYE_ON)
                    memory = self.memory_status(process)
                    can_stop = can_stop and not memory.busy
                if can_stop and goodbye is not None:
                    process, after = self.restart_at_morning(process, folder, night, goodbye, before, previous_units)
                    shutdowns += 1
                else:
                    self.status("PM stays online overnight", night)
                    morning_at = next_morning(self.clock)
                    atomic_json(folder / "online-night.json", {  # pm-lint: allow PM005 -- bedside choice evidence
                        "scheduled_online": evening.leave_online, "memory_work": memory.model_dump(mode="json"),  # pm-lint: allow PM007 -- bedside memory evidence
                        "morning_clock": morning_at.isoformat(),
                    })
                    after = MemoryStore.from_checkpoint(self.home / "ghost.pickle")
                    self.leave_running(process, morning_at)
                saved = dialogue(after)
                self.status("checking morning recall", night)
                morning = self.talk(process, evening.morning, require_reply=True)
                snapshot_request: dict[str, JsonValue] = {"name": f"night-{night}-morning"}  # pm-lint: allow PM004 -- HTTP request
                snapshot = process.post("/control/snapshot", snapshot_request)  # pm-lint: allow PM005 -- HTTP request
                morning_store = MemoryStore.from_checkpoint(Path(str(snapshot["path"])))
                verify_dialogue(saved, dialogue(morning_store))
                verify_exchanges(dialogue(morning_store), self.exchanges)
                changed = memory_changes(before, morning_store)
                published = [unit for unit in prepared if unit.commit_id in morning_store.consolidations]
                previous_units.update(unit.commit_id for unit in published)
                retrieved = morning_memory(morning_store, morning, set(changed))
                atomic_json(folder / "morning.json", {"snapshot": snapshot, "retained_memory_ids": [key for key in sorted(changed)],  # pm-lint: allow PM005 -- morning receipt on disk
                            "retrieved_memory_ids": [key for key in sorted(retrieved)],
                            "published_units": [unit.model_dump(mode="json") for unit in published],  # pm-lint: allow PM007 -- morning memory evidence
                            "exchange": morning.model_dump(mode="json")})  # pm-lint: allow PM007 -- morning exchange evidence export
                if evening.sleep and (not published or not changed):
                    raise RuntimeError("Morning checkpoint did not retain any of the prepared memory units.")
                if evening.sleep and not retrieved:
                    raise RuntimeError("Morning cognition did not cite any of the retained reconsolidation records.")
                self.status("overnight verified", night)
                before = morning_store
                if night < len(evenings):
                    self.status("daytime online", night)
                    evening_time = self.clock.replace(hour=21, minute=0, second=0, microsecond=0)
                    self.leave_running(process, evening_time)
        except Exception as error:
            self.observe_runtime(ObservationTrigger.Failure, "Complete the current overnight stage.",
                [ObservationEvidence(id="exception", text=repr(error))])
            raise
        finally:
            self.finish_run(night)
        self.status("complete", len(evenings))
        return shutdowns


def next_morning(clock: datetime) -> datetime:
    morning = clock.replace(hour=7, minute=0, second=0, microsecond=0)
    return morning if morning > clock else morning + timedelta(days=1)


def dialogue(store: MemoryStore) -> DialogueById:
    result: DialogueById = {}
    for row in store.all_features:
        if row.feature_type is FeatureType.Dialogue and row.interlocus is InterlocusType.Public:
            result[row.id] = DialogueRecord(record_id=row.id, content=row.content, speaker=row.override_source or store.get_entity_name(row.entity_id) or "",
                timestamp=row.timestamp_world_begin, delivery_id=str(row.metadata.get("delivery_id", "")),
                correlation_id=str(row.metadata.get("reply_correlation", "")), addressee_entity_id=row.addressee_entity_id,
                sources=SpeechSources.model_validate(row.metadata))
    return result  # pm-lint: allow PM005 -- dialogue records keyed by knoxel ID


def verify_dialogue(before: DialogueById, after: DialogueById) -> None:
    missing = [key for key, row in before.items() if after.get(key) != row]
    if missing:
        raise RuntimeError(f"Accepted public dialogue was missing or changed: {missing}.")


def verify_exchanges(saved: DialogueById, exchanges: list[Exchange]) -> None:
    for exchange in exchanges:
        messages = [row for row in saved.values() if row.delivery_id == exchange.delivery_id]
        replies = [row for row in saved.values() if row.correlation_id == exchange.stimulus_id]
        if len(messages) != 1 or messages[0].record_id != exchange.input_record_id or messages[0].content != exchange.text:
            raise RuntimeError(f"Lost or duplicated Jesse's message {exchange.delivery_id}.")
        if len(replies) > 1:
            raise RuntimeError(f"Duplicated correlated reply to {exchange.delivery_id}.")
        if exchange.reply is not None:
            retained = saved.get(exchange.reply.output_id)
            if (retained is None or retained.content != exchange.reply.content
                    or retained.correlation_id != exchange.reply.correlation_id
                    or retained.addressee_entity_id != exchange.reply.addressee_entity_id):
                raise RuntimeError(f"Lost or duplicated Spamuel's reply to {exchange.delivery_id}.")
            if exchange.reply_link is ReplyLink.Correlation:
                saved_link = ReplyLink.Correlation if retained.correlation_id == exchange.stimulus_id else None
            else:
                saved_link = retained.sources.reply_link(exchange.input_record_id,
                    require_reply=exchange.reply_link is ReplyLink.Stimulus)
            if saved_link is None or saved_link != exchange.reply_link:
                raise RuntimeError(f"Lost the reply evidence for {exchange.delivery_id}.")


def memory_changes(before: MemoryStore, after: MemoryStore) -> MemoryById:
    return {key: row for key, row in after.all_knoxels.items()  # pm-lint: allow PM005 -- changed records keyed by knoxel ID
            if row.lineage is not None and row.lineage.operation.startswith("memory_reconsolidation.")
            and before.all_knoxels.get(key) != row}


def verify_memory(before: MemoryStore, after: MemoryStore, retained_ids: set[int]) -> None:
    lost = [key for key in retained_ids if before.all_knoxels[key] != after.all_knoxels.get(key)]
    if lost:
        raise RuntimeError(f"Reconsolidation changes were missing or altered after reload: {lost}.")


def verify_boot(stopped: MemoryStore, restarted: MemoryStore, stop: Feature, clock: datetime) -> Feature:
    startups = [row for row in restarted.all_features if row.metadata.get("origin") == "lifecycle"
                and row.metadata.get("event") == "startup" and row.id not in stopped.all_knoxels]
    if len(startups) != 1:
        raise RuntimeError(f"Expected one newly persisted boot record, found {len(startups)}.")
    startup = startups[0]
    expected = (clock.replace(tzinfo=None) - stop.timestamp_world_begin).total_seconds()
    if (startup.metadata["end_kind"] != CLEAN_END or startup.metadata["clean_shutdown_id"] != stop.id
            or startup.metadata["boot_id"] == stop.metadata["boot_id"] or startup.metadata["gap_seconds"] != expected
            or last_known_activity(stopped) != stop.timestamp_world_begin):
        raise RuntimeError(f"Boot did not report the expected clean gap of {expected}s: {startup.metadata}.")
    return startup


def morning_memory(store: MemoryStore, morning: Exchange, changed: set[int]) -> set[int]:
    inputs = {row.id for row in store.all_features if row.metadata.get("delivery_id") == morning.delivery_id}
    selected: set[int] = set()
    for turn in store.gwt_turns.values():
        if turn.slot_id == "main" and inputs.intersection(turn.stimulus_ids):
            selected.update(turn.knoxel_ids)
    return selected & changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--continue-smoke-from", help="Run ID of an interrupted first evening.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        raise ValueError("Use letters, digits, hyphens or underscores for the run ID.")
    continuation_id: str | None = args.continue_smoke_from
    if continuation_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", continuation_id):
        raise ValueError("Use letters, digits, hyphens or underscores for the prior run ID.")
    root = DEFAULT_RUNS / "overnight" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    with ProcessLock(root / "scenario.lock"):
        config = root / "launch-config.yaml"
        copy_simulation_config(args.config, config, SimulationConfigOverrides(profile=True, overnight=True))
        frozen = freeze_source(root / "source", config)
        for name, evenings in (("smoke", EVENINGS[:1]), ("multi-day", EVENINGS)):
            stage = root / name
            stage.mkdir()
            prior_turns: list[Exchange] = []
            if name == "smoke" and continuation_id is not None:
                prior_run = root.parent / continuation_id
                with ProcessLock(prior_run / "scenario.lock"):
                    earlier = prior_run / "smoke"
                    journal = ChatJournal.model_validate(read_json(earlier / "chat.json"))
                    prior_turns = journal.turns
                    if not prior_turns or tuple(turn.text for turn in prior_turns) != evenings[0].dialogue[:len(prior_turns)]:
                        raise ValueError("Continue from a first evening interrupted during ordinary conversation.")
                    home = fork_home(earlier / "spamuel", stage / "spamuel")
                verify_exchanges(dialogue(home.open_store()), prior_turns)
                clock = TypeAdapter(VirtualClockState).validate_python(read_json(home.root / "runtime/clock.json")["clock"])
                start = datetime.fromisoformat(clock["now"])
                atomic_json(stage / "continuation.json", {"prior_run": str(prior_run),  # pm-lint: allow PM005 -- continuation evidence on disk
                            "retained_inputs": [turn.input_record_id for turn in prior_turns], "clock": start.isoformat()})
            else:
                home = canon_home(stage / "spamuel")
                last_event = last_known_activity(home.open_store())
                if last_event is None:
                    raise RuntimeError("The shipped Spamuel canon has no continuity time.")
                start = (last_event.replace(tzinfo=timezone.utc) + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
            run = OvernightRun(stage, frozen.source, frozen.config, start)
            run.exchanges.extend(prior_turns)
            if prior_turns:
                run.save_chat()
            try:
                shutdowns = run.play(evenings)
                if name == "smoke" and not shutdowns:
                    raise RuntimeError("The smoke night stayed online. A shutdown and restart are still required before the multi-day run.")
            except Exception as error:
                atomic_json(stage / "failure.json", {"error": repr(error), "world_time": run.clock.isoformat()})  # pm-lint: allow PM005 -- failure receipt on disk
                raise
            if args.smoke_only:
                break


if __name__ == "__main__":
    main()
