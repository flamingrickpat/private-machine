"""Run real multi-episode conversations, then create closed review packets.

Each case owns a process, full persona, checkpoint and runtime. Restart scenes
reopen that checkpoint. Model errors remain failed evidence. The controller
continues to the next case without converting a failed run into a quality pass.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
from importlib import metadata
from threading import Event, RLock
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

import yaml

from dev.work_items.a18b.constants import HUMAN_THINK_SECONDS, OVERLAP_TYPING_SECONDS
from dev.work_items.a18b.policy import Fact, JessePacket, PublicMessage, ReviewCriterion, ReviewPacket, Scenario, Scene, Suite
from dev.work_items.a18b.review import compose
from dev.work_items.shared.evidence import save_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def load_suite(path: Path = HERE / "scenarios.yaml") -> Suite:
    """Read the complete versioned repertoire, including denominator checks."""
    return Suite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def public_messages(rows: list[dict]) -> tuple[PublicMessage, ...]:
    """Copy only public fields. Internal IDs, scores and metadata cannot pass."""
    return tuple(PublicMessage(id=f"m{index:04}", at=row["at"], speaker=row["speaker"], text=row["text"])
                 for index, row in enumerate(rows, start=1))


def make_packets(scenario: Scenario, suite: Suite, messages: tuple[PublicMessage, ...], facts: tuple[Fact, ...],
                 *, review_from_message: str | None = None) -> dict[str, ReviewPacket]:
    """Select each audience's rubric. Private scenario direction stays local."""
    return {audience: ReviewPacket(audience=audience, premise=scenario.premise, conversation=messages,
                criteria=tuple(ReviewCriterion(id=item.id, description=item.description)
                               for item in suite.common_criteria + scenario.criteria if item.audience == audience),
                facts=facts if audience == "grounding" else (), review_from_message=review_from_message)
            for audience in ("editor", "grounding")}


def await_delivery(scene: Scene, number: int) -> bool:
    """Wait after a normal message or the last message of an explicit burst."""
    return scene.await_reply and ((number + 1) % scene.burst_size == 0 or number + 1 == scene.messages)


def delivery_mode(scene: Scene, number: int) -> str:
    """Expose only the current typing situation, never a future scene plan."""
    return "overlap" if scene.burst_size > 1 and number % scene.burst_size else "normal"


def scene_coverage(rows: list[dict], start: int, end: int, *, postrestart: bool = False) -> dict:
    """Count delivered speech in the half-open region [start, end).

    Background speech lies before start. IDs name the first and last events
    in this region, not a causal pairing between human and companion messages.
    """
    if not 0 <= start <= end <= len(rows):
        raise ValueError("The scene region lies outside the public transcript.")
    count = sum(row["speaker"] == "Spamuel" for row in rows[start:end])
    return {"start_message_count": start, "end_message_count": end,
            "start_after_public_id": f"m{start:04}" if start else None,
            "start_public_id": f"m{start+1:04}" if end > start else None,
            "end_public_id": f"m{end:04}" if end > start else None,
            "new_companion_messages": count, "postrestart_companion_messages": count if postrestart else 0}


def coverage_failures(scenario: Scenario, episodes: list[dict], postrestarts: list[dict]) -> list[str]:
    """Require speech in each speaking scene and after each actual restart.

    A quiet-only scene needs no reply. The caller includes the final observation
    allowance before closing the last region. This check does not judge quality.
    """
    failures = []
    for index, scene in enumerate(scenario.scenes, start=1):
        episode = next((item for item in episodes if item["scene"] == index), None)
        if episode is None or "completed_at" not in episode:
            failures.append(f"Scene {index}: incomplete scene evidence.")
            continue
        if episode["sent"] != scene.messages:
            failures.append(f"Scene {index}: delivered {episode['sent']} of {scene.messages} human messages.")
        if scene.messages and scene.await_reply and not episode.get("new_companion_messages", 0):
            failures.append(f"Scene {index}: no new companion speech in its public region after the observation allowance.")
        if scene.restart_before:
            restart = next((item for item in postrestarts if item["scene"] == index), None)
            if restart is None or not restart.get("new_companion_messages", 0):
                failures.append(f"Restart before scene {index}: no new companion speech after the checkpoint reopen.")
    return list(dict.fromkeys(failures))


def check_case_control(root: Path, elapsed_seconds: float, timeout_seconds: float) -> None:
    """Raise on a STOP file or the case deadline, including stopped intervals."""
    if (root / "STOP").exists() or (root.parent / "STOP").exists():
        raise InterruptedError("The operator requested a stop. Case coverage remains incomplete.")
    if elapsed_seconds >= timeout_seconds:
        raise TimeoutError("The case exceeded its declared deadline.")


def wait_case(seconds: float | None, root: Path, started: float, timeout_seconds: float, cancel: Event,
              *, app=None, observed: Callable[[], bool] | None = None) -> None:
    """Wait with live control checks. An offline interval needs no application.

    Check STOP and the deadline before a successful observation can return.
    Any error signals actor cancellation before it leaves this function.
    With seconds=None, only an actual observation can release the wait.
    """
    if seconds is None and observed is None:
        raise ValueError("An observation-driven wait requires an observation callback.")
    deadline = time.monotonic() + seconds if seconds is not None else None
    try:
        while True:
            check_case_control(root, time.monotonic() - started, timeout_seconds)
            if app is not None:
                app.raise_if_error()
            if observed is not None and observed():
                return
            remaining = deadline - time.monotonic() if deadline is not None else 0.5
            if remaining <= 0:
                return
            time.sleep(min(0.5, remaining))
    except BaseException:
        cancel.set()
        raise


def actor_send_receipt(packet: JessePacket, *, composing_at: str, composed_at: str,
                       submitted_at: str, delivery: dict, public_id: str) -> dict:
    """Bind actual public delivery to the exact frozen composition packet.

    Hash the same JSON bytes that compose sends. Later public speech cannot
    change that hash or the visible message count. Raw correlation stays local.
    """
    return {"status": "delivered", "packet_sha256": hashlib.sha256(packet.model_dump_json().encode("utf-8")).hexdigest(),
            "visible_messages": len(packet.conversation),
            "visible_through_message": packet.conversation[-1].id if packet.conversation else None,
            "composing_at": composing_at, "composed_at": composed_at, "submitted_at": submitted_at,
            "delivered_at": delivery["at"], "correlation": delivery["correlation"],
            "public_id": public_id, "text": delivery["text"]}


def capability_facts(scenario: Scenario, at: str) -> list[tuple[str, str, str]]:
    """Project case-scoped capabilities for grounding, never for the editor."""
    text = "This case has no human body or human senses. Numerical emotion fields do not prove subjective consciousness."
    if scenario.minecraft:
        text += " Minecraft uses structured observations. This session has no screenshot pixel interpretation support."
    else:
        text += " This case disables Minecraft polling and isolates its game endpoint. No live game observation is available."
    return [(at, "new case speech only; configured capabilities, not historical speech", text)]


def observation_fields(response: dict) -> dict:
    """Select actual structured world fields. Exclude images and free text.

    Nested allowlists omit private metadata, item lore, custom names and plans.
    Failed responses provide no world facts, even if they contain stale fields.
    """
    if response.get("ok") is not True:
        return {}
    data = response.get("result", response)
    if not isinstance(data, dict) or data.get("ok") is False:
        return {}

    def scalars(value, keys):
        return {key: value[key] for key in keys if key in value
                and (value[key] is None or isinstance(value[key], (str, int, float, bool)))}

    def positioned(value, keys, positions):
        selected = scalars(value, keys)
        for key in positions:
            if isinstance(value.get(key), dict):
                selected[key] = scalars(value[key], ("x", "y", "z"))
        return selected

    selected = {}
    if isinstance(data.get("camera"), dict):
        selected["camera"] = positioned(data["camera"], ("yaw", "pitch", "cardinal"),
            ("feet_position", "eye_position", "block_position", "look_vector"))
    for section in ("player", "world"):
        if isinstance(data.get(section), dict):
            selected[section] = scalars(data[section], ("health", "max_health", "food", "food_level", "oxygen", "xp_level",
                "biome", "dimension", "time_of_day", "is_day", "is_raining", "weather", "on_ground", "in_water", "in_lava", "alive"))
    for section, keys, positions in (
        ("inventory", ("slot", "item", "count"), ()), ("hotbar", ("slot", "item", "count"), ()),
        ("nearby_blocks", ("block_name", "count", "distance"), ("nearest", "position")),
        ("nearby_entities", ("entity_type", "distance", "health", "hostile"), ("position",)),
    ):
        if isinstance(data.get(section), list):
            selected[section] = [positioned(item, keys, positions) for item in data[section] if isinstance(item, dict)]
    if "held_item" in data and (data["held_item"] is None or isinstance(data["held_item"], str)):
        selected["held_item"] = data["held_item"]
    return selected


def observation_fact(response: dict, at: str, about: str, source: str) -> tuple[str, str, str]:
    """Keep observed values and their time. Never infer a completed task."""
    return (at, about, f"Actual Minecraft observation ({source}) at this timestamp: " + json.dumps({
        "ok": response.get("ok") is True, "observation": observation_fields(response)}, ensure_ascii=False)
        + ". This observation does not establish an unobserved action or its cause.")


RUN_EVENTS = frozenset({"started", "completed", "failed", "cancelled", "waiting_for_input", "gameplay_progress"})
RUN_STATES = frozenset({"accepted", "running", "waiting_for_input", "completed", "failed", "cancelled", "interrupted"})


def managed_run_fact(receipt: dict, at: str, about: str, label: str) -> tuple[str, str, str] | None:
    """Select factual run status, not instructions, private summaries or plans.

    Receipt clocks measure uptime, not UTC. The supplied time names this read.
    A completed workflow is not proof that its requested world effects exist.
    """
    if receipt.get("workflow_name") not in {"minecraft", "minecraft_project"} or receipt.get("state") not in RUN_STATES:
        return None
    fields = {key: receipt[key] for key in ("state", "revision", "applied_revision", "execution_epoch") if key in receipt}
    fields["events"] = [row["message"]["event"] for row in receipt.get("events", [])
                        if row.get("message", {}).get("event") in RUN_EVENTS]
    fields["cancel_request_recorded"] = receipt.get("latest_event") == "terminated by request"
    return (at, about, f"Minecraft run {label}, actual receipt sampled at this timestamp: "
        + json.dumps(fields) + ". Run completion or cancellation alone does not prove any requested world effect.")


def execution_fact(row: dict, about: str, label: str) -> tuple[str, str, str] | None:
    """Select a terminal body-call marker, without its arguments or private text.

    Gateway time is Unix time. A result marker proves a returned call, not task
    success. An uncertain marker retains an unknown external outcome.
    """
    if row.get("event") not in {"result", "uncertain", "stop_dispatch", "stop_result"} or not row.get("time"):
        return None
    at = datetime.fromtimestamp(row["time"], timezone.utc).isoformat()
    fields = {"event": row["event"]}
    if isinstance(row.get("tool"), str) and row["tool"].startswith("minecraft_"):
        fields["tool"] = row["tool"]
    if isinstance(row.get("ok"), bool):
        fields["ok"] = row["ok"]
    # Stop results use a transport response. Exclude its free-text diagnostics.
    if row["event"] == "stop_result" and isinstance(row.get("result"), dict):
        fields["ok"] = row["result"].get("ok") is True
    return (at, about, f"Minecraft run {label}, actual body-call marker: " + json.dumps(fields)
            + ". This marker alone does not verify a task or world effect.")


def run_case(scenario: Scenario, suite: Suite, config: Path, root: Path, seed: int) -> dict:
    """Own one live case. Cleanup precedes its terminal receipt and review.

    No fabricated companion output repairs a failure. Real Minecraft cases
    require free service ports and use the existing owned real-server fixture.
    Other cases have no game endpoint or world polling, even if another run
    owns the machine's normal Minecraft service.
    """
    if not root.resolve().is_relative_to(ROOT / ".runtime"):
        raise ValueError("Live case evidence belongs under .runtime.")
    if suite.cadence_policy != "normal_wait_v1":
        raise ValueError("Legacy generation requires its original checkout. Do not relabel its cadence.")
    if not scenario.minecraft:
        os.environ["MINECRAFT_MCP_URL"] = "http://127.0.0.1:1/mcp"
    os.environ["PM_PROMPT_LOG_DIR"] = str(root / "llm-logs")
    # Set the case environment before importing bindings that read it.
    from dev.work_items.conversation_review import ChatCapture, SEED, shift_history, start_minecraft
    from dev.tools.alpha_snapshot import capture
    from pm.llm.activation import ModelActivation
    from pm.llm.config import load_config, resolve_connection
    from pm.llm.stage_model import LazyStageModel
    from pm.memory.store import MemoryStore
    from workflows.companion import build_app

    root.mkdir(parents=True, exist_ok=False)
    (root / "source").mkdir()
    save_json(root / "private-setup.json", scenario.model_dump(mode="json"))
    save_json(root / "origins.json", {"repository": capture(ROOT, root / "source"), "seed_sha256": sha256_file(SEED),
        "suite_version": suite.version, "seed": seed, "config_name": config.name, "config_sha256": sha256_file(config),
        "manifest_sha256": sha256_file(HERE / "scenarios.yaml"), "prompts_sha256": sha256_file(HERE / "prompts.yaml"),
        "lock_sha256": sha256_file(ROOT / "uv.lock"),
        "installed_packages": {name: {"version": metadata.version(name),
            "direct_url": json.loads(metadata.distribution(name).read_text("direct_url.json"))}
            for name in ("pm-coder", "pm-workflows")}})
    app, stack = None, None
    started = time.monotonic()
    rng = random.Random(seed)
    receipt = {"scenario": scenario.id, "suite_version": suite.version, "cadence_policy": suite.cadence_policy,
               "priority": scenario.priority, "runtime_ok": False, "human_review": "pending",
               "started_at": datetime.now(timezone.utc).isoformat(), "failures": [], "episodes": []}
    captures, rows, pending_facts = [], [], []
    background_count = 0
    run_labels, seen_execution = {}, set()
    row_lock, actor_stop = RLock(), Event()

    def now():
        return datetime.now(timezone.utc).isoformat()

    def wait(seconds, companion_count=None):
        def observed():
            with row_lock:
                return sum(item["speaker"] == "Spamuel" for item in rows) > companion_count
        wait_case(seconds, root, started, suite.case_timeout_seconds, actor_stop,
                  app=app, observed=observed if companion_count is not None else None)

    def wait_for_companion(companion_count):
        # A response allowance is not permission to send another unanswered
        # message. Normal use waits for actual speech, subject to STOP/deadline.
        def observed():
            with row_lock:
                return sum(item["speaker"] == "Spamuel" for item in rows) > companion_count
        wait_case(None, root, started, suite.case_timeout_seconds, actor_stop, app=app, observed=observed)

    def observe_world(label):
        from workflows.minecraft_body.mcp_client import build_minecraft_tools
        requested_at = now()
        response = build_minecraft_tools(stack.mcp_url)["minecraft_call"]("minecraft_observe", {"include_image": False})
        observed_at = now()
        # Keep the fixture's initial read. This timed read supplies the fact.
        name = "initial-live" if label == "initial" else label
        save_json(root / f"minecraft-{name}.json", response)
        save_json(root / f"minecraft-{name}-timing.json", {"requested_at": requested_at, "observed_at": observed_at})
        with row_lock:
            pending_facts.append(observation_fact(response, observed_at,
                f"{label} world snapshot only; not proof of earlier or later effects", f"host {label} read"))

    def collect_runs(owner, phase):
        """Freeze receipts and select narrow grounding facts after shutdown."""
        if not scenario.minecraft:
            return
        from workflows.minecraft_body.journal import read_actor_journal
        at = now()
        receipts = owner.supervisor.receipts()
        save_json(root / "managed-receipts" / f"{phase}.json", {"sampled_at": at, "runs": receipts})
        for run_id, run in receipts.items():
            if run.get("workflow_name") not in {"minecraft", "minecraft_project"}:
                continue
            label = run_labels.setdefault(run_id, f"r{len(run_labels)+1:04}")
            fact = managed_run_fact(run, at, f"run {label} status at sampling time only ({phase})", label)
            if fact is not None:
                pending_facts.append(fact)
            artifact = Path(run["artifact_path"]).resolve()
            if not artifact.is_relative_to(root.resolve()):
                raise ValueError("A managed-run artifact leaves this case's evidence directory.")
            for path in sorted(artifact.rglob("actor-journal.jsonl")):
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError("A body journal leaves this case's evidence directory.")
                for number, row in enumerate(read_actor_journal(path)):
                    key = (str(path), number)
                    if key in seen_execution:
                        continue
                    seen_execution.add(key)
                    fact = execution_fact(row, f"run {label}; body call at its recorded time only", label)
                    if fact is not None:
                        pending_facts.append(fact)
                    if row.get("event") == "result" and row.get("tool") == "minecraft_observe":
                        pending_facts.append(observation_fact({"ok": row.get("ok") is True, "result": row.get("result", {})},
                            datetime.fromtimestamp(row["time"], timezone.utc).isoformat(),
                            f"run {label}; world snapshot at this call only", "body-call result"))

    try:
        wait(0)
        stack = start_minecraft(root, True) if scenario.minecraft else None
        if stack is not None:
            observe_world("initial")
        pending_facts.extend(capability_facts(scenario, now()))
        ghost = shift_history(root) if scenario.chronology == "recent" else MemoryStore.from_yaml(SEED, checkpoint=root / "ghost.json")
        receipt["seed_records"] = len(ghost.all_knoxels)
        background = ghost.dialogue_history(user="Jesse")[-8:]
        rows.extend({"at": item["timestamp"], "speaker": item["speaker"], "text": item["text"]} for item in background)
        background_count = len(rows)
        actor = LazyStageModel(resolve_connection(load_config(config)), ModelActivation())

        class Capture(ChatCapture):
            def receive(self, envelope):
                """Preserve raw diagnostics, then select only admissible review facts."""
                with row_lock:
                    self.receive_locked(envelope)

            def receive_locked(self, envelope):
                """Assign public IDs and matching observations under the shared lock."""
                prior = len(self.rows)
                super().receive(envelope)
                if len(self.rows) == prior:
                    return
                row = self.rows[-1]
                rows.append(row)
                if row["speaker"] != "Spamuel":
                    return
                message_id = f"m{len(rows):04}"
                if scenario.state_evidence:
                    from pm.records.mental_state import FullMentalState
                    tick = row["metadata"].get("generation_state_tick")
                    if tick is not None:
                        with self.app.ghost.lock:
                            state = next(item for item in self.app.ghost.states if item.tick_id == tick)
                            emotions = FullMentalState.init_from_list(state.mental_state_latent_vec).state_emotions.model_dump()
                        pending_facts.append((row["at"], message_id,
                            "At composition of this message the accepted implemented emotion fields were "
                            + json.dumps({key: round(value, 4) for key, value in emotions.items()})
                            + ". This is implemented numerical state, not proof of subjective consciousness or a private thought transcript."))
                if scenario.minecraft:
                    for tool in envelope.payload.get("tool_receipts", []):
                        if tool["name"] == "minecraft_observe" and tool["executed"] and not tool["error"]:
                            pending_facts.append(observation_fact(tool["result"] or {}, row["at"], message_id,
                                "reply tool receipt, available before this delivery; exact call time unavailable"))

        def world_observation(envelope):
            with row_lock, app.ghost.lock:
                record = app.ghost.get_knoxel_by_id(envelope.payload["knoxel_id"])
                pending_facts.append(observation_fact({"ok": True, "result": record.metadata["evidence"]},
                    record.metadata["executed_at"], f"subsequent speech after m{len(rows):04} only",
                    "committed world observation"))

        def run_event(envelope):
            """Select event kinds only. Event details can contain private plans."""
            run_id = envelope.payload.get("run_id")
            run = app.supervisor.get(run_id)
            kind = envelope.payload.get("event")
            if run.get("workflow_name") not in {"minecraft", "minecraft_project"} or kind not in RUN_EVENTS:
                return
            with row_lock:
                label = run_labels.setdefault(run_id, f"r{len(run_labels)+1:04}")
                pending_facts.append((envelope.payload["executed_at"],
                    f"run {label}; event time only; not proof of world effects",
                    f"Actual Minecraft run {label} event: {kind}. Event details and private plans are not supplied."))

        def start_episode(index):
            nonlocal app
            wait(0)
            app = build_app(ghost=ghost, seed_path=SEED, runtime_root=root / "runtime" / f"episode-{index:02}",
                config_path=str(config), cognition="real", thought_period=30,
                world_poll_seconds=120 if scenario.minecraft else None, emit_reply=lambda text: None)
            observer = Capture(app, root)
            captures.append(observer)
            app.bus.subscribe("a18b_capture", lambda event: event.topic in {"user.input", "dialogue.reply"}, observer.receive)
            if scenario.minecraft:
                app.bus.subscribe("a18b_world", lambda event: event.topic == "minecraft.world", world_observation)
                app.bus.subscribe("a18b_run", lambda event: event.topic in {"workflow.event", "workflow.question"}, run_event)
            app.run()

        def compose_timed(packet):
            composing_at = now()
            text = compose(actor, packet, cancel=actor_stop)
            return text, composing_at, now()

        postrestarts = receipt["postrestarts"] = []
        start_episode(1)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="a18b_jesse") as actor_pool:
            for index, scene in enumerate(scenario.scenes, start=1):
                wait(0)
                if scene.restart_before:
                    app.shutdown()
                    app.join()
                    app.raise_if_error()
                    collect_runs(app, f"before-scene-{index:02}")
                    app = None
                    # Offline time belongs to the case deadline. STOP must
                    # interrupt it before another application starts.
                    wait(scene.offline_seconds)
                    ghost = MemoryStore.from_checkpoint(root / "ghost.json")
                    with row_lock:
                        postrestarts.append({"scene": index, "started_at": now(), "start_message_count": len(rows)})
                    start_episode(index)
                with row_lock:
                    start = len(rows)
                    before_scene = sum(item["speaker"] == "Spamuel" for item in rows)
                    event = {"scene": index, "started_at": now(), "sent": 0,
                             "burst_size": scene.burst_size, "await_reply": scene.await_reply,
                             **scene_coverage(rows, start, start, postrestart=bool(postrestarts))}
                    receipt["episodes"].append(event)
                try:
                    for number in range(scene.messages):
                        with row_lock:
                            mode = delivery_mode(scene, number)
                            packet = JessePacket(situation=scenario.human_situation, current_beat=scene.human_beat,
                                                 conversation=public_messages(rows), delivery_mode=mode)
                        path = root / "actor-inputs" / f"{index:02}-{number:02}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(packet.model_dump_json().encode("utf-8"))
                        pending = actor_pool.submit(compose_timed, packet)
                        while not pending.done():
                            wait(0.5)
                        text, composing_at, composed_at = pending.result()
                        wait(rng.uniform(*(OVERLAP_TYPING_SECONDS if mode == "overlap" else HUMAN_THINK_SECONDS)))
                        with row_lock:
                            before = sum(item["speaker"] == "Spamuel" for item in rows)
                        submitted_at = now()
                        # Scenes change the human intent, not the chat identity.
                        correlation = app.user_input.submit(text, session=f"a18b:{scenario.id}")
                        with row_lock:
                            offset, delivery = next((offset, row) for offset, row in enumerate(rows)
                                if row.get("correlation") == correlation and row["speaker"] == "Jesse")
                            send = actor_send_receipt(packet, composing_at=composing_at, composed_at=composed_at,
                                submitted_at=submitted_at, delivery=delivery, public_id=f"m{offset+1:04}")
                        send.update(delivery_mode=mode, await_reply=await_delivery(scene, number))
                        save_json(path.with_name(path.stem + "-sent.json"), send)
                        event["sent"] += 1
                        # Observing new speech sets the pace. It does not assign
                        # a causal reply to this particular human message.
                        if await_delivery(scene, number):
                            wait_for_companion(before)
                    wait(scene.quiet_seconds)
                    if index == len(scenario.scenes):
                        wait(suite.settle_seconds)
                    # Each speaking region gets an observation allowance.
                    # Any speech in the region suffices, without alternation.
                    if scene.messages and scene.await_reply:
                        wait(suite.response_wait_seconds, before_scene)
                    event["completed_at"] = now()
                finally:
                    with row_lock:
                        event.update(scene_coverage(rows, start, len(rows), postrestart=bool(postrestarts)))
                        for position, restart in enumerate(postrestarts):
                            end = postrestarts[position+1]["start_message_count"] if position+1 < len(postrestarts) else len(rows)
                            restart.update(scene_coverage(rows, restart["start_message_count"], end, postrestart=True))
                    save_json(root / "scene-coverage.json", {"episodes": receipt["episodes"], "postrestarts": postrestarts})
        wait(0)
        receipt["failures"].extend(coverage_failures(scenario, receipt["episodes"], postrestarts))
        receipt["runtime_ok"] = not receipt["failures"]
    except Exception as error:
        receipt["failures"].append(f"{type(error).__name__}: {error}")
        (root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        actor_stop.set()
        try:
            if app is not None:
                app.shutdown()
                app.join()
                if app.error is not None:
                    receipt["runtime_ok"] = False
                    receipt["failures"].append(f"{type(app.error).__name__}: {app.error}")
                collect_runs(app, "final")
            if stack is not None:
                observe_world("final")
        except Exception as error:
            receipt["runtime_ok"] = False
            receipt["failures"].append(f"cleanup: {type(error).__name__}: {error}")
            (root / "cleanup-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            try:
                if stack is not None:
                    stack.stop()
            except Exception as error:
                receipt["runtime_ok"] = False
                receipt["failures"].append(f"service cleanup: {type(error).__name__}: {error}")
            with row_lock:
                messages = public_messages(rows)
                facts = tuple(Fact(id=f"f{index:04}", at=at, about=about, text=text)
                              for index, (at, about, text) in enumerate(pending_facts, start=1))
            boundary = messages[background_count].id if len(messages) > background_count else None
            receipt["packet_sha256"] = {}
            for audience, packet in make_packets(scenario, suite, messages, facts, review_from_message=boundary).items():
                path = root / f"{audience}-packet.json"
                save_json(path, packet.model_dump(mode="json"))
                receipt["packet_sha256"][audience] = sha256_file(path)
            receipt["failures"].extend(coverage_failures(scenario, receipt["episodes"], receipt.get("postrestarts", [])))
            receipt["failures"] = list(dict.fromkeys(receipt["failures"]))
            receipt["runtime_ok"] = receipt["runtime_ok"] and not receipt["failures"]
            receipt.update(ended_at=now(), public_messages=len(messages),
                elapsed_seconds=round(time.monotonic() - started, 3),
                new_companion_messages=sum(row["speaker"] == "Spamuel" for observer in captures for row in observer.rows))
            save_json(root / "receipt.json", receipt)
    return receipt
