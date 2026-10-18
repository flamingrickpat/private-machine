"""The companion assembly: which bus modules run, which workflows are managed, and their wiring.

``build_app`` constructs one application in a fixed order. The runtime entry
points in ``pm.app`` receive this module; the runtime never imports it.
"""
from __future__ import annotations

from datetime import datetime
import time
from pathlib import Path

from pm.constants import Sampler
from pm.memory.store import MemoryStore
from pm.memory.store_tools import build_tools
from workflows.dialogue_assistant.host import AssistantDialogue
from workflows.demo.host import CheckSubprocess, DummyDialogue, DummyMainCogLoop, ScheduleSubprocesses, TaskTable, ToolPlusDialogue
from workflows.main_cognition.host import InnerThoughtLoop, MainCogLoop
from workflows.self_management.host import SelfManagement
from workflows.steward.host import StewardModule
from workflows.dialogue_story.host import StoryDialogue
from workflows.public_speech.user_input import UserInputProxy
from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_id
from workflows.memory_consolidation.artifact_consolidation import Consolidator
from pm.runs.supervisor import ManagedRunSupervisor
from pm.runs.workflow_spec import WorkflowSpec
from pm.host.application import Application
from workflows.steward.gate import Steward
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.bash_view import MemoryBashTarget
from pm.runs.workflow_module import PROJECT_ROOT, WorkflowModule
from pm.app.operator_config import default_config_path
from pm.paths import REPO_ROOT
from workflows.cognitive_control.schemas import ControlConfig


BANNER = "DUMMY COGNITION \u2014 no LLM calls\n"


REAL_BANNER = "REAL COGNITION \u2014 model activates on first request\n"


DEFAULT_SEED = REPO_ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"


def _managed_workflows(stage_model=None, embeddings=None, ghost=None) -> dict:
    from workflows.minecraft.host import build_minecraft_module_tools, persona_context, project_memory_config
    from workflows.note_task.steps import build_note_tools
    from workflows.memory_reconsolidation.host import build_reconsolidation_tools

    base = PROJECT_ROOT / "workflows" / "local_task"
    return {
        "local_task": WorkflowSpec(base / "local_task.workflow.md", namespace="local_task"),
        "local_task_question": WorkflowSpec(
            base / "local_task_question.workflow.md", namespace="local_task_question",
        ),
        "note_task": WorkflowSpec(
            PROJECT_ROOT / "workflows" / "note_task" / "note_task.workflow.md",
            namespace="note_task", tools_factory=build_note_tools(stage_model),
        ),
        # The character sorts through its memories ("sleeps on it"). The tool
        # runs pm.memory.reconsolidation on the live store; one run at a time.
        "memory_reconsolidation": WorkflowSpec(
            PROJECT_ROOT / "workflows" / "memory_reconsolidation" / "memory_reconsolidation.workflow.md",
            namespace="memory_reconsolidation",
            tools_factory=build_reconsolidation_tools(ghost, stage_model, embeddings),
            singleton=True,
        ),
        "minecraft": WorkflowSpec(
            PROJECT_ROOT / "workflows" / "minecraft" / "minecraft.workflow.md",
            namespace="minecraft",
            tools_factory=build_minecraft_module_tools(
                stage_model.complete if stage_model is not None else None,
                stage_model.measure_context if stage_model is not None else None,
            ),
            resumable=True,
            singleton=True, singleton_key="minecraft_body",
            request_context=persona_context,
        ),
        "minecraft_project": WorkflowSpec(
            PROJECT_ROOT / "workflows" / "minecraft_project" / "minecraft_project.workflow.md",
            namespace="minecraft_project",
            tools_factory=build_minecraft_module_tools(stage_model.complete if stage_model is not None else None,
                stage_model.measure_context if stage_model is not None else None),
            resumable=True,
            singleton=True, singleton_key="minecraft_body",
            request_context=persona_context,
            memory_config_factory=(lambda store, cancel: project_memory_config(store, cancel, embeddings))
                if stage_model is not None else None,
        ),
    }


def _cognition_workflow_options(supervisor) -> list[dict]:
    """Build the main-cognition action space from the live configuration.

    Dialogue and self-management are always available. Managed workflow names
    come from the supervisor, so a configured workflow can steer work without
    a code change.
    """
    options = [
        {"name": "dialogue_story", "description": "Generate one in-character Story reply."},
        {"name": "dialogue_assistant", "description": "Assistant reply with scoped tools. Inventory questions use arguments.minecraft_read=inventory; position, health, and surroundings use minecraft_read=state. Both make one live read without a managed task."},
        {"name": "self_management", "description": "Start, status, update, terminate, or explicitly resume an interrupted body run."},
        {"name": "focus_management", "description": "Private deliberate mental work: scoped recall of preferences, conversations, or goals, and a temporary focus or goal proposal. Set arguments.instruction to the mental task. Advice returns through ordinary appraisal. It commits no emotion or memory change directly."},
    ]
    for name in sorted(supervisor.workflows):
        descriptions = {"local_task": "Control worker. Writes task telemetry; cannot execute arbitrary instructions or create a requested note.",
                        "note_task": "Real pm-coder note writer. Exact UTF-8 byte verification. Optional arguments.note_spec has path and text; path stays in outbox/.",
                        "memory_reconsolidation": "Sort through memories and sleep on them: builds topic and day summaries, facts, and updated self and relationship narratives from the stored history. Slow; runs in the background. Use ToolCall when the character decides to sort through memories, sleep on what happened, or reflect on the past days.",
                        "minecraft": "One sustained survival specialist leaf. Standalone completion requires an explicit supported verifier, currently wood_return. Use minecraft_project for other goals that need verified completion.",
                        "minecraft_project": "Generic Minecraft project planner. Durable subtasks, verified prerequisites, fresh specialists, and independent world verification. Use for multi-step goals. Inventory questions use direct Assistant observation."}
        description = descriptions.get(name, f"Managed workflow {name}.")
        spec = supervisor.workflows[name]
        if spec.singleton:
            description += (f" One active run for {spec.singleton_key or name}. "
                            "A busy start returns its owner; choose wait, update, or terminate through self_management.")
        options.append({"name": name, "description": description})
    return options


def _cognition_modules(cognition, *, ghost, target, bus, on_error, clock,
                      runtime_root, workflow_paths, stage_model, config_path,
                      supervisor, codelet_definitions=None, thought_period=30.0, embeddings=None,
                      workflow_models=None):
    """Return the (main_cog, story, assistant, self_management, inner, mental) modules."""
    if cognition == "real":
        from workflows.dialogue_story.host import build_story_tools
        from workflows.dialogue_assistant.host import build_assistant_tools
        from workflows.main_cognition.host import build_main_cognition_tools
        from workflows.self_management.host import build_self_management_tools
        from workflows.inner_thought.host import PrivateThoughtLoop, build_inner_thought_tools

        if stage_model is None:
            from pm.llm.stage_model import LazyStageModel
            from pm.llm.activation import ModelActivation
            from pm.llm.config import load_config, resolve_connection

            config = load_config(config_path)
            from workflows.inference_settings import CHAT_CONCURRENCY
            stage_model = LazyStageModel(resolve_connection(config), ModelActivation(), max_concurrency=CHAT_CONCURRENCY)
        main_cog = MainCogLoop(
            "main_cog", bus, on_error, clock, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "main_cognition" / "main_cognition.workflow.md",
            tools_factory=build_main_cognition_tools(ghost, runtime_root, stage_model.complete,
                                                   cancel_provider=lambda: main_cog.run_stop,
                                                   embeddings=embeddings),
            runtime_root=runtime_root,
            ghost=ghost,
            workflow_catalogue=_cognition_workflow_options(supervisor),
            embedding_config_path=config_path,
            codelet_definitions=codelet_definitions,
            # Agent codelets open pm-coder sessions through classification sessions.
            workflow_models=workflow_models,
        )
        story = StoryDialogue(
            "dialogue", bus, on_error, clock, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "dialogue_story" / "dialogue_story.workflow.md",
            tools_factory=build_story_tools(stage_model, cancel_provider=lambda: story.run_stop),
            runtime_root=runtime_root,
            ghost=ghost,
        )
        assistant = AssistantDialogue(
            "assistant", bus, on_error, clock, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "dialogue_assistant" / "dialogue_assistant.workflow.md",
            tools_factory=build_assistant_tools(stage_model, supervisor, cancel_provider=lambda: assistant.run_stop),
            runtime_root=runtime_root,
            ghost=ghost,
        )
        self_management = SelfManagement(
            "self_management", bus, on_error, clock, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "self_management" / "self_management.workflow.md",
            tools_factory=build_self_management_tools(supervisor),
            runtime_root=runtime_root,
            ghost=ghost,
        )
        self_management.managed_workflows = frozenset(supervisor.workflows)
        # The private loop owns its timer in real mode. Its triggers decide
        # whether a tick produces a thought; an idle tick costs no model call.
        inner_thought = PrivateThoughtLoop(
            "inner_thought", bus, on_error, clock, thought_period,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "inner_thought" / "inner_thought.workflow.md",
            tools_factory=build_inner_thought_tools(ghost, runtime_root, stage_model.complete,
                                                  cancel_provider=lambda: inner_thought.run_stop,
                                                  embeddings=embeddings),
            runtime_root=runtime_root,
            ghost=ghost,
        )
        # The deliberate mental-work agent runs one scoped pm-coder session and
        # returns advice. Main appraisal still owns every accepted change.
        from workflows.focus_management.host import FocusManagement
        focus_management = FocusManagement(
            "focus_management", bus, on_error, clock, 1.0,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "focus_management" / "focus_management.workflow.md",
            runtime_root=runtime_root, ghost=ghost, embeddings=embeddings,
            workflow_models=workflow_models,
        )
        return main_cog, story, assistant, self_management, inner_thought, focus_management
    main_cog = DummyMainCogLoop(
        "main_cog", bus, on_error, clock, None,
        **workflow_paths("main_cognition"),
    )
    dialogue = DummyDialogue(
        "dialogue", bus, on_error, clock, None,
        **workflow_paths("dialogue"),
    )
    inner_thought = InnerThoughtLoop(
        "inner_thought", bus, on_error, clock, thought_period,
        **workflow_paths("inner_thought"),
    )
    return main_cog, dialogue, None, None, inner_thought, None


def build_app(
    seed_path,
    runtime_root,
    thought_period: float = 30.0,
    task_period: float = 0.1,
    clock=time.monotonic,
    emit_reply=print,
    ghost=None,
    checkpoint_path=None,
    *,
    cognition: str = "dummy",
    stage_model=None,
    config_path: str | None = None,
    workflow_models: WorkflowModels | None = None,
    record_lifecycle: bool = True,
    world_poll_seconds: float | None = 120.0,
    world_quiet_seconds: float = 120.0,
    codelet_definitions: str | None = None,
    cognitive_control_config: ControlConfig | dict | None = None,
) -> Application:
    """Construct an application without service activation.

    Real cognition shares its activation owner with lazy workflow classification.
    A custom stage model can supply ``workflow_models`` explicitly. The caller
    owns application execution, shutdown, and its separate checkpoint path.
    Real cognition polls structured world facts after 120 seconds of silence.
    ``world_poll_seconds=None`` disables that source. Polling never chooses work.
    ``codelet_definitions`` names a codelet definitions file for real
    cognition; ``None`` uses ``workflows/main_cognition/codelets.yaml``.
    ``cognitive_control_config`` supplies wait, focus, mental-session, and
    private-episode limits. A one-second scheduler checks waits without inference.
    """
    if cognition not in {"real", "dummy"}:
        raise ValueError("cognition must be 'real' or 'dummy'.")
    if world_poll_seconds is not None and world_poll_seconds <= 0:
        raise ValueError("world_poll_seconds must be positive or None.")
    if world_quiet_seconds < 0:
        raise ValueError("world_quiet_seconds must not be negative.")
    if ghost is not None and checkpoint_path is not None:
        raise ValueError("Supply a Ghost or a checkpoint path, not both.")
    resolved_config = config_path or default_config_path()
    if cognition == "real" and stage_model is None:
        from pm.llm.stage_model import LazyStageModel
        from pm.llm.activation import ModelActivation
        from pm.llm.config import load_config, resolve_connection

        model_config = load_config(resolved_config)
        model_owner = ModelActivation()
        from workflows.inference_settings import CHAT_CONCURRENCY
        stage_model = LazyStageModel(resolve_connection(model_config), model_owner, max_concurrency=CHAT_CONCURRENCY)
        if workflow_models is None:
            workflow_models = WorkflowModels(model_config, model_owner)

    if ghost is None:
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        if checkpoint is not None and checkpoint.exists():
            ghost = MemoryStore.from_checkpoint(checkpoint)
        else:
            ghost = MemoryStore.from_yaml(Path(seed_path), checkpoint=checkpoint)
    target = MemoryBashTarget(ghost, mode="read")
    bus = BusSystem()
    tasks = TaskTable(clock)
    app = Application(bus, [])

    on_error = app.on_error
    runtime_root = Path(runtime_root)

    consolidator = Consolidator(ghost, clock=clock)
    context_embeddings = None
    if cognition == "real":
        from pm.llm.config import load_config
        from pm.memory.embeddings import EmbeddingService
        context_embeddings = EmbeddingService(load_config(resolved_config).embedding)
    supervisor = ManagedRunSupervisor(
        ghost,
        runtime_root=runtime_root,
        workflows=_managed_workflows(stage_model, context_embeddings, ghost),
        clock=clock,
        task_root=runtime_root / "managed_tasks",
        on_error=on_error,
        workflow_models=workflow_models,
    )

    def _on_terminal(receipt):
        consolidator.consolidate(receipt["run_id"], receipt["artifact_path"])

    def _event_sink(event):
        if event.get("event") == "gameplay_progress":
            run = supervisor.status(event["run_id"])
            consolidator.consolidate(event["run_id"], run["artifact_path"])
        try:
            run = supervisor.status(event["run_id"])
            parent = run.get("parent_correlation", "")
            event = {**event, "artifact_path": run.get("artifact_path", ""),
                     "executed_at": datetime.utcnow().isoformat()}
        except Exception:
            parent = ""
        topic = "workflow.question" if event.get("event") == "waiting_for_input" else "workflow.event"
        bus.put(Envelope(
            id=new_id(), source="managed_run", kind="telemetry", topic=topic,
            correlation_id=parent, payload=dict(event),
        ))

    supervisor.on_terminal = _on_terminal
    supervisor.event_sink = _event_sink
    steward = Steward(supervisor, clock=clock)

    def workflow_paths(name: str):
        return {
            "working_target": target,
            "workflow": PROJECT_ROOT / "workflows" / "demo" / f"{name}.workflow.md",
            "tools_factory": build_tools,
            "runtime_root": runtime_root,
            "ghost": ghost,
        }

    user_input = UserInputProxy("user_input", bus, on_error, clock, ghost, emit_reply)
    main_cog, dialogue, assistant, self_management, inner_thought, focus_management = _cognition_modules(
        cognition, ghost=ghost, target=target, bus=bus, on_error=on_error,
        clock=clock, runtime_root=runtime_root, workflow_paths=workflow_paths,
        stage_model=stage_model, config_path=resolved_config, supervisor=supervisor,
        codelet_definitions=codelet_definitions, thought_period=thought_period,
        embeddings=context_embeddings, workflow_models=workflow_models,
    )
    scheduler = ScheduleSubprocesses(
        "scheduler", bus, on_error, clock, task_period,
        **workflow_paths("schedule"), tasks=tasks,
    )
    status = CheckSubprocess(
        "status", bus, on_error, clock, None,
        **workflow_paths("status"), tasks=tasks,
    )
    tool_dialogue = ToolPlusDialogue(
        "tool_dialogue", bus, on_error, clock, None,
        **workflow_paths("tool_dialogue"),
    )
    steward_module = None
    if cognition == "real":
        from workflows.steward.host import build_steward_module_tools

        steward_module = StewardModule(
            "steward", bus, on_error, clock, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "steward" / "steward.workflow.md",
            tools_factory=build_steward_module_tools(steward),
            runtime_root=runtime_root, ghost=ghost,
        )

    modules = [
        user_input, main_cog, dialogue, inner_thought,
        scheduler, status, tool_dialogue,
    ]
    if cognition == "real":
        from workflows.cognitive_control.host import CognitiveControl

        control = CognitiveControl(
            "cognitive_control", bus, on_error, clock, 1.0, ghost=ghost,
            config=ControlConfig.model_validate(cognitive_control_config or {}),
        )
        control.cancel_private = inner_thought.cancel_run
        main_cog.cognitive_control = control
        inner_thought.cognitive_control = control
        def release_private_results(envelopes):
            """Release unpublished private results after a discarded main cycle."""
            inner_thought.release_results(envelopes)
            if focus_management is not None:
                focus_management.release_results(envelopes)
        main_cog.release_private_results = release_private_results
        if focus_management is not None:
            focus_management.cognitive_control = control
            modules.append(focus_management)
            app.focus_management = focus_management
        modules.append(control)
        app.cognitive_control = control
        if world_poll_seconds is not None:
            from workflows.minecraft_body.world_events import WorldEvents
            from workflows.minecraft_body.mcp_client import build_minecraft_tools, DEFAULT_MCP_URL
            from workflows.minecraft_body.lease import BODY_LEASES
            import os
            endpoint = os.environ.get("MINECRAFT_MCP_URL", DEFAULT_MCP_URL)
            transport = build_minecraft_tools(endpoint, request_timeout_seconds=30)["minecraft_call"]

            def observe_world():
                # Reads share the command lock with actor calls. They never
                # claim the mutation lease or bypass an in-flight action.
                from pm_workflows.runtime import WorkflowTerminated
                try:
                    with BODY_LEASES.command(endpoint, cancel=world_events.stop_event):
                        return transport("minecraft_observe", {"include_image": False})
                except WorkflowTerminated:
                    return {"ok": False, "cancelled": True}

            world_events = WorldEvents(
                "world_events", bus, on_error, clock, world_poll_seconds,
                ghost=ghost, observe=observe_world,
                journal=runtime_root / "world-events.jsonl", quiet_seconds=world_quiet_seconds,
            )
            modules.append(world_events)
            app.world_events = world_events
        from workflows.minecraft_body.capabilities import capability_snapshot
        from workflows.introspection.snapshot import build_capability_overlay, overlay_public
        from workflows.public_speech.reply_recovery import ReplyRecovery

        def provider():
            # The overlay is rebuilt by the private loop and cached on the app.
            # A public reply reads the cache, so no repository scan delays speech.
            snapshot = capability_snapshot(ghost, supervisor.workflows)
            snapshot["self_model"] = overlay_public(getattr(app, "capability_overlay", None))
            return snapshot

        main_cog.capability_provider = provider
        def retrieval_query(text):
            from workflows.main_cognition.model_json import StageOutputError, extract_json
            raw = stage_model.complete([
                {"role": "system", "content": "Rewrite this message as a short memory search query. "
                 "Search for original user reports or concrete executed events, not earlier answers to recall requests. "
                 "Remove boilerplate about remembering, recorded history, and supporting conversations. "
                 "Retain actual topics, outcomes, corrections, and dates. For an unspecified failed method, search for a practical attempt "
                 "that did not work and its reported outcome. Never invent a specific topic. "
                 "Also choose evidence_use: event for factual recall, preferences, or past events; utterance for what someone said; "
                 "general for other requests or uncertainty. A request to evaluate newly reported advice is general, "
                 "even when it mentions a familiar topic. What do you actually know about that claim asks for supported "
                 "knowledge, not a past conversation. Use event only when the user asks to recall an event, retained "
                 "preference, name, or plan; use utterance when recalling what someone previously said. "
                 "Return only JSON with query and evidence_use."},
                {"role": "user", "content": text},
            ], temperature=Sampler.TempInstruct, enable_thinking=False)
            try:
                result = extract_json(raw)
            except StageOutputError:
                return {"query": text, "evidence_use": "general", "status": "literal_fallback"}
            query = result.get("query")
            use = result.get("evidence_use")
            if not isinstance(query, str) or not query.strip() or use not in {"event", "utterance", "general"}:
                return {"query": text, "evidence_use": "general", "status": "literal_fallback"}
            return {"query": query, "evidence_use": use, "status": "rewritten"}
        main_cog.retrieval_query_provider = retrieval_query
        dialogue.capability_provider = provider
        assistant.capability_provider = provider
        # Context planning for public replies shares one embedding service and
        # its disk cache. Construction makes no network request.
        dialogue.context_embeddings = context_embeddings
        assistant.context_embeddings = context_embeddings
        recovery = ReplyRecovery("reply_recovery", bus, on_error, clock, None, ghost=ghost, supervisor=supervisor)
        recovery.paused_source_ids = control.paused_source_ids
        modules.append(recovery)
        app.reply_recovery = recovery
    if assistant is not None:
        modules.append(assistant)
    if self_management is not None:
        modules.append(self_management)
    if steward_module is not None:
        modules.append(steward_module)
    for module in modules:
        if isinstance(module, WorkflowModule):
            module.workflow_models = workflow_models
        bus.subscribe(module.module_id, module.accepts, module.deliver)
    app.modules = modules

    app.user_input = user_input
    app.main_cog = main_cog
    app.dialogue = dialogue
    app.assistant = assistant
    app.self_management = self_management
    app.inner_thought = inner_thought
    app.scheduler = scheduler
    app.status = status
    app.tool_dialogue = tool_dialogue
    app.steward_module = steward_module
    app.ghost = ghost
    app.bus = bus
    app.tasks = tasks
    app.supervisor = supervisor
    app.stop_workers = supervisor.stop

    def finish_workers():
        """Join all owned work before recording a successful clean stop."""
        supervisor.shutdown()
        if record_lifecycle and app.error is None:
            from pm.app.lifecycle_records import record_shutdown
            record_shutdown(ghost, boot_id=app.lifecycle_boot["boot_id"])

    app.join_workers = finish_workers
    app.consolidator = consolidator
    app.steward = steward
    app.workflow_models = workflow_models

    # A18: give the private loop a source grant and a fresh capability overlay.
    # The dialogue modules read the cached public projection; the loop rebuilds
    # it each cycle, so a source or configuration change becomes visible.
    if cognition == "real":
        from workflows.introspection.snapshot import build_capability_overlay

        active_module_ids = frozenset(module.module_id for module in modules)

        def rebuild_overlay():
            previous = getattr(app, "capability_overlay", None)
            overlay = build_capability_overlay(
                ghost=ghost, active_modules=active_module_ids, config_path=resolved_config)
            # Keep the prior overlay as history so a changed source or
            # configuration revision marks its dependent belief stale.
            overlay.previous = previous
            app.capability_overlay = overlay
            return overlay

        app.rebuild_capability_overlay = rebuild_overlay
        rebuild_overlay()
        if inner_thought is not None:
            inner_thought.inspector_provider = rebuild_overlay

    # Durable lifecycle evidence. The boot event feeds the first main cycle so
    # the companion can appraise a real offline gap. Only real cognition runs
    # that cycle, so the dummy fixture records the event without a stimulus.
    from pm.app.lifecycle_records import record_startup
    boot = record_startup(ghost) if record_lifecycle else None
    app.lifecycle_boot = boot
    if cognition == "real":
        # Queue original pending requests before the startup cycle. Otherwise
        # startup can retrieve that request and execute it without its cause,
        # while asynchronous recovery dispatches it again in another cycle.
        recovery.resume()
        inner_thought.resume()
        if focus_management is not None:
            focus_management.resume()
    if cognition == "real" and boot is not None:
        bus.put(Envelope(
            id=new_id(), source="system", kind="data", topic="system.lifecycle",
            correlation_id="", payload={"knoxel_id": boot["knoxel_id"]},
        ))
    return app

