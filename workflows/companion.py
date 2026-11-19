"""The companion assembly: which bus modules run, which workflows are managed, and their wiring.

This module owns the composition of one running companion. It knows nothing
about cognition itself; it decides which bus modules exist, which managed
workflows the supervisor may start, and which capabilities each module
receives. The pieces it assembles live in ``pm/`` (runtime mechanics) and in
the sibling ``workflows/`` packages (behavior).

Owned state, all local to ``build_app``:

- the ``MemoryStore`` (``ghost``) and its embedding service,
- the ``BusSystem`` and the ordered module list that subscribes to it,
- the ``ManagedRunSupervisor`` and its workflow specifications,
- the ``Steward`` gate and the consolidation owner for terminal runs.

Inputs: a persona seed or checkpoint, a runtime root, a cognition mode
(``real`` or ``dummy``), and the model configuration path. Output: one
``Application``; the caller owns execution, shutdown, and checkpoints.

Boundaries and side effects: real cognition loads the model configuration,
initialises embeddings, probes a configured Minecraft body, and records
lifecycle evidence. Dummy cognition builds fixed prototype modules and makes
no model call. Assembly order matters: modules are appended in the order the
bus should see them, and the overlay rebuild, run recovery, and parked-run
restoration happen only after every module exists.

Invariants:

- ``build_app`` raises on an unusable argument combination instead of
  substituting a quieter behavior.
- Every managed workflow name in the action catalogue comes from the
  supervisor, so configuration alone can add or remove steerable work.
- A module receives its collaborators before it subscribes to the bus.

Reading map: ``_managed_workflows`` (managed run specifications),
``_cognition_workflow_options`` (the main-cognition action catalogue),
``_cognition_modules`` (the cognitive module set per cognition mode),
``_minecraft_endpoint`` (body endpoint resolution), then ``build_app``.
"""
from __future__ import annotations


from pm.utils.operating_system import OperatingSystem
from pm import vclock

from datetime import datetime
import time
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

from pm.constants import Sampler
from pm.memory.store import MemoryStore
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
from workflows.main_cognition.catalogue import WorkflowCatalogue
from pm.runs.workflow_module import PROJECT_ROOT, CognitiveWorkflow, WorkflowModule
from pm.app.operator_config import default_config_path
from pm.paths import REPO_ROOT
from workflows.cognitive_control.schemas import ControlConfig
from pm.llm.mcp_config import ToolCallConfig

if TYPE_CHECKING:
    from pm.introspection.capabilities import CapabilityOverlay
    from pm.llm.config import ModelConfig
    from pm.llm.stage_model import LazyStageModel
    from pm.memory.embeddings import EmbeddingService
    from workflows.focus_management.host import FocusManagement
    from workflows.tool_call.catalogue import ToolCatalogue


BANNER = "DUMMY COGNITION \u2014 no LLM calls\n"


REAL_BANNER = "REAL COGNITION \u2014 model activates on first request\n"


DEFAULT_SEED = REPO_ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"


def _managed_workflows(
    stage_model: LazyStageModel | None = None,
    embeddings: EmbeddingService | None = None,
    ghost: MemoryStore | None = None,
    tool_call: ToolCallConfig | None = None,
    minecraft_endpoint: str | None = None,
    *,
    enable_minecraft: bool = True,
) -> dict[str, WorkflowSpec]:
    """Return the managed workflow specifications, keyed by workflow name.

    The mapping is keyed by the workflow name that the supervisor, the action
    catalogue, and the prompt descriptions all use; membership is dynamic, so
    it stays a real dict rather than a fixed record.

    ``tool_call`` is registered only when its configuration names at least
    one MCP server. Without tools, routing never offers it, so the companion
    cannot claim tools it lacks. A missing header variable raises
    McpConfigError here, at application start.

    ``minecraft_endpoint`` is the body endpoint that the assembly resolved from
    the ``minecraft`` section of the model configuration. The body workflows
    take it as an argument, so a real run needs no environment variable.

    ``stage_model`` and ``embeddings`` are None in dummy cognition; the tool
    factories then report the missing model when a run actually needs one.
    """
    from workflows.minecraft.host import build_minecraft_module_tools, MinecraftContextOwner, ProjectMemoryOwner
    from workflows.memory_reconsolidation.host import build_reconsolidation_tools

    base = PROJECT_ROOT / "workflows" / "local_task"
    optional: dict[str, WorkflowSpec] = {}
    if tool_call is not None and tool_call.mcp_servers:
        from workflows.tool_call.host import WORKFLOW_NAME, build_workflow_spec
        optional[WORKFLOW_NAME] = build_workflow_spec(tool_call)
    result = {**optional,
        "local_task": WorkflowSpec(base / "local_task.workflow.md", namespace="local_task"),
        "local_task_question": WorkflowSpec(
            base / "local_task_question.workflow.md", namespace="local_task_question",
        ),
        "note_task": WorkflowSpec(
            PROJECT_ROOT / "workflows" / "note_task" / "note_task.workflow.md",
            namespace="note_task",
        ),
        # The character sorts through its memories ("sleeps on it"). The tool
        # runs pm.memory.reconsolidation on the live store; one run at a time.
        "memory_reconsolidation": WorkflowSpec(
            PROJECT_ROOT / "workflows" / "memory_reconsolidation" / "memory_reconsolidation.workflow.md",
            namespace="memory_reconsolidation",
            runtime_owner=build_reconsolidation_tools(ghost, stage_model, embeddings),
            singleton=True,
        ),
        "minecraft": WorkflowSpec(
            PROJECT_ROOT / "workflows" / "minecraft" / "minecraft.workflow.md",
            namespace="minecraft",
            resumable=True,
            singleton=True, singleton_key="minecraft_body",
            request_context_owner=MinecraftContextOwner(),
            runtime_owner=build_minecraft_module_tools(stage_model, mcp_url=minecraft_endpoint),
        ),
        "minecraft_project": WorkflowSpec(
            PROJECT_ROOT / "workflows" / "minecraft_project" / "minecraft_project.workflow.md",
            namespace="minecraft_project",
            resumable=True,
            singleton=True, singleton_key="minecraft_body",
            request_context_owner=MinecraftContextOwner(),
            runtime_owner=build_minecraft_module_tools(stage_model, mcp_url=minecraft_endpoint),
            memory_config_owner=ProjectMemoryOwner(embeddings) if stage_model is not None else None,
        ),
    }
    if not enable_minecraft:
        result.pop("minecraft")
        result.pop("minecraft_project")
    return result




class DummyWorkflowPaths(TypedDict):
    """Keyword arguments that every dummy prototype module constructor shares.

    The prototype modules differ only in their loop details, so the assembly
    builds this one keyword set per workflow folder and unpacks it.
    """

    working_target: MemoryBashTarget
    workflow: Path
    tools_factory: None
    runtime_root: Path
    ghost: MemoryStore


class CognitionModules(NamedTuple):
    """The cognitive modules of one assembly, named by their role.

    Dummy cognition leaves the optional roles as None instead of substituting
    a stand-in module, so a caller that needs one fails visibly.
    """

    main_cog: CognitiveWorkflow
    dialogue: CognitiveWorkflow
    assistant: CognitiveWorkflow | None
    self_management: CognitiveWorkflow | None
    inner_thought: CognitiveWorkflow
    focus_management: FocusManagement | None


def _cognition_modules(
    cognition: str,
    *,
    ghost: MemoryStore,
    target: MemoryBashTarget,
    application: Application,
    consolidator: Consolidator,
    runtime_root: Path,
    workflow_paths: Callable[[str], DummyWorkflowPaths],
    stage_model: LazyStageModel | None,
    config_path: str,
    supervisor: ManagedRunSupervisor,
    codelet_definitions: str | None = None,
    thought_period: float = 30.0,
    embeddings: EmbeddingService | None = None,
    workflow_models: WorkflowModels | None = None,
    minecraft_endpoint: str | None = None,
) -> CognitionModules:
    """Return the cognitive modules of one assembly, by role.

    ``minecraft_endpoint`` is the resolved body endpoint, or None when this
    character has no Minecraft body. The assistant reads the world through it.
    ``stage_model`` is None only in dummy cognition, which builds fixed
    prototype loops and no private thought planning.
    """
    if cognition == "real":
        from pm.llm.config import load_config
        from workflows.dialogue_assistant.host import AssistantMcpServices
        from workflows.main_cognition.host import MainCognitionOptions
        from workflows.self_management.host import build_self_management_tools
        from workflows.inner_thought.host import PrivateThoughtLoop

        if stage_model is None:
            from pm.llm.stage_model import LazyStageModel
            from pm.llm.activation import ModelActivation
            from pm.llm.config import load_config, resolve_connection

            config = load_config(config_path)
            stage_model = LazyStageModel(resolve_connection(config), ModelActivation(), capture_options=config.renderer_capture)
        main_cog = MainCogLoop(
            "main_cog", application, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "main_cognition" / "main_cognition.workflow.md",
            runtime_root=runtime_root,
            stage_model=stage_model,
            ghost=ghost,
            consolidator=consolidator,
            options=MainCognitionOptions(
                workflow_catalogue=tuple(WorkflowCatalogue(supervisor, None).options()),
                embedding_config_path=config_path, ghost_backups=load_config(config_path).ghost_backups,
                codelet_definitions=codelet_definitions, workflow_models=workflow_models,
                codelet_embeddings=embeddings),
        )
        main_cog.embedding_service = embeddings
        story = StoryDialogue(
            "dialogue", application, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "dialogue_story" / "dialogue_story.workflow.md",
            tools_factory=None,
            runtime_root=runtime_root,
            stage_model=stage_model,
            ghost=ghost,
        )
        assistant = AssistantDialogue(
            "assistant", application, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "dialogue_assistant" / "dialogue_assistant.workflow.md",
            tools_factory=None,
            mcp_services=AssistantMcpServices(supervisor, minecraft_endpoint),
            runtime_root=runtime_root,
            stage_model=stage_model,
            ghost=ghost,
        )
        admission = getattr(stage_model, "admission", None)
        main_cog.inference_admission = admission
        story.inference_admission = admission
        assistant.inference_admission = admission
        self_management = SelfManagement(
            "self_management", application, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "self_management" / "self_management.workflow.md",
            tools_factory=build_self_management_tools(supervisor),
            runtime_root=runtime_root,
            stage_model=stage_model,
            ghost=ghost,
        )
        self_management.managed_workflows = frozenset(supervisor.workflows)
        # The private loop owns its timer in real mode. Its triggers decide
        # whether a tick produces a thought; an idle tick costs no model call.
        inner_thought = PrivateThoughtLoop(
            "inner_thought", application, thought_period,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "inner_thought" / "inner_thought.workflow.md",
            runtime_root=runtime_root,
            stage_model=stage_model,
            ghost=ghost,
            thought_delay=load_config(config_path).thought_delay,
        )
        inner_thought.inference_admission = admission
        # Deliberate mental work enters the private slot of the loop above.
        # This module admits the selected action and delivers its result; it
        # runs no model. Main appraisal still owns every accepted change.
        from workflows.focus_management.host import FocusManagement
        focus_management = FocusManagement("focus_management", application, 1.0, ghost=ghost)
        return CognitionModules(main_cog, story, assistant, self_management, inner_thought, focus_management)
    main_cog = DummyMainCogLoop(
        "main_cog", application, None,
        **workflow_paths("main_cognition"),
    )
    dialogue = DummyDialogue(
        "dialogue", application, None,
        **workflow_paths("dialogue"),
    )
    inner_thought = InnerThoughtLoop(
        "inner_thought", application, thought_period,
        **workflow_paths("inner_thought"),
    )
    return CognitionModules(main_cog, dialogue, None, None, inner_thought, None)


def _minecraft_endpoint(loaded: ModelConfig) -> str | None:
    """Return the Minecraft body endpoint for this run, or None for no body.

    The ``minecraft`` section of the model configuration is the normal source,
    so a real session needs no environment variable. ``MINECRAFT_MCP_URL``
    stays a test override: the real gates and the live tests start their own
    server on a random port and name it. A character with no body omits the
    section and gets no world poll and no body tools.
    """
    override = OperatingSystem.environment.get("MINECRAFT_MCP_URL")
    if override:
        return override
    return loaded.minecraft.mcp_url if loaded.minecraft is not None else None


def build_app(
    seed_path: str | Path | None,
    runtime_root: str | Path,
    thought_period: float = 30.0,
    task_period: float = 0.1,
    clock: vclock.SystemClock | None = None,
    emit_reply: Callable[[str], None] = print,
    ghost: MemoryStore | None = None,
    checkpoint_path: str | Path | None = None,
    *,
    cognition: str = "dummy",
    stage_model: LazyStageModel | None = None,
    config_path: str | None = None,
    workflow_models: WorkflowModels | None = None,
    record_lifecycle: bool = True,
    world_poll_seconds: float | None = 120.0,
    world_quiet_seconds: float = 120.0,
    codelet_definitions: str | None = None,
    cognitive_control_config: ControlConfig | dict[str, Any] | None = None,
    tool_call: ToolCallConfig | None = None,
    enable_minecraft: bool = True,
    bus_observer: Callable[[Envelope, str], None] | None = None,
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
    ``None`` reads the ``cognitive_control`` section of the model configuration,
    and an empty section uses the operator defaults.
    ``tool_call`` gives the companion MCP tools. ``None`` reads the
    ``tool_call`` section of the model configuration in real cognition;
    dummy cognition has no tools.
    ``ghost``, ``checkpoint_path``, and ``seed_path`` select where the memory
    store comes from: an existing store, a checkpoint to load, or the persona
    seed. Supplying both a store and a checkpoint path raises ValueError.
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
        stage_model = LazyStageModel(resolve_connection(model_config), model_owner, capture_options=model_config.renderer_capture)
        if workflow_models is None:
            workflow_models = WorkflowModels(model_config, model_owner)

    if ghost is None:
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        if checkpoint is not None:
            if checkpoint.with_suffix(".pickle").exists():
                checkpoint = checkpoint.with_suffix(".pickle")
            elif not checkpoint.exists() and checkpoint.with_suffix(".json").exists():
                checkpoint = checkpoint.with_suffix(".json")
        if checkpoint is not None and checkpoint.exists():
            ghost = MemoryStore.from_checkpoint(checkpoint)
        else:
            ghost = MemoryStore.load(Path(seed_path), checkpoint=checkpoint)
    # Image tags in prompts and embeddings resolve through this store. Alt
    # text comes from the same stage model as cognition. Without a model
    # (dummy cognition), no lookup is installed and tags stay unresolved.
    from pm.memory.media_alt import MediaLibrary
    from pm.records.media import install_media_lookup
    install_media_lookup(MediaLibrary(ghost, stage_model) if stage_model is not None else None)
    target = MemoryBashTarget(ghost, mode="read")
    bus = BusSystem()
    bus.observer = bus_observer
    app = Application(bus, [], clock=clock)
    tasks = TaskTable(app.clock.monotonic)

    on_error = app.on_error
    runtime_root = Path(runtime_root)
    app.runtime_root = runtime_root

    consolidator = Consolidator(ghost, clock=app.clock)
    context_embeddings = None
    minecraft_endpoint = None
    if cognition == "real":
        from pm.llm.config import load_config
        from pm.memory.embeddings import EmbeddingService
        loaded = load_config(resolved_config)
        context_embeddings = EmbeddingService(loaded.embedding)
        ghost.initialize_embeddings(context_embeddings)
        if tool_call is None:
            tool_call = loaded.tool_call
        if cognitive_control_config is None:
            cognitive_control_config = loaded.cognitive_control
        minecraft_endpoint = _minecraft_endpoint(loaded) if enable_minecraft else None
        if loaded.minecraft is not None and minecraft_endpoint is not None:
            # The operator named a body in the model configuration, so the
            # endpoint must answer now. Failing here beats a session where
            # every Minecraft task ends unverified.
            from workflows.minecraft_body.mcp_client import probe_body
            probe_body(minecraft_endpoint, timeout_seconds=loaded.minecraft.probe_timeout_seconds)
    supervisor = ManagedRunSupervisor(
        ghost,
        runtime_root=runtime_root,
        workflows=_managed_workflows(stage_model, context_embeddings, ghost, tool_call, minecraft_endpoint,
                                     enable_minecraft=enable_minecraft),
        clock=app.clock.monotonic,
        task_root=runtime_root / "managed_tasks",
        on_error=on_error,
        workflow_models=workflow_models,
        stage_model=stage_model,
    )

    def _on_terminal(receipt: dict[str, Any]) -> None:
        """Consolidate the artifacts of a run whose receipt reached a terminal state."""
        consolidator.consolidate(receipt["run_id"], receipt["artifact_path"])

    def _event_sink(event: dict[str, Any]) -> None:
        """Enrich one supervisor event and put it on the bus as telemetry.

        The supervisor owns the durable event record; this sink only adds the
        artifact path and the wall-clock time, then forwards a copy. A run
        without a readable receipt still emits its event, because losing
        telemetry must not change how the run finished.
        """
        if event.get("event") == "gameplay_progress":
            run = supervisor.status(event["run_id"])
            consolidator.consolidate(event["run_id"], run["artifact_path"])
        try:
            run = supervisor.status(event["run_id"])
            parent = run.get("parent_correlation", "")
            event = {**event, "artifact_path": run.get("artifact_path", ""),
                     "executed_at": vclock.utcnow().isoformat()}
        except Exception:
            parent = ""
        topic = "workflow.question" if event.get("event") == "waiting_for_input" else "workflow.event"
        bus.put(Envelope(
            id=new_id(), source="managed_run", kind="telemetry", topic=topic,
            correlation_id=parent, payload=dict(event),
        ))

    supervisor.on_terminal = _on_terminal
    supervisor.event_sink = _event_sink
    steward = Steward(supervisor, clock=app.clock.monotonic)

    def workflow_paths(name: str) -> DummyWorkflowPaths:
        """Return the constructor keywords of the dummy prototype workflow ``name``."""
        return {
            "working_target": target,
            "workflow": PROJECT_ROOT / "workflows" / "demo" / f"{name}.workflow.md",
            "tools_factory": None,
            "runtime_root": runtime_root,
            "ghost": ghost,
        }

    user_input = UserInputProxy("user_input", app, ghost, emit_reply)
    main_cog, dialogue, assistant, self_management, inner_thought, focus_management = _cognition_modules(
        cognition, ghost=ghost, target=target, application=app, consolidator=consolidator,
        runtime_root=runtime_root, workflow_paths=workflow_paths,
        stage_model=stage_model, config_path=resolved_config, supervisor=supervisor,
        codelet_definitions=codelet_definitions, thought_period=thought_period,
        embeddings=context_embeddings, workflow_models=workflow_models,
        minecraft_endpoint=minecraft_endpoint,
    )
    scheduler = ScheduleSubprocesses(
        "scheduler", app, task_period,
        **workflow_paths("schedule"), tasks=tasks,
    )
    status = CheckSubprocess(
        "status", app, None,
        **workflow_paths("status"), tasks=tasks,
    )
    tool_dialogue = ToolPlusDialogue(
        "tool_dialogue", app, None,
        **workflow_paths("tool_dialogue"),
    )
    steward_module = None
    if cognition == "real":
        from workflows.steward.host import build_steward_module_tools

        steward_module = StewardModule(
            "steward", app, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "steward" / "steward.workflow.md",
            tools_factory=build_steward_module_tools(steward),
            runtime_root=runtime_root, ghost=ghost, stage_model=stage_model,
        )

    modules = [
        user_input, main_cog, dialogue, inner_thought,
        scheduler, status, tool_dialogue,
    ]
    if cognition == "real":
        from workflows.cognitive_control.host import CognitiveControl

        control = CognitiveControl(
            "cognitive_control", app, 1.0, ghost=ghost,
            config=ControlConfig.model_validate(cognitive_control_config or {}),
        )
        from workflows.entity_context.owner import EntityContextOwner

        entity_context_owner = EntityContextOwner(ghost)
        control.entity_context_owner = entity_context_owner
        control.inference_admission = main_cog.inference_admission
        control.cancel_private = inner_thought.yield_run
        def cancel_critical_private() -> None:
            """Prioritize reconsideration over private work that is calculating now.

            Deliberate mental work runs in the same private loop. Its episode
            is closed as displaced at the next private opportunity.
            """
            inner_thought.cancel_run()
        control.cancel_critical_private = cancel_critical_private
        control.enable_input_triage()
        user_input.register_triage_input = control.register_input
        dialogue.cognitive_control = control.publication_control()
        if assistant is not None:
            assistant.cognitive_control = control.publication_control()
        def cancel_public_drafts(epoch: int) -> None:
            """Cancel obsolete speech calculation without retracting accepted speech."""
            dialogue.defer_for_critical(epoch)
            if assistant is not None:
                assistant.defer_for_critical(epoch)
        control.cancel_public_drafts = cancel_public_drafts
        from workflows.input_triage.host import InputTriage
        from workflows.main_cognition.host import MainCognitionOptions

        if stage_model is None:
            raise RuntimeError("Early input triage requires the real configured stage model.")
        input_triage = InputTriage(
            "input_triage", app, None,
            working_target=MemoryBashTarget(ghost, mode="read", recovery_only=True),
            workflow=PROJECT_ROOT / "workflows" / "input_triage" / "input_triage.workflow.md",
            tools_factory=None,
            runtime_root=runtime_root, ghost=ghost, stage_model=stage_model,
        )
        input_triage.cognitive_control = control
        input_triage.entity_context_owner = entity_context_owner
        modules.insert(1, input_triage)
        app.input_triage = input_triage
        from workflows.interaction.host import Interaction
        from workflows.interaction.records import interaction_receipts

        interaction = Interaction(
            "interaction", app, None,
            working_target=MemoryBashTarget(ghost, mode="read", recovery_only=True),
            workflow=PROJECT_ROOT / "workflows" / "interaction" / "interaction.workflow.md",
            tools_factory=None,
            runtime_root=runtime_root, ghost=ghost, stage_model=stage_model,
        )
        interaction.embedding_service = context_embeddings
        interaction.cognitive_control = control
        control.validate_selection = interaction.validate_selection
        control.refresh_selection = interaction.refresh_selection
        from workflows.main_cognition.schemas import ActionRequest

        def refresh_public_action(request: ActionRequest) -> None:
            """Reconsider selected direction without repeating accepted external work."""
            if request.interaction_selection_id:
                interaction.refresh_selection(request.interaction_selection_id)
            elif request.knoxel_id is not None:
                main_cog.defer_public_action(request)
        control.refresh_action = refresh_public_action
        interaction.main_cognition = main_cog
        main_cog.interaction_enabled = True
        modules.insert(2, interaction)
        main_cog.cognitive_control = control
        main_cog.entity_context_owner = entity_context_owner
        control.prioritize_critical = main_cog.prioritize_critical
        inner_thought.cognitive_control = control
        def release_private_results(envelopes: list[Envelope]) -> None:
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
        if world_poll_seconds is not None and minecraft_endpoint is not None:
            from workflows.minecraft_body.world_events import WorldEvents
            from workflows.minecraft_body.mcp_client import MinecraftTransport, MinecraftTransportSettings
            from workflows.minecraft_body.lease import BODY_LEASES
            endpoint = minecraft_endpoint
            transport = MinecraftTransport(MinecraftTransportSettings(endpoint=endpoint, request_timeout_seconds=30), None)

            def observe_world() -> dict[str, Any]:
                # Reads share the command lock with actor calls. They never
                # claim the mutation lease or bypass an in-flight action.
                from pm_workflows.runtime import WorkflowTerminated
                try:
                    with BODY_LEASES.command(endpoint, cancel=world_events.stop_event):
                        return transport.call("minecraft_observe", {"include_image": False})
                except WorkflowTerminated:
                    return {"ok": False, "cancelled": True}

            world_events = WorldEvents(
                "world_events", app, world_poll_seconds,
                ghost=ghost, observe=observe_world,
                journal=runtime_root / "world-events.jsonl", quiet_seconds=world_quiet_seconds,
            )
            modules.append(world_events)
            app.world_events = world_events
        from workflows.minecraft_body.capabilities import CapabilitySnapshotOwner
        from workflows.public_speech.reply_recovery import ReplyRecovery, ReplyRecoverySelection
        capability_owner = CapabilitySnapshotOwner(ghost, supervisor, app)
        main_cog.capability_owner = capability_owner
        if "tool_call" in supervisor.workflows:
            # The tool list is discovered at route time and cached, so a
            # server that starts later becomes visible without a restart.
            from workflows.tool_call.catalogue import ToolCatalogue
            catalogue = ToolCatalogue(tool_call)
            main_cog.workflow_catalogue_owner = WorkflowCatalogue(supervisor, catalogue)
        from workflows.main_cognition.retrieval_query import QueryRewriter
        main_cog.retrieval_query_provider = QueryRewriter(stage_model)
        dialogue.capability_owner = capability_owner
        assistant.capability_owner = capability_owner
        # Context planning for public replies shares one embedding service and
        # its disk cache. Construction makes no network request.
        dialogue.context_embeddings = context_embeddings
        assistant.context_embeddings = context_embeddings
        # Compute the input token budget from the model's serving context.
        if stage_model is not None:
            import asyncio as _aio
            from threading import Event as _Evt
            from pm.llm.workflow_sessions import serving_context as _serving
            _ctx, _ = _aio.run(_serving(stage_model.connection, app.shutdown_event))
            _budget = min(int(_ctx * stage_model.connection.context_input_fraction),
                          _ctx - (stage_model.connection.completion_token_limit or 0))
            dialogue.context_budget = _budget
            assistant.context_budget = _budget
        recovery = ReplyRecovery("reply_recovery", app, None, ghost=ghost, supervisor=supervisor)
        recovery.selection = ReplyRecoverySelection(ghost, control, main_cog)
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

    def finish_workers() -> None:
        """Join all owned work before recording a successful clean stop."""
        supervisor.shutdown(preserve_parked=app.drained.is_set())
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
        from workflows.introspection.inspection_owner import CapabilityInspectionOwner
        inspection_owner = CapabilityInspectionOwner(ghost,
            frozenset(module.module_id for module in modules), Path(resolved_config), app)
        inspection_owner.refresh()
        if inner_thought is not None:
            inner_thought.inspection_owner = inspection_owner

    # Durable lifecycle evidence. The boot event feeds the first main cycle so
    # the companion can appraise a real offline gap. Only real cognition runs
    # that cycle, so the dummy fixture records the event without a stimulus.
    from pm.app.lifecycle_records import record_startup
    boot = record_startup(ghost) if record_lifecycle else None
    app.lifecycle_boot = boot
    # Startup opens accepted memory. Executable inboxes, parked workers, and
    # undelivered replies from a previous process are deliberately not replayed.
    if cognition == "real" and boot is not None:
        bus.put(Envelope(
            id=new_id(), source="system", kind="data", topic="system.lifecycle",
            correlation_id="", payload={"knoxel_id": boot["knoxel_id"]},
        ))
    return app

