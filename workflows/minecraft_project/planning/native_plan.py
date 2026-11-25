"""Provide a task group's draft tools and validator to a native session."""
from __future__ import annotations
from workflows.minecraft.runtime import require_minecraft_runtime
from pm.runs.managed_boundary import BoundaryReadSettings

from contextlib import nullcontext
from dataclasses import replace, dataclass
from pydantic import BaseModel, JsonValue, TypeAdapter
from pm.llm.session import ModelSession
from pm.context_render.prompt import RenderSettings
from workflows.minecraft_project.planning.context_contracts import PlanningContext, ArchiveReference, SourceBinding, SourceRead, DecisionIdentity
import json

from pm.runs.draft_session import run_draft_session
from pm.runs.role_runtime import PMWorkflowRuntime
from pm.memory.agent_tools import MemoryReadSession, MemoryTraceFile
from workflows.minecraft_body.gateway import READ_TOOLS
from workflows.minecraft_project.evidence_selection import catalog, read_source
from pm_workflows.python_role import RoleContext
from workflows.minecraft_project.schemas import EvidenceRead, PlanProposal
from workflows.minecraft_project.planning.draft import TaskDraft, DraftSettings, PlanningBoundary
from workflows.minecraft_project.planning.run_context import identity
from workflows.minecraft_project.planning.context_selection import native_decision_context
from workflows.minecraft_project.tree_store import TreeStore
from pm.llm.role_model import role_model
from pm.context_render.prompt import RendererCapture, RendererInput
from workflows.minecraft_body.planning_prompt import MinecraftActionTool
from workflows.minecraft_body.world_text import world_result_text, foreign_operands_text
from workflows.minecraft_project.planning.context_sources import decision_identity
from workflows.minecraft_project.planning.rendering import (ProjectDecisionRenderInput, render_project_decision, NativePlannerRenderInput, render_native_planner)



class WorldReadRequest(BaseModel):
    tool: str
    arguments: dict[str, JsonValue]


class NativePlanningInput(BaseModel):
    identity: DecisionIdentity
    context: PlanningContext
    mode: str = "native_mcp_draft"


@dataclass(slots=True)
class NativePlanningTools:
    boundary: PlanningBoundary
    model: ModelSession
    render_settings: RenderSettings
    context_input: ProjectDecisionRenderInput
    sources: dict[str, SourceBinding]

    def plan_context(self) -> str:
        with RendererCapture(self.model.capture_settings("minecraft.native_context"),
                RendererInput("minecraft.native_context", self.context_input, self.render_settings)) as capture:
            rendered = render_project_decision(self.context_input, self.render_settings)
            capture.save_output(rendered)
        return rendered.turns[1].content.text

    def world_read(self, request: WorldReadRequest) -> str:
        self.boundary.check_current()
        if request.tool not in READ_TOOLS:
            raise ValueError("Planning requires a read-only tool. Read world_contracts for its exact schema.")
        runtime = self.boundary.runtime
        body_runtime = require_minecraft_runtime(runtime)
        transport = body_runtime.minecraft_transport
        with transport.leases.command(transport.settings.endpoint, body_runtime.stop_event):
            reply = require_minecraft_runtime(runtime).minecraft_transport.call(request.tool, request.arguments)
        self.boundary.check_current()
        wire = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(reply)
        if not wire["ok"]:
            return "Read failed: " + foreign_operands_text(wire)
        return world_result_text(request.tool, wire["result"])

    def world_contracts(self) -> tuple[MinecraftActionTool, ...]:
        return tuple(row for row in require_minecraft_runtime(self.boundary.runtime).minecraft_transport.tool_schemas() if row.name in READ_TOOLS)

    def evidence_list(self) -> tuple[str, ...]:
        return tuple(sorted(self.sources))

    def evidence_read(self, request: EvidenceRead) -> SourceRead:
        self.boundary.check_current()
        if request.source.replace("\\", "/") not in self.sources:
            raise ValueError("Unknown source. Read evidence_list for its exact name.")
        return read_source(self.sources, request)

def plan_session(context: RoleContext, store: TreeStore, payload: PlanningContext,
                 *, scope: set[str] | None = None) -> PlanProposal:
    """Edit a retained draft through MCP and validate each completed turn.

    This group owns plan meaning, evidence access, and its validator. The shared
    runtime owns the native session and temporary listeners. Only later plan
    commit, independent review, and world verification can release an actor.
    """
    runtime = context.runtime
    if runtime is None:
        raise RuntimeError("Minecraft planning requires its workflow runtime.")
    name = f"decisions/{len(store.state['decisions']) + 1:04d}-planner"
    store.write(name + "/input.json", NativePlanningInput(identity=decision_identity(context, store), context=payload))
    store.state["decisions"].append({"role": "planner", "path": name, "identity": identity(context, store)})
    store.save()
    instruction_revision = require_minecraft_runtime(runtime).instructions()["revision"]

    boundary = PlanningBoundary(runtime, instruction_revision)
    draft = TaskDraft(store, payload.goal, name + "/draft.json", boundary,
        DraftSettings(base=store.load() if store.state["revision"] else None,
            scope=frozenset(scope) if scope is not None else None,
            recovery_required=payload.gameplay_history.recovery is not None,
            refinement_target=payload.refinement_target,
            require_leaf_change=bool(store.state.get("require_leaf_change", False))))
    server = draft.server()
    archive = f"planning/context/{len(store.state['decisions']):04d}-full-input.json"
    store.write(archive, payload)
    sources = catalog(context, store)
    selected_context = native_decision_context(payload)
    selected_context.context_archive = ArchiveReference(original_receipt=archive, catalog_source=archive)

    session_model = role_model(runtime)
    render_settings = session_model.render_settings()
    context_input = ProjectDecisionRenderInput("Minecraft native context",
        "Read these original task facts and technical operands.", (), context=selected_context, native=True)
    tools = NativePlanningTools(boundary, session_model, render_settings, context_input, sources)
    server.tool()(tools.plan_context)
    server.tool()(tools.world_read)
    server.tool()(tools.world_contracts)
    server.tool()(tools.evidence_list)
    server.tool()(tools.evidence_read)

    prompt = (
        "You are the companion's private Minecraft planner. Use the project_plan MCP tools to edit the task draft. "
        "Read plan_context and task_list first. You do not need to return a JSON tree or reproduce unchanged nodes. "

        "Use ordinary condition objects for edits. Full context remains in context_archive through evidence_read. "
        "The backend assigns new IDs, parent links, and revision counters. task_edit changes only named fields. "
        "condition_set accepts an inclusive box for exact rectangular block cells. condition_copy preserves exact operands. "
        "Different materials require separate exact conditions. Observe an uncertain marker before choosing its role. "
        "A review is a proposed diagnosis. Check it against the request and evidence before changing a requirement. "
        "Choose concrete goal, story, and leaf outcomes with independently observable criteria. "
        "Read project_intent: preserve original constraints except for changes the current instruction explicitly requests. "
        "Keep a coherent task in one leaf when a capable actor can gather, craft, build, and verify it. "
        "Do not add formal preparation leaves or detailed arithmetic merely for ceremony. "
        "Use already-owned materials and adequate existing components. A survey can report an unsuitable site without altering it. "
        "Read current world facts and actual tool contracts when needed; do not invent a surveyed site or recipe. "
        "Keep completed work exact. An accepted reference records earlier attainment, while fresh block criteria establish final construction. "
        "Account for materials, safe terrain, doors, clear interior, entry, exit, and the user's current dimensions when requested. "
        "Inventory targets use the original baseline for an unchanged request. User changes use their new wording and current state. "
        "Preserve a return route and change failed methods; prioritize real recovery over optional work. "
        "For refinement_target, add descendants under that same target and retain its original criteria. "
        "Use plan_outcome for a survey, external question, blocker, or explicit proof restoration. "
        "Use plan_check for convenient feedback. When your draft is ready, end the turn with a short explanation. "
        "The backend then checks the actual draft and continues this same conversation if repair is needed. "
        "A valid draft is a plan; it does not establish a completed game task. "
        "Optional memory tools expose original preferences, prior outcomes, and graph sources. "
        "This decision group has a fresh memory snapshot. Reinspect earlier evidence that affects this draft. "
        "Use them when past context can change your decision. Ordinary choices need no search. "
        "Interpret dates, sources, and corrections yourself. A current explicit request can change an older preference. "
        "If no preference is available, choose a practical option within the current request. "
        "Match planning effort to this task's consequences and requested review mode. "
        "For a small reversible task, prefer one useful executable leaf over speculative future phases. "
        "A request to do something and tell the partner the result has two owners. Your leaf performs the "
        "world action and records the actual outcome. After verification the public reply workflow reads the "
        "server returns and speaks. A harvest with random drops needs a before/after inventory read in the "
        "actor instructions and a world criterion such as the original cell becoming air. Zero drops is a "
        "valid observed outcome. Keep the original block in the baseline. Remove tautological count tests and "
        "speech predicates if an earlier proposal invented them.")
    initial_input = NativePlannerRenderInput(prompt, payload.goal)
    with RendererCapture(session_model.capture_settings("minecraft.native_planner"),
            RendererInput("minecraft.native_planner", initial_input, render_settings)) as capture:
        initial_prompt = render_native_planner(initial_input, render_settings)
        capture.save_output(initial_prompt)
    prompt = initial_prompt.turns[0].content.text
    session = (nullcontext(runtime.agent_model) if runtime.agent_model is not None else
               runtime.classification_session("Default", False, runtime.stop_event))
    memory = None
    if isinstance(runtime, PMWorkflowRuntime) and runtime.memory_context is not None:
        memory = MemoryReadSession(replace(runtime.memory_context.create(),
            trace=MemoryTraceFile(store.path_for(name + "/memory.jsonl"))))
    # Each group receives a fresh grant. Native child processes receive only
    # its read listener. Cleanup revokes access before another group starts.
    with (memory if memory is not None else nullcontext()), session as model:
        result = run_draft_session(replace(runtime, agent_model=replace(model, enable_thinking=True)),
            run_id=context.task_id + "-" + name.replace("/", "-"), prompt=prompt, servers={"project_plan": server},
            validator=draft, artifact_dir=store.path_for(name + "/native"), memory=memory)
        revisions = dict(memory.accessed) if memory is not None else {}
    boundary.check_current()
    store.write(name + "/output.json", result.value.model_dump(mode="json"))
    store.write(name + "/session.json", {"session_ref": result.session_ref,
        "validation_attempts": result.validation_attempts, "mode": "native_mcp_draft"})
    store.state["planning_memory_revisions"] = revisions
    store.write(name + "/memory-revisions.json", {"snapshot_id": memory.snapshot_id if memory is not None else None,
                                               "revisions": revisions})
    return result.value
