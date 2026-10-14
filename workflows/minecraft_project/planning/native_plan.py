"""Provide a task group's draft tools and validator to a native session."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

from pm.runs.draft_session import run_draft_session
from workflows.minecraft_body.gateway import READ_TOOLS
from workflows.minecraft_body.lease import BODY_LEASES
from workflows.minecraft_project.evidence_selection import catalog, read_source
from workflows.minecraft_project.schemas import EvidenceRead, PlanProposal
from workflows.minecraft_project.planning.draft import TaskDraft
from workflows.minecraft_project.planning.run_context import identity, InstructionsChanged


def plan_session(context, store, payload: dict, *, scope: set[str] | None = None) -> PlanProposal:
    """Edit a retained draft through MCP and validate each completed turn.

    This group owns plan meaning, evidence access, and its validator. The shared
    runtime owns the native session and temporary listeners. Only later plan
    commit, independent review, and world verification can release an actor.
    """
    runtime = context.runtime
    name = f"decisions/{len(store.state['decisions']) + 1:04d}-planner"
    store.write(name + "/input.json", {"identity": identity(context, store), "context": payload,
                                       "mode": "native_mcp_draft"})
    store.state["decisions"].append({"role": "planner", "path": name, "identity": identity(context, store)})
    store.save()
    instruction_revision = runtime.tools["managed_instructions"]()["revision"]

    def check_current():
        runtime.check_cancelled()
        if runtime.tools["managed_instructions"]()["revision"] != instruction_revision:
            raise InstructionsChanged("Project instructions changed during planning. Retain effects and read the new request.")

    draft = TaskDraft(store, payload["goal"], lambda value: store.write(name + "/draft.json", value),
        check_current, base=store.load() if store.state["revision"] else None, scope=scope,
        refinement_target=payload.get("refinement_target", ""))
    server = draft.server()
    sources = catalog(context, store)

    @server.tool()
    def plan_context() -> dict:
        """Read the current request, observations, history, and repair feedback."""
        return payload

    @server.tool()
    def world_read(tool: str, arguments: dict = {}) -> dict:
        """Obtain current read-only Minecraft facts. This planner cannot mutate the world."""
        check_current()
        if tool not in READ_TOOLS:
            return {"ok": False, "error": "Planning uses read-only tools. Use world_contracts to read their actual schemas."}
        with BODY_LEASES.command(runtime.tools["minecraft_endpoint"](), runtime.stop_event):
            reply = runtime.tools["minecraft_call"](tool, arguments)
        check_current()
        return reply

    @server.tool()
    def world_contracts() -> list[dict]:
        """Read the configured schemas for allowed world observations."""
        return [row for row in runtime.tools["minecraft_tool_schemas"]() if row["name"] in READ_TOOLS]

    @server.tool()
    def evidence_list() -> list[str]:
        """List original evidence sources that this planning group can inspect."""
        return sorted(sources)

    @server.tool()
    def evidence_read(source: str, start_line: int = 1, line_count: int = 100, query: str = "") -> dict:
        """Read numbered original source lines without reconstructing their contents."""
        check_current()
        if source.replace("\\", "/") not in sources:
            return {"ok": False, "error": "Unknown source. Use evidence_list to read its exact name."}
        return read_source(sources, EvidenceRead(source=source, start_line=start_line, line_count=line_count, query=query))

    prompt = (
        "You are the companion's private Minecraft planner. Use the project_plan MCP tools to edit the task draft. "
        "Read plan_context and task_list first. You do not need to return a JSON tree or reproduce unchanged nodes. "
        "The backend assigns new IDs, parent links, and revision counters. task_edit changes only named fields. "
        "condition_set accepts an inclusive box for exact rectangular block cells. condition_copy preserves exact operands. "
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
        "Current goal: " + payload["goal"])
    session = (nullcontext(runtime.agent_model) if runtime.agent_model is not None else
               runtime.classification_session("Default", False, runtime.stop_event))
    with session as model:
        result = run_draft_session(replace(runtime, agent_model=replace(model, enable_thinking=True)),
            run_id=context.task_id + "-planner", prompt=prompt, servers={"project_plan": server},
            validate=draft.validate, artifact_dir=store.path_for(name + "/native"))
    check_current()
    store.write(name + "/output.json", result.value.model_dump(mode="json"))
    store.write(name + "/session.json", {"session_ref": result.session_ref,
        "validation_attempts": result.validation_attempts, "mode": "native_mcp_draft"})
    return result.value
