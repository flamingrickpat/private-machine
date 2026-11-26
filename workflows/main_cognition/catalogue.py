from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from workflows.input_triage.schemas import BodyRunTarget
from pm.runs.supervisor import ManagedRunSupervisor
from pm.runs.run_receipt import ACTIVE_STATES
from workflows.main_cognition.schemas import ManagedRunView, WorkflowOption
from workflows.tool_call.catalogue import ToolCatalogue


@dataclass(frozen=True, slots=True)
class WorkflowCatalogue:
    supervisor: ManagedRunSupervisor
    tool_catalogue: ToolCatalogue | None

    def body_target(self) -> BodyRunTarget | None:
        from workflows.input_triage.schemas import BodyRunTarget
        spec = self.supervisor.workflows.get("minecraft_project")
        if spec is None or spec.reservation is None:
            return None
        owner = spec.reservation.owner()
        rows = self.supervisor.ghost.read_run_receipts()
        if owner and owner in rows:
            row = rows[owner]
            return BodyRunTarget(run_id=owner, execution_epoch=row["execution_epoch"], state=row["state"])
        for row in reversed(list(rows.values())):
            if row["workflow_name"] in {"minecraft_project", "minecraft"}:
                if row["state"] == "cancelled":
                    return BodyRunTarget.model_validate(row)
                break
        return None

    def occupied_resources(self) -> dict[str, str]:  # pm-lint: allow PM004 -- run descriptions keyed by shared workflow resource
        """Read accepted leases without taking the admission mutex under the memory lock."""
        occupied: dict[str, str] = {}  # pm-lint: allow PM004 -- active owners indexed by singleton key
        for run_id, row in self.supervisor.ghost.read_run_receipts().items():
            run = ManagedRunView.model_validate(row)
            if run.state not in ACTIVE_STATES or run.workflow_name is None:
                continue
            spec = self.supervisor.workflows.get(run.workflow_name)
            if spec is not None and spec.singleton:
                resource = run.singleton_key or spec.singleton_key or run.workflow_name
                occupied[resource] = f"{run.workflow_name} run {run_id} ({run.state})"
        spec = self.supervisor.workflows.get("minecraft_project")
        if spec is not None and spec.reservation is not None:
            owner = spec.reservation.owner()
            if owner:
                occupied.setdefault(spec.singleton_key, owner)
        return occupied  # pm-lint: allow PM005 -- owners indexed by shared workflow resource

    def direct_request_options(self) -> list[WorkflowOption]:
        """List registered workflows that accept the original partner request as input."""
        occupied = self.occupied_resources()
        return [WorkflowOption(name=name, description=spec.direct_request, routine_start=True)
                for name, spec in sorted(self.supervisor.workflows.items())
                if spec.routine_start and spec.direct_request
                and (not spec.singleton or (spec.singleton_key or name) not in occupied)]

    def options(self) -> list[WorkflowOption]:
        occupied = self.occupied_resources()
        management = "Start, status, update, terminate, or explicitly resume an interrupted run."
        if occupied:
            management += " Active resource owners: " + ", ".join(f"{key}: {owner}" for key, owner in occupied.items())
            management += ". Use the owner's run ID for status or a change of task."
        options: list[WorkflowOption] = [
            WorkflowOption(name='dialogue_story', description='Generate one in-character Story reply.'),
            WorkflowOption(name='self_management', description=management),
            WorkflowOption(name='focus_management', description='Private deliberate mental work: scoped recall of preferences, conversations, or goals, and a temporary focus or goal proposal. Set arguments.instruction to the mental task. Advice returns through ordinary appraisal. It commits no emotion or memory change directly.'),
            WorkflowOption(name='theory_of_mind', description='Private thinking about another person or agent: what they usually do, what they know or believe, what they may want, and how uncertain that is. Set arguments.instruction to one bounded question. It sends no message and asks nobody. Its result returns through ordinary appraisal.'),
        ]
        descriptions: dict[str, str] = {
            'local_task': 'Control worker. Writes task telemetry; cannot execute arbitrary instructions or create a requested note.',
            'note_task': 'Exact UTF-8 note writer with independent byte verification. Optional arguments.note_spec has path and text; path stays in outbox/.',
            'memory_reconsolidation': 'Sort through past experiences while awake (background), or sleep while sorting them (sleep). Choose reconsolidation_mode on the memory ToolCall. During sleep an incoming message or the end of the memory work wakes the character.',
            'minecraft': 'One sustained survival specialist leaf. Standalone completion requires an explicit supported verifier, currently wood_return. Use minecraft_project for other goals that need verified completion.',
            'minecraft_project': 'Generic Minecraft project planner. Durable subtasks, verified prerequisites, fresh specialists, and independent world verification. Use for multi-step goals.',
        }
        for name in sorted(self.supervisor.workflows):
            spec = self.supervisor.workflows[name]
            if spec.singleton and (spec.singleton_key or name) in occupied:
                continue
            description: str = descriptions.get(name, f'Managed workflow {name}.')
            if name in {'assistant', 'tool_call'}:
                from workflows.tool_call.host import WORKFLOW_DESCRIPTION
                description = ("Private deliberate tool use as yourself. Look, inspect, or act through the listed MCPs. "
                               "ToolCall returns your conclusion as a new private stimulus."
                               if name == "assistant" else WORKFLOW_DESCRIPTION)
                if self.tool_catalogue is None:
                    description += " (listed at route time)"
                else:
                    suites = self.tool_catalogue.config.mcp_servers
                    for server in suites:
                        owner = occupied.get("minecraft_body") if server.name in spec.reserved_suites else None
                        state = f"busy with {owner} (status, wait or terminate)" if owner else "available for this run"
                        description += f"\n{server.name}: {state}."
                    description += "\n" + self.tool_catalogue.describe(compact=True)
            if spec.singleton:
                description += f' One active run for {spec.singleton_key or name}. A busy start returns its owner; choose wait, update, or terminate through self_management.'
            resource = (spec.singleton_key or name) if spec.singleton else ""
            options.append(WorkflowOption(name=name, description=description,
                                          routine_start=spec.routine_start, exclusive_resource=resource))
        return options
