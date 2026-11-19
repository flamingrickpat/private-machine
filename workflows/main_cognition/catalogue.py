from __future__ import annotations

from dataclasses import dataclass
from pm.runs.supervisor import ManagedRunSupervisor
from workflows.main_cognition.schemas import WorkflowOption
from workflows.tool_call.catalogue import ToolCatalogue


@dataclass(frozen=True, slots=True)
class WorkflowCatalogue:
    supervisor: ManagedRunSupervisor
    tool_catalogue: ToolCatalogue | None

    def options(self) -> list[WorkflowOption]:
        options: list[WorkflowOption] = [
            WorkflowOption(name='dialogue_story', description='Generate one in-character Story reply.'),
            WorkflowOption(name='dialogue_assistant', description='Assistant reply with scoped tools. Inventory questions use arguments.minecraft_read=inventory; position, health, and surroundings use minecraft_read=state. Both make one live read without a managed task.'),
            WorkflowOption(name='self_management', description='Start, status, update, terminate, or explicitly resume an interrupted body run.'),
            WorkflowOption(name='focus_management', description='Private deliberate mental work: scoped recall of preferences, conversations, or goals, and a temporary focus or goal proposal. Set arguments.instruction to the mental task. Advice returns through ordinary appraisal. It commits no emotion or memory change directly.'),
            WorkflowOption(name='theory_of_mind', description='Private thinking about another person or agent: what they usually do, what they know or believe, what they may want, and how uncertain that is. Set arguments.instruction to one bounded question. It sends no message and asks nobody. Its result returns through ordinary appraisal.'),
        ]
        descriptions: dict[str, str] = {
            'local_task': 'Control worker. Writes task telemetry; cannot execute arbitrary instructions or create a requested note.',
            'note_task': 'Exact UTF-8 note writer with independent byte verification. Optional arguments.note_spec has path and text; path stays in outbox/.',
            'memory_reconsolidation': 'Sort through memories and sleep on them: builds topic and day summaries, facts, and updated self and relationship narratives from the stored history. Slow; runs in the background. Use ToolCall when the character decides to sort through memories, sleep on what happened, or reflect on the past days.',
            'minecraft': 'One sustained survival specialist leaf. Standalone completion requires an explicit supported verifier, currently wood_return. Use minecraft_project for other goals that need verified completion.',
            'minecraft_project': 'Generic Minecraft project planner. Durable subtasks, verified prerequisites, fresh specialists, and independent world verification. Use for multi-step goals.',
        }
        for name in sorted(self.supervisor.workflows):
            description: str = descriptions.get(name, f'Managed workflow {name}.')
            if name == 'tool_call':
                from workflows.tool_call.host import WORKFLOW_DESCRIPTION
                description = WORKFLOW_DESCRIPTION
                if self.tool_catalogue is None:
                    description += " (listed at route time)"
                else:
                    description += "\n" + self.tool_catalogue.describe()
            spec = self.supervisor.workflows[name]
            if spec.singleton:
                description += f' One active run for {spec.singleton_key or name}. A busy start returns its owner; choose wait, update, or terminate through self_management.'
            options.append(WorkflowOption(name=name, description=description))
        return options
