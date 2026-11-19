from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, JsonValue
from pm.context_render.prompt import RenderSettings, RenderedPrompt
from workflows.minecraft.execution_contracts import LeafSpecification, ActorAcceptance, UncertainWorldEffect
from workflows.minecraft_project.planning.context_contracts import RecoveryFacts, GameplayHistory
from workflows.minecraft_project.planning.context_text import (condition_text, identity_text, intent_text, node_text,
    read_text, recovery_text, gameplay_sections, caller_requirements_text)
from workflows.minecraft_project.planning.rendering import TechnicalSection, ProjectDecisionRenderInput, render_project_decision
from workflows.minecraft_body.world_text import observation_text, foreign_operands_text, coordinates


class ReconciliationEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    uncertain_world_effects: tuple[UncertainWorldEffect, ...]
    target_reads: tuple[JsonValue, ...]
    instruction_revision: int
    observation_seq: int


@dataclass(frozen=True, slots=True)
class SpecialistRenderInput:
    skill: str
    leaf: LeafSpecification
    acceptance: ActorAcceptance
    current_observation: JsonValue
    server_rules: str
    recovery: RecoveryFacts | None
    reconciliation: ReconciliationEvidence


@dataclass(frozen=True, slots=True)
class LeafRenderInput:
    leaf: LeafSpecification


def leaf_sections(leaf: LeafSpecification) -> list[TechnicalSection]:
    sections = [TechnicalSection("Current leaf instruction", leaf.instruction, True)]
    if leaf.identity is not None:
        sections.append(TechnicalSection("Current attempt", identity_text(leaf.identity), True))
    if leaf.project_intent is not None:
        sections.append(TechnicalSection("Owner project intent", intent_text(leaf.project_intent), True))
    sections.extend(TechnicalSection("Exact current criterion", condition_text(item), True) for item in leaf.criteria)
    observations = leaf.current_criterion_observations
    if observations is not None:
        text = f"{observations.scope}\nOriginal {observations.original_receipt}; digest={observations.original_digest}\n"
        text += "\n".join(f"{name}: {passed}" for name, passed in observations.checks.items())
        text += "\n" + "\n\n".join(read_text(read) for read in observations.reads)
        sections.append(TechnicalSection("Current criterion observations", text, True))
    project = leaf.project
    if project is not None:
        sections.append(TechnicalSection("Parent project goal", project.goal, True))
        sections.extend(TechnicalSection("Ancestor integration requirement", node_text(node), True)
            for node in project.ancestor_specifications)
        if project.caller_constraints is not None:
            sections.append(TechnicalSection("Caller requirements", caller_requirements_text(project.caller_constraints), True))
        if project.latest_project_check is not None:
            check = project.latest_project_check
            sections.append(TechnicalSection("Latest caller audit", f"{check.original_receipt}: " + "; ".join(check.errors)
                + "\n" + foreign_operands_text(check.checks) + "\n" + foreign_operands_text(check.requirements), True))
    return sections


def render_leaf_specification(input: LeafRenderInput, settings: RenderSettings) -> RenderedPrompt:
    return render_project_decision(ProjectDecisionRenderInput("Minecraft current leaf",
        "These are this attempt's original task operands. Historical notes do not change them.",
        tuple(leaf_sections(input.leaf))), settings)


def reconciliation_sections(reconciliation: ReconciliationEvidence) -> list[TechnicalSection]:
    sections: list[TechnicalSection] = []
    for effect in reconciliation.uncertain_world_effects:
        text = f"{effect.tool}; source={effect.source}; sequence={effect.seq}; time={effect.time}\n"
        text += foreign_operands_text(effect.arguments)
        for furnace in effect.candidate_furnaces:
            text += f"\nCandidate furnace {coordinates(furnace.position)}; source={furnace.source}; sequence={furnace.seq}"
        sections.append(TechnicalSection("Unresolved world effect: reconcile before repeating", text, True))
    for read in reconciliation.target_reads:
        sections.append(TechnicalSection("Fresh reconciliation target read", foreign_operands_text(read), True))
    return sections


def render_reconciliation(input: ReconciliationEvidence, settings: RenderSettings) -> RenderedPrompt:
    return render_project_decision(ProjectDecisionRenderInput("Minecraft reconciliation",
        "Original uncertain mutations and fresh target reads. Never replay an uncertain mutation without reconciliation.",
        tuple(reconciliation_sections(input))), settings)


def render_specialist(input: SpecialistRenderInput, settings: RenderSettings) -> RenderedPrompt:
    sections = leaf_sections(input.leaf)
    sections.append(TechnicalSection("Current body observation", observation_text(input.current_observation), True))
    sections.append(TechnicalSection("Current server rules", input.server_rules, True))
    if input.recovery is not None:
        sections.append(TechnicalSection("Active recovery", recovery_text(input.recovery), True))
    sections.extend(reconciliation_sections(input.reconciliation))
    for node_id, prerequisite in input.acceptance.prerequisites.items():
        text = f"Accepted dependency {node_id}; original proof={prerequisite.verification}; certificate={prerequisite.certificate_digest}\n"
        text += node_text(prerequisite.specification) + "\n" + identity_text(prerequisite.identity)
        text += "\n" + "\n".join(f"{criterion}: {passed}" for criterion, passed in prerequisite.checks.items())
        sections.append(TechnicalSection("Historical accepted dependency", text, True))
    history = input.acceptance.gameplay_history
    if history is not None:
        sections.extend(gameplay_sections(history, False))
    previous = input.acceptance.previous_attempt
    if previous is not None:
        sections.append(TechnicalSection("Previous actor handoff", f"{previous.scope}\nSource={previous.identity.path}\n{previous.handoff}", False))
    return render_project_decision(ProjectDecisionRenderInput("Minecraft specialist", input.skill, tuple(sections)), settings)


def render_gameplay_archive(input: GameplayHistory, settings: RenderSettings) -> RenderedPrompt:
    return render_project_decision(ProjectDecisionRenderInput("Minecraft selected gameplay archive",
        "Earlier observed methods, search coverage, and current recovery. Original journals retain complete receipts.",
        tuple(gameplay_sections(input, False))), settings)
