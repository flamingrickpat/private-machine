from __future__ import annotations

from typing import Literal
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator, TypeAdapter
from workflows.minecraft_project.schemas import Node, TaskTree, Point, Condition, PlanReview, VerificationReview
from workflows.minecraft_body.planning_prompt import MinecraftActionTool
from workflows.minecraft_body.spatial import WorldPosition


class ContextContract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DecisionIdentity(ContextContract):
    task_id: str
    tree_revision: int
    leaf_id: str = ""
    spec_revision: int = 0
    actor_attempt: str = ""
    verifier_attempt: str = ""


class ProjectIntent(ContextContract):
    original_request: str
    original_goal: str
    constraints: tuple[str, ...]
    current_instruction: str


class ReviewPolicy(ContextContract):
    mode: Literal["flexible", "strict"]
    passing_judgments: tuple[str, ...]


class ArchiveReference(ContextContract):
    original_receipt: str
    original_digest: str = ""
    catalog_source: str = ""


class SourceBinding(ContextContract):
    path: str
    sha256: str


class EvidenceCatalog(ContextContract):
    index_source: str
    source_groups: dict[str, int]
    named_sources: tuple[str, ...]


class SourceRead(ContextContract):
    source: str
    sha256: str
    total_lines: int
    start_line: int
    lines: tuple[str, ...]
    query: str


class RecordedRead(ContextContract):
    criterion_id: str = ""
    observed_dimension: str | None = None
    tool: str
    arguments: dict[str, JsonValue]
    time: float
    ok: bool
    result: JsonValue = None
    id: str = ""
    seq: int | None = None
    observation_available: bool | None = None
    original_digest: str = ""
    result_scope: str = ""


class SurveyEvidence(ContextContract):
    identity: DecisionIdentity
    reads: tuple[RecordedRead, ...] = ()
    recipes: tuple[RecordedRead, ...] = ()
    tool_contracts: tuple[MinecraftActionTool, ...] = ()
    original_receipt: str = ""
    original_digest: str = ""
    instruction_revision: int | None = None
    queries: tuple[Condition, ...] = ()
    recipe_queries: tuple[str, ...] = ()
    scope: str = ""
    recipe_scope: str = ""
    read_scope: str = ""
    shared_tool_contracts: str = ""
    archived_read_count: int = 0
    archived_recipe_count: int = 0


class AcceptanceCertificate(ContextContract):
    spec_digest: str
    verification: str
    identity: DecisionIdentity
    dependencies: dict[str, str]
    files: dict[str, str]


class AcceptedInput(ContextContract):
    certificate: AcceptanceCertificate
    checks: dict[str, bool]
    identity: DecisionIdentity
    scope: str


class RetentionCandidate(ContextContract):
    specification: Node
    certificate: AcceptanceCertificate
    archive: str
    source: str
    currently_valid: bool = False


class NodeReference(ContextContract):
    from_node: str
    collection: str = Field(validation_alias="in", serialization_alias="in")


class RetainedCandidate(ContextContract):
    specification: Node | NodeReference
    archive: str
    certificate_digest: str
    required_certificates: dict[str, str]
    changed_dependencies: tuple[str, ...]
    verification: str


class NodeTopology(ContextContract):
    id: str
    parent: str | None
    children: tuple[str, ...]
    depends_on: tuple[str, ...]
    kind: Literal["goal", "story", "leaf"]


class ReviewScope(ContextContract):
    node_id: str
    kind: Literal["goal", "story", "leaf"]


class PrecedingReview(ContextContract):
    node_id: str
    status: str
    original_receipt: str
    scope: str = ""
    summary: str = ""
    target_id: str = ""
    evidence_requests: tuple[JsonValue, ...] = ()


class ArchivedPlanReview(PlanReview):
    original_receipt: str


class PendingReview(ContextContract):
    status: Literal["pending"]


class CertificateReference(ContextContract):
    verification: str
    certificate_digest: str


class AttemptReceipt(ContextContract):
    task_id: str = ""
    verifier_attempt: str = ""
    summary: str = ""
    session: str = ""
    progress_digest: str = ""
    errors: tuple[JsonValue, ...] = ()
    id: str
    path: str
    started: float
    state: str
    leaf_id: str
    spec_revision: int
    tree_revision: int
    actor_attempt: str
    prerequisites: dict[str, CertificateReference]
    report: str = ""
    journal: str = ""
    ended: float | None = None
    actor_run_id: str = ""


class CallerAuditResult(ContextContract):
    errors: tuple[str, ...] = ()
    checks: dict[str, bool] = Field(default_factory=dict[str, bool])
    requirements: JsonValue = None
    iron: JsonValue = None
    homes: tuple[JsonValue, ...] = ()
    routes: tuple[JsonValue, ...] = ()


class CallerRequirements(ContextContract):
    errors: tuple[str, ...]
    requirements: JsonValue
    original_receipt: str
    original_digest: str


class ProjectCheck(ContextContract):
    identity: DecisionIdentity
    original_receipt: str
    original_digest: str
    errors: tuple[str, ...] = ()
    checks: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    requirements: JsonValue = None


class VerificationEvidence(ContextContract):
    read_scope: str = ""
    identity: DecisionIdentity
    checks: dict[str, bool]
    verified: bool
    review: VerificationReview | PendingReview
    specification: Node | NodeReference
    reads: tuple[RecordedRead, ...]
    original_receipt: str
    original_digest: str


class InitialRepairBase(ContextContract):
    original_receipt: str
    original_digest: str
    accepted: Literal[False]
    tree: TaskTree
    tree_error: str = ""


class RejectedProposal(ContextContract):
    original_receipt: str
    original_digest: str
    kind: str = ""
    goal_id: str = ""
    removed_nodes: tuple[str, ...] = ()
    changed_nodes: tuple[JsonValue, ...] = ()
    proposed_topology: tuple[JsonValue, ...] = ()
    stable_parent_ids: tuple[str, ...] = ()
    candidate_tree: TaskTree | None = None
    candidate: JsonValue = None
    validation_errors: tuple[JsonValue, ...] = ()
    repair: str = ""
    error: str = ""
    original_characters: int = 0
    excerpt_start: int = 0
    excerpt: str = ""
    comparison_tree: str = ""
    first_rejected_proposal: RejectedProposal | None = None


class DecisionScope(ContextContract):
    target: str
    node_ids: tuple[str, ...]
    meaning: str


class ReplacementBinding(ContextContract):
    id: str
    parent: str | None
    conditions: str


class BodyLife(ContextContract):
    body_session: str
    death_count: int


class RecoveryFacts(ContextContract):
    causes: tuple[str, ...]
    source: str = ""
    source_index: int | None = None
    seq: int
    started_at: float
    life: BodyLife | None
    current_body_cells: dict[str, Point] | None = None
    health_recovery: JsonValue = None
    next_step: str = ""
    observed_route: JsonValue = None
    lava_cells: tuple[Point, ...] = ()
    prior_life_unverified: bool = False


class ServiceWait(ContextContract):
    source: str = ""
    source_index: int | None = None
    seq: int
    error: str = ""
    reason: str = ""
    observed_at: float


class GameplayOutcome(ContextContract):
    kind: str
    unsuccessful: bool
    reason: str | None = None
    quantities: dict[str, float] = Field(default_factory=dict[str, float])
    inventory_delta: JsonValue = None
    made_progress: bool = False
    completion_verified: bool
    source: str = ""
    source_index: int | None = None
    seq: int
    time: float
    tool: str = ""
    arguments: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    previous_life: BodyLife | None = None
    current_life: BodyLife | None = None
    failed_episode: RecoveryFacts | None = None
    inventory_missing_after_death: tuple[JsonValue, ...] = ()
    inventory_scope: str = ""


class GameplayState(ContextContract):
    position: WorldPosition | None = None
    dimension: str | None = None
    inventory: tuple[JsonValue, ...] = ()
    held_item: JsonValue = None


class GameplayMethod(ContextContract):
    tool: str
    arguments: dict[str, JsonValue]
    ineffective: int
    state: GameplayState
    last: GameplayOutcome


class Diagnosis(ContextContract):
    event: Literal["diagnosis"]
    revision: int | None = Field(default=None, ge=0, description="Managed instruction revision recorded by the body gateway, when available.")
    diagnosis: str
    evidence_sequences: tuple[int, ...]
    next_tool: str
    next_arguments: dict[str, JsonValue]
    recovery_action: bool
    source: str = ""
    source_index: int | None = None
    seq: int | None = None
    time: float = 0


class NavigationFacts(ContextContract):
    ineffective: int
    last: GameplayOutcome | None = None
    position: WorldPosition | None = None
    observed_at: float = 0


class HarvestFailure(ContextContract):
    source: str = ""
    source_index: int | None = None
    seq: int
    time: float
    block: JsonValue = None
    tool_used: JsonValue = None


class HarvestingFacts(ContextContract):
    ineffective: int
    last_failure: HarvestFailure | None


class SearchEvidence(ContextContract):
    source: str = ""
    source_index: int | None = None
    seq: int
    observed_at: float
    world: str
    dimension: str
    method: str
    pattern: str | None
    coverage: JsonValue
    total_matches: int | None
    available: bool
    uncertainty: str


class ArchivedMethodSummary(ContextContract):
    tool: str
    method_count: int
    source_sequences: dict[str, tuple[int, ...]]
    reasons: dict[str, int]
    quantities: dict[str, float]


class ArchivedDiagnosis(ContextContract):
    source: str
    time: float
    evidence_sequences: tuple[int, ...]


class GameplayScope(ContextContract):
    node_ids: tuple[str, ...]
    method_node_ids: tuple[str, ...]
    journals: tuple[str, ...]
    meaning: str


class GameplayHistory(ContextContract):
    scope: GameplayScope | None = None
    archived_methods: tuple[ArchivedMethodSummary, ...] = ()
    archived_diagnoses: tuple[ArchivedDiagnosis, ...] = ()
    decision_scope: str = ""
    recovery: RecoveryFacts | None
    service_wait: ServiceWait | None
    life: BodyLife | None
    failed_recoveries: tuple[GameplayOutcome, ...]
    navigation: NavigationFacts
    harvesting: HarvestingFacts
    methods: tuple[GameplayMethod, ...]
    searches: tuple[SearchEvidence, ...]
    outcome_count: int
    recent_outcomes: tuple[GameplayOutcome, ...]
    diagnoses: tuple[Diagnosis, ...]
    completion_verified: bool
    inventory_items: tuple[JsonValue, ...]
    source_paths: tuple[str, ...]

    @model_validator(mode="before")
    @classmethod
    def read_saved_columns(cls, value: object) -> object:
        if isinstance(value, cls):
            return value
        wire = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(value)
        if wire.get("schema") != "pm.gameplay-planning.v4":
            return value
        names = TypeAdapter[list[str]](list[str]).validate_python(wire["tool_names"])
        kinds = TypeAdapter[list[str]](list[str]).validate_python(wire["outcome_kinds"])
        fields = TypeAdapter[list[str]](list[str]).validate_python(wire["outcome_fields"])
        states = TypeAdapter[list[dict[str, JsonValue]]](list[dict[str, JsonValue]]).validate_python(wire["method_states"])
        inventory_items = TypeAdapter[list[JsonValue]](list[JsonValue]).validate_python(wire["inventory_items"])
        methods = TypeAdapter[list[list[JsonValue]]](list[list[JsonValue]]).validate_python(wire["methods"])
        decoded: list[JsonValue] = []
        last_rows: list[dict[str, JsonValue]] = []
        for row in methods:
            tool_index = TypeAdapter[int](int).validate_python(row[0])
            state_index = TypeAdapter[int](int).validate_python(row[3])
            values = TypeAdapter[list[JsonValue]](list[JsonValue]).validate_python(row[4])
            outcome = dict(zip(fields, values, strict=True))
            kind_index = TypeAdapter[int](int).validate_python(outcome.pop("kind_index"))
            additional = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(outcome.pop("additional_fields"))
            outcome.update(additional)
            outcome["kind"] = kinds[kind_index]
            outcome["tool"] = names[tool_index]
            outcome["arguments"] = row[1]
            state = dict(states[state_index])
            indexes = TypeAdapter[list[int]](list[int]).validate_python(state.pop("inventory_indexes"))
            held = TypeAdapter[int | None](int | None).validate_python(state.pop("held_item_index"))
            state["inventory"] = [inventory_items[index] for index in indexes]
            state["held_item"] = inventory_items[held] if held is not None else None
            decoded.append({"tool": names[tool_index], "arguments": row[1], "ineffective": row[2],
                "state": state, "last": outcome})
            last_rows.append(outcome)
        recent: list[JsonValue] = []
        for raw in TypeAdapter[list[dict[str, JsonValue]]](list[dict[str, JsonValue]]).validate_python(wire["recent_outcomes"]):
            if "method_index" in raw:
                index = TypeAdapter[int](int).validate_python(raw["method_index"])
                row = dict(last_rows[index])
                changes = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(raw.get("changes", {}))
                row.update(changes)
                recent.append(row)
            else:
                recent.append(raw)
        defaults = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(wire["search_defaults"])
        searches: list[JsonValue] = []
        for raw in TypeAdapter[list[dict[str, JsonValue]]](list[dict[str, JsonValue]]).validate_python(wire["searches"]):
            search = dict(defaults)
            search.update(raw)
            searches.append(search)
        result = dict(wire)
        for name in ("schema", "tool_names", "outcome_kinds", "outcome_fields", "method_fields", "method_states", "search_defaults"):
            result.pop(name)
        result["methods"] = decoded
        result["recent_outcomes"] = recent
        result["searches"] = searches
        return result


class BaseDecisionContext(ContextContract):
    identity: DecisionIdentity
    project_intent: ProjectIntent | None = None
    context_archive: ArchiveReference | None = None
    evidence_catalog: EvidenceCatalog | None = None
    evidence_catalog_receipt: ArchiveReference | None = None
    retrieved_evidence: tuple[SourceRead, ...] = ()


class PlanningContext(BaseDecisionContext):
    goal: str
    observation: JsonValue
    server_rules: str
    planning_mode: str
    project_initial_observation: JsonValue
    previous_tree: TaskTree | None
    accepted_nodes: tuple[str, ...]
    feedback: str
    refinement_target: str
    answer: str
    latest_attempt: AttemptReceipt | None
    latest_project_check: ProjectCheck | None = None
    initial_repair_base: InitialRepairBase | None = None
    current_tool_contracts: tuple[MinecraftActionTool, ...] = ()
    gameplay_history: GameplayHistory
    retained_acceptance_candidates: tuple[RetainedCandidate, ...] = ()
    refinement_scope_reason: str = ""
    latest_plan_review: ArchivedPlanReview | None = None
    latest_verification: VerificationEvidence | None = None
    previous_site_survey: SurveyEvidence | None = None
    rejected_proposal: RejectedProposal | None = None
    requested_world_survey: SurveyEvidence | None = None
    plan_constraints: CallerRequirements | None = None
    full_history: ArchiveReference | None = None
    tree_topology: tuple[NodeTopology, ...] = ()
    decision_scope: DecisionScope | None = None
    replacement_binding: ReplacementBinding | None = None


class ReviewContext(BaseDecisionContext):
    goal: str
    tree: TaskTree
    review_policy: ReviewPolicy
    plan_constraints: CallerRequirements | None
    latest_project_check: ProjectCheck | None
    server_rules: str
    leaf_prerequisites: dict[str, tuple[str, ...]]
    accepted_nodes: tuple[str, ...]
    active_recovery: RecoveryFacts | None
    project_initial_observation: JsonValue
    original_observation: JsonValue
    observation_path: str
    current_site_survey: SurveyEvidence
    survey_path: str
    tree_topology: tuple[NodeTopology, ...]
    review_scope: ReviewScope
    dependency_specifications: tuple[Node, ...]
    accepted_dependencies: dict[str, AcceptedInput]
    preceding_reviews: tuple[PrecedingReview, ...]


class RetentionContext(BaseDecisionContext):
    goal: str
    observation: JsonValue
    feedback: str
    current_nodes: tuple[Node, ...]
    candidates: tuple[RetentionCandidate, ...]


class VerificationContext(BaseDecisionContext):
    review_policy: ReviewPolicy
    specification: Node
    reads: tuple[RecordedRead, ...]
    checks: dict[str, bool]
    original_receipt: str


DecisionContext = PlanningContext | ReviewContext | RetentionContext | VerificationContext


class DecisionRole(StrEnum):
    Reviewer = "reviewer"
    Retention = "retention"
    Verifier = "verifier"


class SourceCatalogArchive(ContextContract):
    sources: dict[str, SourceBinding]


class SourceReadBatch(ContextContract):
    reads: tuple[SourceRead, ...]


class DecisionInputReceipt(ContextContract):
    identity: DecisionIdentity
    messages: tuple[JsonValue, ...]
    context: DecisionContext
    input_tokens: int
    input_ceiling: int


class RejectedDecision(ContextContract):
    text: str
    error: str


class RawDecision(ContextContract):
    text: str
