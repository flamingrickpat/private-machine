from __future__ import annotations

from workflows.minecraft_body.spatial import WorldPosition
from workflows.minecraft_body.world_text import foreign_operands_text, observation_text, inventory_text, WorldItem, item_text, world_result_text
from workflows.minecraft_project.schemas import VerificationReview, Node, Condition, Point
from workflows.minecraft_project.planning.context_contracts import (
    DecisionContext, PlanningContext, ReviewContext, RetentionContext, VerificationContext, ProjectIntent,
    DecisionIdentity, SurveyEvidence, RecordedRead, AcceptanceCertificate, GameplayHistory, RecoveryFacts,
    GameplayOutcome, GameplayMethod, VerificationEvidence, RejectedProposal, CallerRequirements)
from workflows.minecraft_project.planning.rendering import TechnicalSection



def caller_requirements_text(requirements: CallerRequirements) -> str:
    return (f"Original {requirements.original_receipt}; digest {requirements.original_digest}\n"
        + "\n".join(requirements.errors) + "\n" + foreign_operands_text(requirements.requirements))

def point_text(point: Point | WorldPosition) -> str:
    return f"({point.x}, {point.y}, {point.z})"


def condition_text(condition: Condition) -> str:
    lines = [f"{condition.id}: {condition.description}; kind={condition.kind}; minimum={condition.minimum}; dimension={condition.dimension}"]
    if condition.node_id:
        lines.append("Accepted node: " + condition.node_id)
    if condition.names:
        lines.append("Names: " + ", ".join(condition.names))
    if condition.position is not None:
        lines.append("Position: " + point_text(condition.position))
    if condition.positions:
        lines.append("Exact cells: " + ", ".join(point_text(point) for point in condition.positions))
    if condition.end is not None:
        lines.append("Inclusive end: " + point_text(condition.end))
    if condition.kind == "position":
        lines.append(f"Radius: {condition.radius}")
    return "\n".join(lines)


def node_text(node: Node) -> str:
    lines = [f"{node.id} revision {node.revision}: {node.kind}; parent={node.parent}; specialist={node.specialist}", node.instruction]
    if node.children:
        lines.append("Children: " + ", ".join(node.children))
    if node.depends_on:
        lines.append("Prerequisites: " + ", ".join(node.depends_on))
    if node.deferred:
        lines.append("Deferred: " + node.defer_reason)
    lines.append("Required outcomes:\n" + "\n".join(condition_text(condition) for condition in node.criteria))
    if node.assumptions:
        lines.append("Current assumptions:\n" + "\n".join(condition_text(condition) for condition in node.assumptions))
    return "\n".join(lines)


def identity_text(identity: DecisionIdentity) -> str:
    return (f"Task {identity.task_id}; tree revision {identity.tree_revision}; leaf {identity.leaf_id}; "
        f"specification revision {identity.spec_revision}; actor {identity.actor_attempt}; verifier {identity.verifier_attempt}")


def intent_text(intent: ProjectIntent) -> str:
    return (f"Original request: {intent.original_request}\nOriginal goal: {intent.original_goal}\n"
        f"Current instruction: {intent.current_instruction}\nConstraints:\n" + "\n".join(intent.constraints))


def read_text(read: RecordedRead) -> str:
    return (f"{read.tool} at {read.time}; success={read.ok}; observation available={read.observation_available}\n"
        "Arguments: " + foreign_operands_text(read.arguments) + "\nResult: " + (world_result_text(read.tool, read.result) if read.ok else foreign_operands_text(read.result))
        + ("\nResult scope: " + read.result_scope if read.result_scope else "")
        + ("\nOriginal digest: " + read.original_digest if read.original_digest else ""))


def survey_text(survey: SurveyEvidence) -> str:
    lines = [identity_text(survey.identity)]
    if survey.original_receipt:
        lines.append(f"Original receipt: {survey.original_receipt}; digest: {survey.original_digest}")
    if survey.read_scope:
        lines.append(survey.read_scope)
    if survey.recipe_scope:
        lines.append(survey.recipe_scope)
    if survey.scope:
        lines.append(survey.scope)
    for condition in survey.queries:
        lines.append("Requested query: " + condition_text(condition))
    for pattern in survey.recipe_queries:
        lines.append("Requested recipe: " + pattern)
    lines.extend(read_text(read) for read in survey.reads)
    lines.extend(read_text(read) for read in survey.recipes)
    for tool in survey.tool_contracts:
        lines.append(f"Tool {tool.name}: {tool.description}\nArguments: " + foreign_operands_text(tool.parameters))
    if survey.shared_tool_contracts:
        lines.append("Shared contracts: " + survey.shared_tool_contracts)
    return "\n\n".join(lines)


def certificate_text(certificate: AcceptanceCertificate) -> str:
    lines = [identity_text(certificate.identity), f"Exact specification digest: {certificate.spec_digest}",
        "Independent verification: " + certificate.verification]
    for node_id, expected_digest in certificate.dependencies.items():
        lines.append(f"Required certificate {node_id}: {expected_digest}")
    for path, expected_digest in certificate.files.items():
        lines.append(f"Required file {path}: {expected_digest}")
    return "\n".join(lines)


def recovery_text(recovery: RecoveryFacts) -> str:
    lines = ["Active recovery: " + ", ".join(recovery.causes),
        f"Started at {recovery.started_at}; source {recovery.source or recovery.source_index}; sequence {recovery.seq}"]
    if recovery.life is not None:
        lines.append(f"Body session {recovery.life.body_session}; observed deaths {recovery.life.death_count}")
    if recovery.prior_life_unverified:
        lines.append("The preceding life's recovery is unverified.")
    if recovery.current_body_cells is not None:
        lines.extend(f"{name}: {point_text(point)}" for name, point in recovery.current_body_cells.items())
    if recovery.lava_cells:
        lines.append("Observed lava cells: " + ", ".join(point_text(point) for point in recovery.lava_cells))
    if recovery.health_recovery is not None:
        lines.append("Current health recovery facts: " + foreign_operands_text(recovery.health_recovery))
    if recovery.observed_route is not None:
        lines.append("Observed route: " + foreign_operands_text(recovery.observed_route))
    if recovery.next_step:
        lines.append(recovery.next_step)
    return "\n".join(lines)


def outcome_text(outcome: GameplayOutcome) -> str:
    lines = [f"{outcome.kind}: {outcome.tool}; unsuccessful={outcome.unsuccessful}; progress={outcome.made_progress}; "
        f"verified={outcome.completion_verified}; source={outcome.source or outcome.source_index}; sequence={outcome.seq}; time={outcome.time}",
        "Arguments: " + foreign_operands_text(outcome.arguments)]
    if outcome.reason:
        lines.append("Reason: " + outcome.reason)
    for resource, quantity in outcome.quantities.items():
        lines.append(f"Observed quantity {resource}: {quantity}")
    if outcome.inventory_delta:
        lines.append("Observed inventory delta: " + foreign_operands_text(outcome.inventory_delta))
    if outcome.previous_life is not None:
        lines.append(f"Previous life: session {outcome.previous_life.body_session}, death count {outcome.previous_life.death_count}")
    if outcome.current_life is not None:
        lines.append(f"Current life: session {outcome.current_life.body_session}, death count {outcome.current_life.death_count}")
    if outcome.failed_episode is not None:
        lines.append("Failed episode:\n" + recovery_text(outcome.failed_episode))
    if outcome.inventory_missing_after_death:
        lines.append("Missing inventory after death: " + foreign_operands_text(list(outcome.inventory_missing_after_death)))
    if outcome.inventory_scope:
        lines.append(outcome.inventory_scope)
    return "\n".join(lines)


def method_text(method: GameplayMethod) -> str:
    lines = [f"Method {method.tool}; consecutive ineffective attempts={method.ineffective}",
        "Exact arguments: " + foreign_operands_text(method.arguments)]
    if method.state.position is not None:
        lines.append(f"Observed starting position: {point_text(method.state.position)}; dimension={method.state.dimension}")
    lines.append("Observed held item: " + (item_text(WorldItem.model_validate(method.state.held_item)) if method.state.held_item is not None else "empty"))
    lines.append("Observed supplies: " + inventory_text(method.state.inventory))
    lines.append("Last result:\n" + outcome_text(method.last))
    return "\n".join(lines)


def gameplay_sections(history: GameplayHistory, native: bool) -> list[TechnicalSection]:
    sections: list[TechnicalSection] = []
    if history.source_paths:
        sections.append(TechnicalSection("Original journal sources", "\n".join(f"{index}: {path}" for index, path in enumerate(history.source_paths)), True))
    if history.recovery is not None:
        sections.append(TechnicalSection("Active recovery", recovery_text(history.recovery), True))
    if history.service_wait is not None:
        wait = history.service_wait
        sections.append(TechnicalSection("Unavailable world service", f"{wait.error or wait.reason}; observed at {wait.observed_at}; "
            f"source {wait.source or wait.source_index}; sequence {wait.seq}", True))
    navigation = history.navigation
    text = f"Consecutive ineffective movements: {navigation.ineffective}; observed at {navigation.observed_at}"
    if navigation.position is not None:
        text += "\nLast observed position: " + point_text(navigation.position)
    if navigation.last is not None:
        text += "\n" + outcome_text(navigation.last)
    sections.append(TechnicalSection("Current navigation", text, True))
    for failure in history.failed_recoveries[-2:]:
        sections.append(TechnicalSection("Observed failed recovery", outcome_text(failure), True))
    for diagnosis in history.diagnoses[-2 if native else -4:]:
        sections.append(TechnicalSection("Latest private diagnosis", f"{diagnosis.diagnosis}\nEvidence sequences: {diagnosis.evidence_sequences}; "
            f"source {diagnosis.source or diagnosis.source_index}; time {diagnosis.time}\n"
            f"Next tool {diagnosis.next_tool}; recovery action={diagnosis.recovery_action}\nArguments: "
            + foreign_operands_text(diagnosis.next_arguments), True))
    if not native:
        for method in reversed(history.methods):
            sections.append(TechnicalSection("Recent method", method_text(method), False))
        for search in reversed(history.searches):
            text = (f"Pattern {search.pattern}; dimension {search.dimension}; observed at {search.observed_at}; "
                f"available={search.available}; total matches={search.total_matches}\n"
                f"Method: {search.method}; source {search.source or search.source_index}; sequence {search.seq}\n"
                "Coverage: " + foreign_operands_text(search.coverage) + "\n" + search.uncertainty)
            sections.append(TechnicalSection("Historical search coverage", text, False))
    return sections


def base_sections(context: DecisionContext) -> list[TechnicalSection]:
    sections = [TechnicalSection("Decision identity", identity_text(context.identity), True)]
    if context.project_intent is not None:
        sections.append(TechnicalSection("Project intent", intent_text(context.project_intent), True))
    if context.context_archive is not None:
        archive = context.context_archive
        sections.append(TechnicalSection("Full input archive", f"{archive.original_receipt}; digest={archive.original_digest}; "
            f"catalog source={archive.catalog_source}", True))
    if context.evidence_catalog is not None:
        catalog = context.evidence_catalog
        text = f"Index source: {catalog.index_source}\nNamed sources: " + ", ".join(catalog.named_sources)
        text += "\nSource groups: " + ", ".join(f"{name}: {count}" for name, count in catalog.source_groups.items())
        sections.append(TechnicalSection("Original evidence catalog", text, True))
    for read in context.retrieved_evidence:
        lines = "\n".join(f"{read.start_line + index}: {line}" for index, line in enumerate(read.lines))
        sections.append(TechnicalSection("Requested original evidence", f"{read.source}; digest {read.sha256}; "
            f"total lines {read.total_lines}\n{lines}", True))
    return sections


def verification_text(evidence: VerificationEvidence) -> str:
    specification = evidence.specification
    text = (node_text(specification) if isinstance(specification, Node)
            else f"Exact node {specification.from_node} in {specification.collection}")
    text += f"\nVerified={evidence.verified}; receipt={evidence.original_receipt}; digest={evidence.original_digest}\n"
    text += "\n".join(f"{criterion}: {passed}" for criterion, passed in evidence.checks.items())
    text += f"\nIndependent review: {evidence.review.status}"
    if isinstance(evidence.review, VerificationReview):
        text += f": {evidence.review.summary}; criteria={evidence.review.criterion_ids}"
    text += "\n" + evidence.read_scope
    text += "\n" + "\n".join(read_text(read) for read in evidence.reads)
    return text


def rejection_text(rejected: RejectedProposal) -> str:
    lines = [f"Rejected output: {rejected.original_receipt}; digest={rejected.original_digest}; kind={rejected.kind}", rejected.repair]
    if rejected.error:
        lines.append(rejected.error)
    if rejected.removed_nodes:
        lines.append("Removed stable nodes: " + ", ".join(rejected.removed_nodes))
    if rejected.candidate_tree is not None:
        lines.extend(node_text(node) for node in rejected.candidate_tree.nodes)
    if rejected.candidate is not None:
        lines.append("Rejected external model candidate: " + foreign_operands_text(rejected.candidate))
    if rejected.changed_nodes:
        lines.append("Rejected external node edits: " + foreign_operands_text(list(rejected.changed_nodes)))
    if rejected.validation_errors:
        lines.append("Validation errors: " + foreign_operands_text(list(rejected.validation_errors)))
    if rejected.excerpt:
        lines.append(f"Original output characters {rejected.original_characters}, excerpt at {rejected.excerpt_start}:\n{rejected.excerpt}")
    if rejected.first_rejected_proposal is not None:
        lines.append("First rejection:\n" + rejection_text(rejected.first_rejected_proposal))
    return "\n".join(lines)


def planning_sections(context: PlanningContext, native: bool) -> tuple[TechnicalSection, ...]:
    sections = base_sections(context)
    sections.append(TechnicalSection("Current goal", context.goal, True))
    sections.append(TechnicalSection("Current body observation", observation_text(context.observation), True))
    if context.project_initial_observation != context.observation:
        sections.append(TechnicalSection("Original body observation", observation_text(context.project_initial_observation), True))
    sections.append(TechnicalSection("Current server rules", context.server_rules, True))
    sections.append(TechnicalSection("Accepted node identities", ", ".join(context.accepted_nodes), True))
    if context.full_history is not None:
        archive = context.full_history
        sections.append(TechnicalSection("Full gameplay history archive", f"{archive.original_receipt}; digest={archive.original_digest}", True))
    if context.decision_scope is not None:
        scope = context.decision_scope
        sections.append(TechnicalSection("Evidence scope", f"Target {scope.target}; nodes {scope.node_ids}\n{scope.meaning}", True))
    if context.latest_attempt is not None:
        attempt = context.latest_attempt
        sections.append(TechnicalSection("Latest actor attempt", f"{attempt.id}; node={attempt.leaf_id}; state={attempt.state}; "
            f"revision={attempt.spec_revision}; source={attempt.path}; summary={attempt.summary}", True))
    if context.feedback:
        sections.append(TechnicalSection("Repair feedback", context.feedback, True))
    if context.answer:
        sections.append(TechnicalSection("Owner answer", context.answer, True))
    if context.previous_tree is not None:
        if native:
            sections.append(TechnicalSection("Current draft", f"Goal {context.previous_tree.goal_id}; revision {context.previous_tree.revision}. "
                "Read task_list and task_get for the exact current specifications.", True))
        else:
            sections.extend(TechnicalSection("Current specification", node_text(node), True) for node in context.previous_tree.nodes)
    if context.initial_repair_base is not None:
        repair = context.initial_repair_base
        sections.append(TechnicalSection("Unaccepted repair base", f"Original {repair.original_receipt}; digest {repair.original_digest}; {repair.tree_error}", True))
        sections.extend(TechnicalSection("Original unaccepted specification", node_text(node), True) for node in repair.tree.nodes)
    for node in context.tree_topology:
        sections.append(TechnicalSection("Task topology", f"{node.id}: {node.kind}; parent={node.parent}; children={node.children}; prerequisites={node.depends_on}", False))
    if context.refinement_target:
        sections.append(TechnicalSection("Refinement target", context.refinement_target + "\n" + context.refinement_scope_reason, True))
    if context.replacement_binding is not None:
        binding = context.replacement_binding
        sections.append(TechnicalSection("Replacement binding", f"{binding.id}; parent={binding.parent}\n{binding.conditions}", True))
    if context.latest_verification is not None:
        sections.append(TechnicalSection("Latest independent verification", verification_text(context.latest_verification), True))
    if context.latest_plan_review is not None:
        review = context.latest_plan_review
        sections.append(TechnicalSection("Latest plan review", f"{review.status}: {review.summary}; target={review.target_id}; receipt={review.original_receipt}", True))
    if context.rejected_proposal is not None:
        sections.append(TechnicalSection("Rejected proposal", rejection_text(context.rejected_proposal), True))
    if context.requested_world_survey is not None:
        sections.append(TechnicalSection("Requested world survey", survey_text(context.requested_world_survey), True))
    if context.previous_site_survey is not None:
        sections.append(TechnicalSection("Earlier site survey", survey_text(context.previous_site_survey), False))
    for candidate in context.retained_acceptance_candidates:
        specification = candidate.specification
        text = node_text(specification) if isinstance(specification, Node) else f"Node {specification.from_node} in {specification.collection}"
        text += f"\nArchive={candidate.archive}; certificate={candidate.certificate_digest}; verification={candidate.verification}"
        text += "\nChanged dependencies: " + ", ".join(candidate.changed_dependencies)
        text += "\nRequired certificates: " + ", ".join(f"{node}: {digest}" for node, digest in candidate.required_certificates.items())
        sections.append(TechnicalSection("Retained historical attainment", text, True))
    if context.plan_constraints is not None:
        sections.append(TechnicalSection("Caller plan constraints", caller_requirements_text(context.plan_constraints), True))
    if context.latest_project_check is not None:
        check = context.latest_project_check
        text = f"Original {check.original_receipt}; digest {check.original_digest}\n" + "\n".join(check.errors)
        text += "\nCaller check results: " + foreign_operands_text(check.checks)
        text += "\nCaller requirements: " + foreign_operands_text(check.requirements)
        sections.append(TechnicalSection("Latest caller check", text, True))
    sections.extend(gameplay_sections(context.gameplay_history, native))
    if not native:
        for tool in context.current_tool_contracts:
            sections.append(TechnicalSection("Current tool contract", f"{tool.name}: {tool.description}\nArguments: "
                + foreign_operands_text(tool.parameters), True))
    return tuple(sections)


def review_sections(context: ReviewContext) -> tuple[TechnicalSection, ...]:
    sections = base_sections(context)
    sections.append(TechnicalSection("Goal", context.goal, True))
    sections.append(TechnicalSection("Passing review policy", f"Mode {context.review_policy.mode}; permitted judgments: {context.review_policy.passing_judgments}", True))
    sections.extend(TechnicalSection("Reviewed specification", node_text(node), True) for node in context.tree.nodes)
    sections.extend(TechnicalSection("Required dependency specification", node_text(node), True) for node in context.dependency_specifications)
    sections.append(TechnicalSection("Current site survey", survey_text(context.current_site_survey), True))
    sections.append(TechnicalSection("Current server rules", context.server_rules, True))
    sections.append(TechnicalSection("Original observation", observation_text(context.project_initial_observation), True))
    if context.original_observation != context.project_initial_observation:
        sections.append(TechnicalSection("Planning observation", observation_text(context.original_observation), True))
    sections.append(TechnicalSection("Review scope", f"Node {context.review_scope.node_id}; kind={context.review_scope.kind}", True))
    if context.active_recovery is not None:
        sections.append(TechnicalSection("Active recovery", recovery_text(context.active_recovery), True))
    for node_id, dependency in context.accepted_dependencies.items():
        text = certificate_text(dependency.certificate) + "\n" + dependency.scope
        text += "\n" + "\n".join(f"{criterion}: {passed}" for criterion, passed in dependency.checks.items())
        sections.append(TechnicalSection("Accepted dependency " + node_id, text, True))
    sections.append(TechnicalSection("Leaf prerequisites", "\n".join(f"{node}: {dependencies}" for node, dependencies in context.leaf_prerequisites.items()), True))
    sections.extend(TechnicalSection("Task topology", f"{node.id}: {node.kind}; parent={node.parent}; children={node.children}; prerequisites={node.depends_on}", True)
        for node in context.tree_topology)
    if context.latest_project_check is not None:
        check = context.latest_project_check
        text = f"Original {check.original_receipt}; digest {check.original_digest}\n" + "\n".join(check.errors)
        text += "\nCaller checks: " + foreign_operands_text(check.checks)
        text += "\nCaller requirements: " + foreign_operands_text(check.requirements)
        sections.append(TechnicalSection("Latest project check", text, True))
    for review in context.preceding_reviews:
        text = f"{review.node_id}: {review.status}; {review.summary}; {review.scope}; receipt={review.original_receipt}"
        sections.append(TechnicalSection("Earlier scope review", text, False))
    if context.plan_constraints is not None:
        sections.append(TechnicalSection("Caller constraints", caller_requirements_text(context.plan_constraints), True))
    return tuple(sections)


def retention_sections(context: RetentionContext) -> tuple[TechnicalSection, ...]:
    sections = base_sections(context)
    sections.append(TechnicalSection("Goal", context.goal, True))
    sections.append(TechnicalSection("Current observation", observation_text(context.observation), True))
    sections.append(TechnicalSection("Repair feedback", context.feedback, True))
    sections.extend(TechnicalSection("Current specification", node_text(node), True) for node in context.current_nodes)
    for candidate in context.candidates:
        text = node_text(candidate.specification) + "\n" + certificate_text(candidate.certificate)
        text += f"\nArchive={candidate.archive}; source={candidate.source}; currently valid={candidate.currently_valid}"
        sections.append(TechnicalSection("Historical retention candidate", text, True))
    return tuple(sections)


def verification_sections(context: VerificationContext) -> tuple[TechnicalSection, ...]:
    sections = base_sections(context)
    sections.append(TechnicalSection("Exact acceptance specification", node_text(context.specification), True))
    sections.append(TechnicalSection("Independent mechanical checks", "\n".join(f"{criterion}: {passed}" for criterion, passed in context.checks.items()), True))
    sections.append(TechnicalSection("Passing review policy", f"Mode {context.review_policy.mode}; permitted judgments: {context.review_policy.passing_judgments}", True))
    sections.append(TechnicalSection("Original receipt", context.original_receipt, True))
    sections.extend(TechnicalSection("Independent original observation", read_text(read), True) for read in context.reads)
    return tuple(sections)
