"""Validate evaluation boundaries and compute scores from explicit judgments.

Model judgments are observations, not truth labels. Unknown criteria, invented
citations and incomplete panels fail validation. Uncertain criteria earn no
credit. Suite selection never changes the declared acceptance denominator.
"""
from __future__ import annotations

from datetime import datetime
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Closed(BaseModel):
    """Reject undeclared fields at an actor or reviewer boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicMessage(Closed):
    """One delivered speech event, without internal identity or metadata."""

    id: str
    at: datetime
    speaker: Literal["Jesse", "Spamuel"]
    text: str = Field(min_length=1)


class Fact(Closed):
    """A selected observation for the evidence reviewer, with an explicit scope."""

    id: str
    at: datetime
    about: str
    text: str


class Criterion(Closed):
    """One equal-weight criterion assigned to exactly one review audience."""

    id: str
    audience: Literal["editor", "grounding"]
    description: str
    critical: bool = False


class ReviewPacket(Closed):
    """The entire model-visible review input. No private setup enters it."""

    audience: Literal["editor", "grounding"]
    premise: str
    conversation: tuple[PublicMessage, ...]
    criteria: tuple["ReviewCriterion", ...]
    facts: tuple[Fact, ...] = ()
    review_from_message: str | None = None

    @model_validator(mode="after")
    def audience_boundary(self):
        """Keep implementation evidence outside the literary review."""
        if self.audience == "editor" and self.facts:
            raise ValueError("The editor cannot receive private evidence.")
        ids = [item.id for item in self.conversation] + [item.id for item in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("Review source identifiers must be unique.")
        if not self.criteria or len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError("A review requires unique criteria.")
        if self.review_from_message is not None and self.review_from_message not in {item.id for item in self.conversation}:
            raise ValueError("The review boundary must name a supplied public message.")
        return self


class ReviewCriterion(Closed):
    """Expose a criterion without its gate weight, audience or critical flag."""

    id: str
    description: str


class JessePacket(Closed):
    """Give Jesse only human circumstances, the current beat and public speech."""

    situation: str
    current_beat: str
    conversation: tuple[PublicMessage, ...]
    delivery_mode: Literal["normal", "overlap"] = "normal"


class Citation(Closed):
    """A verbatim span in an allowed message or selected observation."""

    source_id: str
    quote: str = Field(min_length=1)


class Judgment(Closed):
    """One model judgment. The host computes all numerical results."""

    criterion_id: str
    verdict: Literal["pass", "good_enough", "fail", "uncertain"]
    reason: str = Field(min_length=1)
    citations: tuple[Citation, ...] = ()


class Review(Closed):
    """A complete review from one fresh model request."""

    judgments: tuple[Judgment, ...]
    summary: str = Field(min_length=1)


class CitationError(ValueError):
    """A known source exists, but the model's quotation is not verbatim."""


def validate_review(review: Review, packet: ReviewPacket) -> None:
    """Reject missing criteria and citations that do not match supplied text."""
    wanted = {item.id for item in packet.criteria}
    actual = [item.criterion_id for item in review.judgments]
    if len(actual) != len(wanted) or set(actual) != wanted:
        raise ValueError("The review must judge each supplied criterion exactly once.")
    texts = {item.id: item.text for item in (*packet.conversation, *packet.facts)}
    new_companion_ids = None
    if packet.review_from_message is not None:
        boundary = next(index for index, item in enumerate(packet.conversation)
                        if item.id == packet.review_from_message)
        new_companion_ids = {item.id for item in packet.conversation[boundary:] if item.speaker == "Spamuel"}
    for judgment in review.judgments:
        if judgment.verdict != "uncertain" and not judgment.citations:
            raise ValueError("A decided criterion requires a source quotation.")
        for citation in judgment.citations:
            if citation.source_id not in texts:
                raise ValueError("A review citation does not match its supplied source.")
        if judgment.verdict in {"pass", "good_enough"} and new_companion_ids is not None and not any(
                item.source_id in new_companion_ids for item in judgment.citations):
            raise ValueError("A pass requires a quotation from new companion speech, not historical background alone.")
    # Quotation repair cannot repair an invalid source or a wrong review target.
    # Check those contracts across the whole panel before selecting any spans.
    for judgment in review.judgments:
        for citation in judgment.citations:
            if citation.quote not in texts[citation.source_id]:
                raise CitationError("A review quotation is not a verbatim span of its known source.")


class Scene(Closed):
    """A human-directed episode. Future beats stay outside Jesse's input."""

    human_beat: str
    messages: int = Field(default=6, ge=0)
    quiet_seconds: float = Field(default=0, ge=0)
    restart_before: bool = False
    offline_seconds: float = Field(default=0, ge=0)
    burst_size: int = Field(default=1, ge=1, le=5)
    await_reply: bool = True


class Scenario(Closed):
    """Keep private direction separate from the public story premise."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    minecraft: bool = False
    priority: Literal["normal", "edge"] = "normal"
    chronology: Literal["recent", "long-gap"] = "recent"
    premise: str
    human_situation: str
    state_evidence: bool = False
    scenes: tuple[Scene, ...] = Field(min_length=2)
    criteria: tuple[Criterion, ...] = Field(min_length=1)


class Gate(Closed):
    """Keep thresholds in versioned settings, with exact fraction arithmetic."""

    strong_case_percent: int = Field(default=75, ge=0, le=100)
    strong_suite_percent: int = Field(default=80, ge=0, le=100)
    minimum_numerator: int = Field(default=1, ge=0)
    minimum_denominator: int = Field(default=3, gt=0)
    review_repeats: int = Field(default=2, ge=1)


class Suite(Closed):
    """Declare the full repertoire, reviewer rubric and run controls."""

    version: str
    cadence_policy: Literal["legacy", "normal_wait_v1"] = "legacy"
    gate: Gate = Gate()
    response_wait_seconds: float = Field(default=180, gt=0)
    settle_seconds: float = Field(default=120, ge=0)
    case_timeout_seconds: float = Field(default=10800, gt=0)
    review_timeout_seconds: float = Field(default=900, gt=0)
    common_criteria: tuple[Criterion, ...]
    scenarios: tuple[Scenario, ...]

    @model_validator(mode="after")
    def repertoire_boundary(self):
        """Require a fixed denominator and exactly one quarter Minecraft cases."""
        ids = [item.id for item in self.scenarios]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("The suite requires unique scenarios.")
        if sum(item.minecraft for item in self.scenarios) * 4 != len(self.scenarios):
            raise ValueError("Exactly 25 percent of declared scenarios must use Minecraft.")
        for scenario in self.scenarios:
            criteria = self.common_criteria + scenario.criteria
            if len(criteria) != len({item.id for item in criteria}):
                raise ValueError("Scenario criteria must have unique identifiers.")
        if self.cadence_policy == "normal_wait_v1":
            if sum(item.priority == "normal" for item in self.scenarios) * 5 < len(self.scenarios) * 4:
                raise ValueError("Normal use must remain at least 80 percent of the repertoire.")
            for scenario in self.scenarios:
                if scenario.priority == "normal" and any(scene.burst_size > 1 for scene in scenario.scenes):
                    raise ValueError("Message bursts belong to explicit lower-priority edge cases.")
                for index, scene in enumerate(scenario.scenes):
                    if not scene.await_reply and scene.messages:
                        later_quiet = index + 1 < len(scenario.scenes) and not scenario.scenes[index+1].messages and scenario.scenes[index+1].quiet_seconds > 0
                        if scene.messages != 1 or not (scene.quiet_seconds > 0 or later_quiet):
                            raise ValueError("No-reply announcements must lead to an explicit quiet interval.")
        if self.gate.minimum_numerator > self.gate.minimum_denominator:
            raise ValueError("The minimum fraction cannot exceed one.")
        return self


def score_case(scenario: Scenario, suite: Suite, panels: dict[str, list[Review]], *, runtime_ok: bool) -> dict:
    """Credit repeated sufficient judgments and retain minor discrepancies.

    A failure in either repeat prevents credit. Missing reviews remain
    incomplete. The model classifies materiality; this function only counts.
    """
    criteria = suite.common_criteria + scenario.criteria
    complete = runtime_ok and all(len(panels.get(audience, [])) == suite.gate.review_repeats
                                 for audience in ("editor", "grounding"))
    decisions = {}
    for criterion in criteria:
        values = [next(item.verdict for item in review.judgments if item.criterion_id == criterion.id)
                  for review in panels.get(criterion.audience, [])]
        decisions[criterion.id] = ("good_enough" if len(values) == suite.gate.review_repeats
                                    and set(values) <= {"pass", "good_enough"} and "good_enough" in values
                                    else "pass" if len(values) == suite.gate.review_repeats and set(values) == {"pass"}
                                    else "fail" if "fail" in values else "uncertain")
    passed = sum(value in {"pass", "good_enough"} for value in decisions.values())
    fraction = Fraction(passed, len(criteria))
    critical = [item.id for item in criteria if item.critical and decisions[item.id] == "fail"]
    unresolved = [item.id for item in criteria if item.critical and decisions[item.id] == "uncertain"]
    return {"scenario": scenario.id, "complete": complete, "passed_criteria": passed, "total_criteria": len(criteria),
            "fraction": str(fraction), "criteria": decisions, "critical_failures": critical,
            "critical_unresolved": unresolved,
            "manual_review_required": [], "review_protocol_complete": True,
            "strong": complete and not critical and not unresolved and fraction >= Fraction(suite.gate.strong_case_percent, 100),
            "floor": complete and fraction >= Fraction(suite.gate.minimum_numerator, suite.gate.minimum_denominator)}


def score_suite(suite: Suite, cases: list[dict]) -> dict:
    """Never turn a selected subset or an unavailable case into a suite pass."""
    rows = {item["scenario"]: item for item in cases}
    if len(rows) != len(cases) or set(rows) - {item.id for item in suite.scenarios}:
        raise ValueError("Suite results contain duplicate or undeclared cases.")
    missing = [item.id for item in suite.scenarios if item.id not in rows or not rows[item.id]["complete"]]
    strong = sum(item["strong"] for item in cases)
    floors = all(item["floor"] for item in cases)
    critical = {item["scenario"]: item["critical_failures"] for item in cases if item["critical_failures"]}
    unresolved = {item["scenario"]: item["critical_unresolved"] for item in cases if item.get("critical_unresolved")}
    manual = {item["scenario"]: item["manual_review_required"] for item in cases if item.get("manual_review_required")}
    normal = [item.id for item in suite.scenarios if item.priority == "normal"]
    strong_normal = sum(rows[item]["strong"] for item in normal if item in rows)
    normal_ok = suite.cadence_policy == "legacy" or Fraction(strong_normal, len(normal)) >= Fraction(suite.gate.strong_suite_percent, 100)
    passed = not missing and floors and not critical and not unresolved and not manual and normal_ok and Fraction(strong, len(suite.scenarios)) >= Fraction(suite.gate.strong_suite_percent, 100)
    uncertain = any(value == "uncertain" for item in cases for value in item["criteria"].values())
    return {"passed": passed, "cadence_policy": suite.cadence_policy,
            "normal_cases": len(normal) if suite.cadence_policy != "legacy" else None,
            "strong_normal_cases": strong_normal if suite.cadence_policy != "legacy" else None,
            "status": "incomplete" if missing else "pass" if passed else "needs_review" if (uncertain or manual) and not critical else "fail",
            "declared_cases": len(suite.scenarios), "evaluated_cases": len(cases), "strong_cases": strong,
            "missing_or_incomplete": missing, "critical_failures": critical, "critical_unresolved": unresolved,
            "manual_review_required": manual,
            "minecraft_cases": sum(item.minecraft for item in suite.scenarios), "human_review": "pending", "cases": cases}
