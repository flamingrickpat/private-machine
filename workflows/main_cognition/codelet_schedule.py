"""Pure codelet scheduling: catalogue, signals, activation, cooldown, budget.

A codelet is a small, declared process that can offer a proposal to attention
competition. This module decides *which* codelets run in one cycle. It never
calls a model, reads the MemoryStore, or writes files. The codelets stage
phase calls it with values that the main module derived from committed state.

The catalogue is a workflow resource (``phases/codelet_catalogue.yaml``).
Its version and a SHA-256 digest of its bytes go into every trace, so a
changed configuration is visible in the committed evidence.

Activation rules:

- Every ``when`` condition must hold. Otherwise the codelet is skipped with
  reason ``gate``.
- Activation is ``base_activation`` plus each ``boost`` weight times its
  signal value, clamped to 0..1.
- A codelet inside its cooldown is skipped with reason ``cooldown``. Cooldown
  counts committed cycles and elapsed seconds. Either limit can hold it back.
- A codelet below the policy threshold is skipped with reason ``threshold``.
- LLM codelets run in activation order until the policy's call budget is
  spent. The rest are skipped with reason ``budget``. Pure codelets cost no
  model call and are not limited by the budget.
- In danger, the LLM budget is the policy's ``danger_llm_codelets`` value,
  normally zero. Urgent body control must not wait for codelet models.
  ``in_danger`` decides danger from the appraised bodily threat only. It
  does not use the ``reactive`` regulation mode or fear: strong emotional
  pain, or fear of losing someone, can also make regulation reactive, and the
  emotional codelets must still run at exactly that moment.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from pm.utils import yaml_util
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pm.records.mental_state import FullMentalState
from workflows.main_cognition.affect_dynamics import THREAT_APPRAISAL_THRESHOLD
from workflows.main_cognition.pathways import PathwayDefinition, pathway_map
from workflows.main_cognition.schemas import (
    PROPOSAL_KINDS, PROPOSAL_ORIGINS, REALITY_LABELS, AppraisalResult, AttentionPolicy, CodeletHistory, Stimulus,
)

DEFAULT_CATALOGUE = Path(__file__).resolve().parent / "phases" / "codelet_catalogue.yaml"

CODELET_KINDS = ("pure", "llm")


def signal(signals: dict[str, float], name: str) -> float:
    """Return one named signal.

    The signal map is sparse by design: a signal that no source supplied
    counts as 0.0. This keeps a gate on an unavailable value closed.
    """
    return float(signals.get(name, 0.0))
TRIGGER_OPERATORS = (">=", "<=", ">", "<", "==")


class Condition(BaseModel):
    """One gate condition over a named signal, such as ``emotion.sadness >= 0.3``."""

    model_config = ConfigDict(extra="forbid")

    signal: str
    op: str = ">="
    value: float = 0.0

    @model_validator(mode="after")
    def known_operator(self):
        if self.op not in TRIGGER_OPERATORS:
            raise ValueError(f"Unknown trigger operator {self.op!r}.")
        return self

    def holds(self, signals: dict[str, float]) -> bool:
        """Return True when the signal satisfies the condition.

        A signal that is absent counts as 0.0. This keeps a gate on an
        unavailable value closed instead of raising during a live cycle.
        """
        actual = signal(signals, self.signal)
        return {
            ">=": actual >= self.value,
            "<=": actual <= self.value,
            ">": actual > self.value,
            "<": actual < self.value,
            "==": actual == self.value,
        }[self.op]

    def label(self) -> str:
        return f"{self.signal} {self.op} {self.value:g}"


class Boost(BaseModel):
    """Add ``weight * signal`` to activation."""

    model_config = ConfigDict(extra="forbid")

    signal: str
    weight: float = 0.0


class CodeletDefinition(BaseModel):
    """One codelet as declared in the catalogue.

    ``kind: pure`` codelets name a Python function in the codelets stage.
    ``kind: llm`` codelets carry ``instructions`` for one short model call.
    ``reality`` is the default reality label of the proposal content.

    A16 declarations: ``pathway`` names the codelet's primary pathway (empty
    when it serves all of them), ``context`` lists the state fields, source
    categories, and temporal scope it requires, and ``consequence`` states
    the expected accepted consequence. Missing evidence stays visible: a
    declaration names what the codelet reads, never what it may invent.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    family: str
    kind: str
    proposal: str
    origin: str = "state"
    reality: str
    description: str = ""
    pathway: str = ""
    context: list[str] = Field(default_factory=list)
    consequence: str = ""
    base_activation: float = Field(default=0.5, ge=0.0, le=1.0)
    when: list[Condition] = Field(default_factory=list)
    boost: list[Boost] = Field(default_factory=list)
    cooldown_ticks: int = Field(default=0, ge=0)
    cooldown_seconds: float = Field(default=0.0, ge=0.0)
    instructions: str = ""

    @model_validator(mode="after")
    def kind_contract(self):
        if self.kind not in CODELET_KINDS:
            raise ValueError(f"Codelet {self.name}: kind must be pure or llm.")
        if self.kind == "llm" and not self.instructions.strip():
            raise ValueError(f"Codelet {self.name}: an llm codelet needs instructions.")
        if self.proposal not in PROPOSAL_KINDS:
            raise ValueError(f"Codelet {self.name}: unknown proposal kind {self.proposal!r}.")
        if self.origin not in PROPOSAL_ORIGINS:
            raise ValueError(f"Codelet {self.name}: unknown origin {self.origin!r}.")
        if self.reality not in REALITY_LABELS:
            raise ValueError(f"Codelet {self.name}: unknown reality label {self.reality!r}.")
        return self


class Catalogue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    policy: dict[str, AttentionPolicy]
    codelets: list[CodeletDefinition]
    pathways: list[PathwayDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_names(self):
        names = [item.name for item in self.codelets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate codelet names: {', '.join(duplicates)}.")
        pathways = pathway_map(self.pathways)
        known = set(names)
        for definition in self.codelets:
            if definition.pathway and definition.pathway not in pathways:
                raise ValueError(f"Codelet {definition.name} names unknown pathway {definition.pathway!r}.")
        for pathway in self.pathways:
            missing = sorted(set(pathway.members) - known)
            if missing:
                raise ValueError(f"Pathway {pathway.name} lists unknown codelets: {', '.join(missing)}.")
        return self

    def pathway_map(self) -> dict[str, PathwayDefinition]:
        return pathway_map(self.pathways)


@dataclass(frozen=True)
class LoadedCatalogue:
    """A validated catalogue and the identity of the exact file it came from."""

    catalogue: Catalogue
    path: str
    digest: str


def load_catalogue(path: str | Path | None = None) -> LoadedCatalogue:
    """Read and validate a codelet catalogue.

    Raises ``ValueError`` (pydantic ``ValidationError``) for an invalid file
    and ``FileNotFoundError`` for a missing one. A broken catalogue must stop
    the cycle; a silent empty catalogue would hide the defect.
    """
    resolved = Path(path) if path else DEFAULT_CATALOGUE
    raw = resolved.read_bytes()
    catalogue = Catalogue.model_validate(yaml_util.safe_load(raw.decode("utf-8")))
    return LoadedCatalogue(catalogue, str(resolved), hashlib.sha256(raw).hexdigest()[:16])


NEGATIVE_EMOTIONS = (
    "sadness", "anger", "fear", "disgust", "resentment", "jealousy",
    "boredom", "shame", "guilt",
)


def state_signals(state: FullMentalState | None) -> dict[str, float]:
    """Flatten the accepted mental state into named scheduler signals.

    Names are ``emotion.<field>``, ``core.<field>``, ``needs.<field>``, and
    ``relationship.<field>`` when a partner state exists.
    ``core.negative_valence`` is ``max(0, -valence)``, because a gate needs a
    positive number for "how bad does it feel".
    ``emotion.negative_strongest`` is the strongest unpleasant emotion, so a
    gate can ask "is something wrong" without listing every emotion.
    """
    if state is None:
        return {}
    signals: dict[str, float] = {}
    for prefix, container in (("emotion", state.state_emotions), ("core", state.state_core),
                              ("needs", state.state_needs), ("relationship", state.state_relationship)):
        if container is None:
            continue
        for name in type(container).model_fields:
            value = getattr(container, name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                signals[f"{prefix}.{name}"] = round(float(value), 4)
    valence = signal(signals, "core.valence")
    signals["core.negative_valence"] = round(max(0.0, -valence), 4)
    signals["core.positive_valence"] = round(max(0.0, valence), 4)
    emotions = {key: value for key, value in signals.items() if key.startswith("emotion.")}
    signals["emotion.strongest"] = max(emotions.values(), default=0.0)
    signals["emotion.negative_strongest"] = max(
        (signal(signals, f"emotion.{name}") for name in NEGATIVE_EMOTIONS), default=0.0)
    return signals


def request_signals(
    stimuli: list[Stimulus],
    *,
    active_goal_count: int,
    selected_memory_count: int,
    seconds_since_user_input: float | None,
    regulation: str,
    appraisal: AppraisalResult,
) -> dict[str, float]:
    """Derive event signals from the cycle input and earlier phase results.

    ``stimulus.user`` and ``stimulus.thought`` are separate signals. A codelet
    gated on a private thought therefore never fires for an outside message.
    ``seconds_since_user_input`` is None when no earlier user message exists.
    """
    kinds = [item.kind for item in stimuli]
    sources = [item.source for item in stimuli]
    signals = {
        "stimulus.count": float(len(kinds)),
        "stimulus.user": 1.0 if any(kind == "user.input" for kind in kinds) or "user_input" in sources else 0.0,
        "stimulus.thought": 1.0 if "thought.created" in kinds else 0.0,
        "stimulus.workflow": 1.0 if any(kind.startswith("workflow.") for kind in kinds) else 0.0,
        "stimulus.system": 1.0 if "system.lifecycle" in kinds else 0.0,
        "stimulus.world": 1.0 if "minecraft.world" in kinds else 0.0,
        "goals.active": float(active_goal_count),
        "memories.selected": float(selected_memory_count),
    }
    idle = seconds_since_user_input
    signals["time.idle_hours"] = round(float(idle) / 3600.0, 4) if idle is not None else 0.0
    signals["regulation.reactive"] = 1.0 if regulation == "reactive" else 0.0
    error = float(appraisal.expectation_error)
    signals["appraisal.disappointment"] = round(max(0.0, -error), 4)
    signals["appraisal.pleasant_surprise"] = round(max(0.0, error), 4)
    axes = {axis: float(value) for axis, value in appraisal.appraisal.items()}
    signals.update({f"appraisal.{axis}": value for axis, value in axes.items()})
    # Inverted axes. The underlying appraisal fields stay as they are; these
    # derived signals let a boost or gate rise when something is missing, such
    # as unfairness or exclusion, because activation math only adds.
    if "fairness" in axes:
        signals["appraisal.unfairness"] = round(1.0 - axes["fairness"], 4)
    if "inclusion_exclusion" in axes:
        signals["appraisal.exclusion"] = round(1.0 - axes["inclusion_exclusion"], 4)
    if "certainty" in axes:
        signals["appraisal.uncertainty"] = round(1.0 - axes["certainty"], 4)
    return signals


def in_danger(signals: dict[str, float]) -> bool:
    """Return True when the cycle faces a physical danger.

    Danger is an appraised bodily threat at or above the A07 threat
    threshold. Fear alone is not danger: in the A15 live gate a private
    thought about losing the user raised fear above 0.55 with no threat at all.
    """
    return signal(signals, "appraisal.bodily_threat") >= THREAT_APPRAISAL_THRESHOLD


@dataclass
class Decision:
    """The scheduler's verdict for one codelet in one cycle."""

    definition: CodeletDefinition
    run: bool
    reason: str
    activation: float
    cause: list[str] = field(default_factory=list)


def _cooldown(definition: CodeletDefinition, history: CodeletHistory | None) -> bool:
    """Return True while the codelet must wait.

    ``history`` comes from committed attention frame evidence, so a restart
    keeps cooldowns. It is None for a codelet that never ran.
    ``cycles_since`` counts committed cycles since the codelet last ran.
    ``seconds_since`` is wall time.
    """
    if history is None:
        return False
    cycles = history.cycles_since
    seconds = history.seconds_since
    if definition.cooldown_ticks and cycles is not None and int(cycles) < definition.cooldown_ticks:
        return True
    if definition.cooldown_seconds and seconds is not None and float(seconds) < definition.cooldown_seconds:
        return True
    return False


def schedule(
    loaded: LoadedCatalogue,
    signals: dict[str, float],
    history: dict[str, CodeletHistory],
    *,
    loop: str = "main",
) -> tuple[list[Decision], AttentionPolicy]:
    """Return one decision per catalogue codelet, in catalogue order.

    Raises ``KeyError`` when the catalogue has no policy for ``loop``.
    """
    policy = loaded.catalogue.policy[loop]
    danger = in_danger(signals)
    budget = policy.danger_llm_codelets if danger else policy.max_llm_codelets
    decisions: list[Decision] = []
    for definition in loaded.catalogue.codelets:
        closed = [condition.label() for condition in definition.when if not condition.holds(signals)]
        cause = [condition.label() for condition in definition.when if condition.holds(signals)]
        activation = definition.base_activation
        for boost in definition.boost:
            value = signal(signals, boost.signal)
            activation += boost.weight * value
            if boost.weight and value:
                cause.append(f"{boost.signal}={value:g}")
        activation = round(min(1.0, max(0.0, activation)), 4)
        if closed:
            decisions.append(Decision(definition, False, "gate", activation, ["closed: " + "; ".join(closed)]))
        elif _cooldown(definition, history.get(definition.name)):
            decisions.append(Decision(definition, False, "cooldown", activation, cause))
        elif activation < policy.activation_threshold:
            decisions.append(Decision(definition, False, "threshold", activation, cause))
        else:
            decisions.append(Decision(definition, True, "scheduled", activation, cause))
    # Spend the LLM budget on the most active codelets. Catalogue order breaks
    # ties, so the same inputs always choose the same set.
    llm = sorted((item for item in decisions if item.run and item.definition.kind == "llm"),
                 key=lambda item: -item.activation)
    for index, item in enumerate(llm):
        if index >= budget:
            item.run = False
            item.reason = "danger_budget" if danger else "budget"
    return decisions, policy


__all__ = [
    "AttentionPolicy",
    "Boost",
    "Catalogue",
    "CodeletDefinition",
    "Condition",
    "DEFAULT_CATALOGUE",
    "Decision",
    "LoadedCatalogue",
    "NEGATIVE_EMOTIONS",
    "in_danger",
    "load_catalogue",
    "request_signals",
    "schedule",
    "signal",
    "state_signals",
]
