"""Pure codelet scheduling: catalogue, signals, activation, cooldown, budget.

A codelet is a small, declared process that can offer a proposal to attention
competition. This module decides *which* codelets run in one cycle. It never
calls a model, reads the MemoryStore, or writes files. The codelets stage
calls it with values that the main module derived from committed state.

The catalogue is a workflow resource (``stages/codelets/catalogue.yaml``).
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

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pm.records.mental_state import FullMentalState
from workflows.main_cognition.affect_dynamics import THREAT_APPRAISAL_THRESHOLD

DEFAULT_CATALOGUE = Path(__file__).resolve().parent / "stages" / "codelets" / "catalogue.yaml"

CODELET_KINDS = ("pure", "llm")
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
        actual = float(signals.get(self.signal, 0.0))
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
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    family: str
    kind: str
    proposal: str
    origin: str = "state"
    reality: str
    description: str = ""
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
        return self


class AttentionPolicy(BaseModel):
    """Per-loop attention settings.

    The main loop uses ``main``. A17 adds an ``inner`` policy for the second
    conscious loop without a code change.
    """

    model_config = ConfigDict(extra="forbid")

    activation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_llm_codelets: int = Field(default=3, ge=0)
    danger_llm_codelets: int = Field(default=0, ge=0)
    capacity: int = Field(default=4, ge=1)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    score_batch_size: int = Field(default=8, ge=1)
    weights: dict[str, float] = Field(default_factory=lambda: {
        "urgency": 0.35, "novelty": 0.15, "goal_relevance": 0.3, "reportability": 0.2})
    danger_urgency_bonus: float = Field(default=0.3, ge=0.0)
    reserved_codelets: dict[str, int] = Field(default_factory=dict)


class Catalogue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    policy: dict[str, AttentionPolicy]
    codelets: list[CodeletDefinition]

    @model_validator(mode="after")
    def unique_names(self):
        names = [item.name for item in self.codelets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate codelet names: {', '.join(duplicates)}.")
        return self


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
    catalogue = Catalogue.model_validate(yaml.safe_load(raw.decode("utf-8")))
    return LoadedCatalogue(catalogue, str(resolved), hashlib.sha256(raw).hexdigest()[:16])


def state_signals(state: FullMentalState | None) -> dict[str, float]:
    """Flatten the accepted mental state into named scheduler signals.

    Names are ``emotion.<field>``, ``core.<field>``, and ``needs.<field>``.
    ``core.negative_valence`` is ``max(0, -valence)``, because a gate needs a
    positive number for "how bad does it feel".
    """
    if state is None:
        return {}
    signals: dict[str, float] = {}
    for prefix, container in (("emotion", state.state_emotions), ("core", state.state_core),
                              ("needs", state.state_needs)):
        for name in type(container).model_fields:
            value = getattr(container, name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                signals[f"{prefix}.{name}"] = round(float(value), 4)
    valence = signals.get("core.valence", 0.0)
    signals["core.negative_valence"] = round(max(0.0, -valence), 4)
    signals["core.positive_valence"] = round(max(0.0, valence), 4)
    emotions = {key: value for key, value in signals.items() if key.startswith("emotion.")}
    signals["emotion.strongest"] = max(emotions.values(), default=0.0)
    return signals


def request_signals(request: dict) -> dict[str, float]:
    """Derive event signals from the cycle request and earlier stage reports.

    ``stimulus.user`` and ``stimulus.thought`` are separate signals. A codelet
    gated on a private thought therefore never fires for an outside message.
    """
    kinds = [str(item.get("kind", "")) for item in request.get("stimuli", [])]
    sources = [str(item.get("source", "")) for item in request.get("stimuli", [])]
    signals = {
        "stimulus.count": float(len(kinds)),
        "stimulus.user": 1.0 if any(kind == "user.input" for kind in kinds) or "user_input" in sources else 0.0,
        "stimulus.thought": 1.0 if "thought.created" in kinds else 0.0,
        "stimulus.workflow": 1.0 if any(kind.startswith("workflow.") for kind in kinds) else 0.0,
        "stimulus.system": 1.0 if "system.lifecycle" in kinds else 0.0,
        "goals.active": float(len(request.get("active_goal_records") or request.get("active_goals") or [])),
        "memories.selected": float(len((request.get("context") or {}).get("candidate_ids", []))),
    }
    idle = request.get("seconds_since_user_input")
    signals["time.idle_hours"] = round(float(idle) / 3600.0, 4) if idle is not None else 0.0
    affect = request.get("affect") or {}
    signals["regulation.reactive"] = 1.0 if affect.get("regulation") == "reactive" else 0.0
    appraisal = request.get("appraisal") or {}
    error = float(appraisal.get("expectation_error", 0.0) or 0.0)
    signals["appraisal.disappointment"] = round(max(0.0, -error), 4)
    signals["appraisal.pleasant_surprise"] = round(max(0.0, error), 4)
    for axis, value in (appraisal.get("appraisal") or {}).items():
        if isinstance(value, (int, float)):
            signals[f"appraisal.{axis}"] = float(value)
    return signals


def in_danger(signals: dict[str, float]) -> bool:
    """Return True when the cycle faces a physical danger.

    Danger is an appraised bodily threat at or above the A07 threat
    threshold. Fear alone is not danger: in the A15 live gate a private
    thought about losing Jesse raised fear above 0.55 with no threat at all.
    """
    return float(signals.get("appraisal.bodily_threat", 0.0)) >= THREAT_APPRAISAL_THRESHOLD


@dataclass
class Decision:
    """The scheduler's verdict for one codelet in one cycle."""

    definition: CodeletDefinition
    run: bool
    reason: str
    activation: float
    cause: list[str] = field(default_factory=list)


def _cooldown(definition: CodeletDefinition, history: dict) -> bool:
    """Return True while the codelet must wait.

    ``history`` comes from committed ``AttentionRecord`` evidence, so a
    restart keeps cooldowns. ``cycles_since`` counts committed cycles since the
    codelet last produced a nomination. ``seconds_since`` is wall time.
    """
    if not history:
        return False
    cycles = history.get("cycles_since")
    seconds = history.get("seconds_since")
    if definition.cooldown_ticks and cycles is not None and int(cycles) < definition.cooldown_ticks:
        return True
    if definition.cooldown_seconds and seconds is not None and float(seconds) < definition.cooldown_seconds:
        return True
    return False


def schedule(
    loaded: LoadedCatalogue,
    signals: dict[str, float],
    history: dict[str, dict],
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
            value = float(signals.get(boost.signal, 0.0))
            activation += boost.weight * value
            if boost.weight and value:
                cause.append(f"{boost.signal}={value:g}")
        activation = round(min(1.0, max(0.0, activation)), 4)
        if closed:
            decisions.append(Decision(definition, False, "gate", activation, ["closed: " + "; ".join(closed)]))
        elif _cooldown(definition, history.get(definition.name, {})):
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
    "in_danger",
    "load_catalogue",
    "request_signals",
    "schedule",
    "state_signals",
]
