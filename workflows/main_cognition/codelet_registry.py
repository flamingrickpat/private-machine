"""The codelet registry: definitions, activation, waves, and pathways.

Port of ``CodeletSignature``, ``CodeletRuntime``, ``ActivationFactors``, and
``CodeletRegistry`` from ``private-machine-head/pm/codelets/codelet.py``.

- The definitions are data in ``codelets.yaml``. A definition holds the name,
  description, families, ``data_aquisation`` list, and prompt of one old
  codelet. Any other old definition can join without a code change.
- The runtime state of each codelet lives in ``CycleState.codelets``
  (:class:`~pm.records.codelet_records.CodeletRegistryState`). It survives
  from cycle to cycle.
- The functions here are pure. They make no model calls and do no I/O,
  except :func:`load_definitions`, which reads the definitions file.

Fixes of old bugs:

- The old score used the cosine *distance* of signature and situation as a
  match. This port uses the cosine similarity.
- The old recency term ``1 - 1 / (1 + dt)`` grew with the time since the
  last run, so it penalized codelets that had not run for a long time. This
  port uses ``1 / (1 + dt)``, which penalizes a recent run.
- The old state match was a placeholder that always returned 0. This port
  matches the codelet families to the typed mental state
  (:func:`family_state_match`).
"""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.constants import (
    ALPHA_MAX_RATING_STEP,
    CODELET_ACTIVATION_DECAY,
    CODELET_BASE_FORGET,
    CODELET_BASE_LEARN_RATE,
    CODELET_COOLDOWN_FACTOR,
    CODELET_MAX_ACTIVATION,
    CODELET_SELECTOR_WEIGHT,
    CODELET_W_ACTIVATION,
    CODELET_W_BASE,
    CODELET_W_RANDOM,
    CODELET_W_RECENCY,
    CODELET_W_STATE,
    CODELET_W_STIMULUS,
    CODELET_W_VECTOR,
    PATHWAY_BOOST_SCALE,
    PATHWAY_SKIP_DECAY,
)
from pm.context_render.story import fill_story_names
from pm.records.codelet_records import CodeletRegistryState, CodeletRuntimeState
from pm.records.mental_state import FullMentalState
from pm.utils import yaml_util

from workflows.main_cognition.codelet_pathways import CodeletActivation, get_codelet_pathways
from workflows.main_cognition.codelet_percepts import CodeletFamily
from workflows.main_cognition.schemas import ActivationScore, PathwayBoost

#: The shipped definitions of the twelve selected codelets.
DEFAULT_DEFINITIONS = Path(__file__).resolve().parent / "codelets.yaml"


class CodeletDefinition(BaseModel):
    """One registered codelet, as data. Port of ``SimpleCodelet`` and ``CodeletSignature``.

    Texts can hold the placeholders ``{companion_name}`` and
    ``{entity_name}``. ``execution`` is ``chain``, ``agent``, or empty. An
    empty value selects ``agent`` when ``MemoryAccessors`` is the first
    family, else ``chain``. ``stimulus_kinds`` lists the stimulus kinds that match the codelet. An
    empty list matches every stimulus.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    description: str = ""
    families: list[CodeletFamily] = Field(default_factory=list)
    data_aquisation: list[str] = Field(default_factory=list)
    prompt: str = ""
    execution: Literal["", "chain", "agent"] = ""
    stimulus_kinds: list[str] = Field(default_factory=list)

    def mode(self) -> str:
        """Return the execution mode: ``chain`` or ``agent``."""
        if self.execution:
            return self.execution
        return "agent" if self.families[:1] == [CodeletFamily.MemoryAccessors] else "chain"

    def signature_text(self, companion_name: str, entity_name: str) -> str:
        """Return the text that the signature embedding encodes.

        Port of ``CodeletSignature.create_embedding_string``: name,
        description, prompt, and families.
        """
        text = f"{self.name}\n{self.description}\n{self.prompt}\n" + ",".join(self.families)
        return fill(text, companion_name, entity_name)


class CodeletDefinitions(BaseModel):
    """The definitions file: a version string and the registered codelets."""

    model_config = ConfigDict(extra="forbid")

    version: str = ""
    codelets: list[CodeletDefinition] = Field(default_factory=list)


class LoadedDefinitions(BaseModel):
    """Validated definitions plus the SHA-256 digest of the file bytes."""

    definitions: CodeletDefinitions = Field(default_factory=CodeletDefinitions)
    digest: str = ""

    def by_name(self) -> dict[str, CodeletDefinition]:
        """Return the definitions by codelet name, in file order."""
        return {item.name: item for item in self.definitions.codelets}


def load_definitions(path: str | Path | None = None) -> LoadedDefinitions:
    """Read and validate a definitions file. ``None`` reads the shipped file.

    Raises ValueError when two definitions have the same name. A missing or
    invalid file raises too.
    """
    data = Path(path or DEFAULT_DEFINITIONS).read_bytes()
    definitions = CodeletDefinitions.model_validate(yaml_util.safe_load(data.decode("utf-8")))
    names = [item.name for item in definitions.codelets]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate codelet names in {path or DEFAULT_DEFINITIONS}.")
    return LoadedDefinitions(definitions=definitions, digest=hashlib.sha256(data).hexdigest())


def fill(text: str, companion_name: str, entity_name: str) -> str:
    """Replace the two name placeholders. See ``pm.context_render.story.fill_story_names``."""
    return fill_story_names(text, companion_name, entity_name)


# ---- state match ----------------------------------------------------------

def _supply_pressure(state: FullMentalState) -> float:
    """Return the most pressing need: 1 minus the lowest supply need, or the closeness drive."""
    needs = state.state_needs
    supplies = [value for name, value in needs.model_dump().items() if name != "closeness_need"]
    return max(1.0 - min(supplies), needs.closeness_need, state.state_emotions.boredom)


#: How strongly the mental state calls for each family, in [0, 1].
#: The old generation left this match as a placeholder. These readouts are an
#: engineering choice of A19F step 5. Tune them with replays.
FAMILY_STATE: dict[CodeletFamily, Callable[[FullMentalState], float]] = {
    CodeletFamily.Appraisals: lambda s: max(value for name, value in s.state_emotions.model_dump().items()
                                            if name != "propriety_inhibition"),
    CodeletFamily.MemoryAccessors: lambda s: (s.state_cognition.mental_aperture + 1.0) / 2.0,
    CodeletFamily.ImaginationSimulation: lambda s: max(s.state_emotions.curiosity, s.state_emotions.playfulness,
                                                       s.state_emotions.amusement, s.state_emotions.surprise),
    CodeletFamily.ValuationTradeOffs: lambda s: 1.0 - s.appraisal_general.certainty,
    CodeletFamily.MetaMonitoringIdentity: lambda s: max(s.state_emotions.pride, s.state_emotions.shame,
                                                        s.state_emotions.guilt, s.appraisal_social.embarrassment),
    CodeletFamily.RegulationCoping: lambda s: max(s.state_emotions.anger, s.state_emotions.fear,
                                                  s.state_emotions.sadness, s.state_emotions.disgust,
                                                  s.state_emotions.resentment, s.state_emotions.jealousy),
    CodeletFamily.DriversHomeostatis: _supply_pressure,
    CodeletFamily.Attention: lambda s: s.state_core.arousal,
    CodeletFamily.ActionTendencies: lambda s: max(s.state_expression_ext.model_dump().values()),
    CodeletFamily.Narratives: lambda s: s.state_cognition.ego_strength,
}


def family_state_match(families: list[CodeletFamily], state: FullMentalState) -> float:
    """Return the mean of the family readouts of ``state``, clamped to [0, 1]."""
    if not families:
        return 0.0
    return max(0.0, min(1.0, math.fsum(FAMILY_STATE[family](state) for family in families) / len(families)))


# ---- activation -----------------------------------------------------------

def cooling(runtime: CodeletRuntimeState, tick: int) -> bool:
    """Return True while the codelet is in its cooldown after its last run."""
    return runtime.last_fired_tick is not None and tick - runtime.last_fired_tick < runtime.cooldown_ticks


def activation_score(definition: CodeletDefinition, runtime: CodeletRuntimeState, *, tick: int,
                     stimulus_kinds: list[str], state: FullMentalState, similarity: float,
                     rng: random.Random) -> ActivationScore:
    """Return the activation score of one codelet for this cycle.

    Port of ``CodeletExecutor.activation_score``. The parts:

    - ``stored``: the stored activation (selector ratings, pathway boosts).
    - ``stimulus``: 1 when the cycle has a stimulus whose kind the codelet
      accepts, else 0.
    - ``state``: :func:`family_state_match`.
    - ``vector``: ``similarity``, the cosine similarity of the signature
      embedding and the situation embedding, clamped to [0, 1].
    - ``base``: the base-level activation.
    - ``recency``: ``1 / (1 + ticks since the last run)``, or 0 before the
      first run.
    - ``noise``: uniform in [0, 1) from ``rng``.

    The weighted sum uses the ``CODELET_W_*`` constants. During the cooldown
    the sum is multiplied by ``CODELET_COOLDOWN_FACTOR``. The score is
    clamped to [0, 1].
    """
    accepted = [kind for kind in stimulus_kinds if not definition.stimulus_kinds or kind in definition.stimulus_kinds]
    stimulus = 1.0 if accepted else 0.0
    state_match = family_state_match(definition.families, state)
    vector = max(0.0, min(1.0, similarity))
    recency = 0.0 if runtime.last_fired_tick is None else 1.0 / (1.0 + max(0, tick - runtime.last_fired_tick))
    noise = rng.random()
    total = (CODELET_W_ACTIVATION * runtime.activation + CODELET_W_STIMULUS * stimulus + CODELET_W_STATE * state_match
             + CODELET_W_VECTOR * vector + CODELET_W_BASE * runtime.base_activation
             - CODELET_W_RECENCY * recency + CODELET_W_RANDOM * noise)
    is_cooling = cooling(runtime, tick)
    if is_cooling:
        total *= CODELET_COOLDOWN_FACTOR
    return ActivationScore(codelet=definition.name, score=round(max(0.0, min(1.0, total)), 4),
                           stored=round(runtime.activation, 4), stimulus=stimulus, state=round(state_match, 4),
                           vector=round(vector, 4), base=round(runtime.base_activation, 4),
                           recency=round(recency, 4), noise=round(noise, 4), cooling=is_cooling)


def ranked(scores: list[ActivationScore]) -> list[ActivationScore]:
    """Return the scores from highest to lowest. Equal scores keep their order."""
    return sorted(scores, key=lambda item: -item.score)


def decay_registry(registry: CodeletRegistryState) -> None:
    """Decay every codelet by one cycle.

    The stored activation keeps ``CODELET_ACTIVATION_DECAY`` of its value.
    The base-level activation keeps ``CODELET_BASE_FORGET`` of its value, so
    it falls slowly (LIDA attentional learning).
    """
    for runtime in registry.codelets.values():
        runtime.activation *= CODELET_ACTIVATION_DECAY
        runtime.base_activation *= CODELET_BASE_FORGET


def add_activation(runtime: CodeletRuntimeState, amount: float) -> None:
    """Add ``amount`` to the stored activation, bounded by ``CODELET_MAX_ACTIVATION``."""
    runtime.activation = max(0.0, min(CODELET_MAX_ACTIVATION, runtime.activation + amount))


def add_rating(runtime: CodeletRuntimeState, rating: float) -> float:
    """Add a selector rating (0 to 1) times ``CODELET_SELECTOR_WEIGHT`` to the stored activation.

    !!! ALPHA RUN ONLY: TEMPORARY FAILSAFE. FIX PROPERLY LATER. !!!
    The rating moves at most ``ALPHA_MAX_RATING_STEP`` from the previous
    rating of the same codelet. In the A19F review run the selector rated
    BoredomThresholdCross 0.95 on a tease. So a codelet can rise over some
    cycles of repetition, but one cycle cannot spike it. The real fix is a
    calibrated selector. Returns the rating that was applied.
    """
    if runtime.last_rating is not None:
        rating = max(runtime.last_rating - ALPHA_MAX_RATING_STEP,
                     min(runtime.last_rating + ALPHA_MAX_RATING_STEP, rating))
    runtime.last_rating = rating
    add_activation(runtime, CODELET_SELECTOR_WEIGHT * rating)
    return rating


def mark_fired(runtime: CodeletRuntimeState, tick: int, percept_ids: list[int]) -> None:
    """Record one run. Port of ``CodeletExecutor.on_commit``.

    The stored activation returns to 0, because the codelet spent it. The
    precursor ids are cleared, because the codelet has seen them.
    """
    runtime.last_fired_tick = tick
    runtime.total_runs += 1
    runtime.activation = 0.0
    runtime.last_percept_ids = list(percept_ids)
    runtime.precursor_ids = []


def reinforce(runtime: CodeletRuntimeState) -> None:
    """Raise the base-level activation after a percept of the codelet won attention.

    The base moves ``CODELET_BASE_LEARN_RATE`` of its distance toward 1.0.
    """
    runtime.base_activation += CODELET_BASE_LEARN_RATE * (1.0 - runtime.base_activation)


def wave_sizes(budget: int, waves: int) -> list[int]:
    """Split the codelet budget of a cycle into waves.

    Each wave gets ``ceil(budget / waves)`` runs until the budget ends. For
    example, a budget of 4 in 2 waves gives ``[2, 2]``, and a budget of 1
    gives ``[1]``.
    """
    size = math.ceil(budget / waves)
    sizes = []
    left = budget
    while left > 0 and len(sizes) < waves:
        sizes.append(min(size, left))
        left -= sizes[-1]
    return sizes


# ---- pathways -------------------------------------------------------------

Chain = list[str | CodeletActivation]


def pathway_boosts(source: str, registered: set[str], pathways: dict[str, list[Chain]] | None = None) -> list[PathwayBoost]:
    """Return the boosts that a run of ``source`` gives along every pathway chain.

    Port of the pathway part of ``ghost_codelets.cognitive_cycle``, with the
    A19F rule for unregistered codelets. In each chain that contains
    ``source``:

    1. The strength is the step strength right after ``source``.
    2. Walk downstream. Each unregistered codelet on the way multiplies the
       strength by ``PATHWAY_SKIP_DECAY``.
    3. The first registered codelet other than ``source`` gets the boost.

    A chain without a registered successor gives no boost. ``pathways``
    defaults to :func:`~workflows.main_cognition.codelet_pathways.get_codelet_pathways`.
    """
    boosts = []
    for name, chains in (pathways if pathways is not None else get_codelet_pathways()).items():
        for chain in chains:
            if source not in chain:
                continue
            start = chain.index(source)
            following = chain[start + 1:]
            if not following:
                continue
            strength = following[0].strength
            skipped = []
            for step in following[1:]:
                if isinstance(step, CodeletActivation):
                    continue
                if step in registered and step != source:
                    boosts.append(PathwayBoost(pathway=name, source=source, target=step,
                                               strength=round(strength * PATHWAY_SKIP_DECAY ** len(skipped), 6),
                                               skipped=skipped))
                    break
                skipped.append(step)
    return boosts


def apply_pathway_boost(registry: CodeletRegistryState, boost: PathwayBoost) -> None:
    """Add ``PATHWAY_BOOST_SCALE * strength`` to the target and record its precursors."""
    runtime = registry.runtime(boost.target)
    add_activation(runtime, PATHWAY_BOOST_SCALE * boost.strength)
    runtime.precursor_ids = list(dict.fromkeys([*runtime.precursor_ids, *boost.precursor_ids]))


__all__ = [
    "CodeletDefinition",
    "CodeletDefinitions",
    "DEFAULT_DEFINITIONS",
    "FAMILY_STATE",
    "LoadedDefinitions",
    "activation_score",
    "add_activation",
    "add_rating",
    "apply_pathway_boost",
    "cooling",
    "decay_registry",
    "family_state_match",
    "fill",
    "load_definitions",
    "mark_fired",
    "pathway_boosts",
    "ranked",
    "reinforce",
    "wave_sizes",
]
