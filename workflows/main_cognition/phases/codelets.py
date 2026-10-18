"""Codelets phase: run the registered codelets in waves and fill the CSM.

Port of the codelet part of ``GhostCodelets.cognitive_cycle`` in
``private-machine-head/pm/ghosts/ghost_codelets.py``. Steps:

1. Build the shared story prefix once
   (:class:`~workflows.main_cognition.story_prefix.StoryPrefix`): the
   story-simulator system prompt, the story (character card, the self shard
   and the group shards with their tails, the feeling, and the new stimuli),
   and an acknowledgement. Every model call of this phase starts with these
   exact bytes, so the provider can reuse the cache.
2. Score every registered codelet
   (:func:`~workflows.main_cognition.codelet_registry.activation_score`).
3. The selector call rates the top candidates. Each rating adds to the
   stored activation (port of ``AgentSelectCodelets``; the old code
   iterated the ratings without ``.items()``).
4. Run waves. Each wave scores the codelets that did not run yet and runs
   the best ones in parallel. Triage sets the total budget. A later wave sees
   the CSM after the earlier waves.
5. Each codelet sees its own view of the CSM (``SituationalModel.snapshot_for``).
   A ``chain`` codelet makes two calls: prose, then typed percepts. An
   ``agent`` codelet runs one pm-coder session through the ``codelet_agent``
   tool, then the same percept call.
6. Mini-appraisal: each percept copies the current mental state, sets its
   appraisal axes, and runs ``compute_state_delta``. The percept stores both
   vectors. It does not change the mental state until attention puts it into
   a workspace shard. Then it becomes causal.
7. Each percept and each recommendation enters the CSM under a pending id.
   The records it is about get an endorsement from the codelet.
8. Pathways boost the next registered codelet downstream.
9. Prune the CSM. The CSM requirements protect their items.

Model calls run in worker threads. Embeddings and every change of the CSM or
the codelet state happen in the phase thread, in run order, so results are
repeatable. Exhausted repairs raise ``ModelFormatError``, and the host
retries the whole cycle.
"""
from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence

from pydantic import BaseModel, Field, create_model, model_validator

from pm.constants import (
    CODELET_OUTPUT_VARIANTS,
    CODELET_PERCEPT_TEMPERATURE,
    CODELET_PERCEPT_TYPES,
    CODELET_SELECTOR_CANDIDATES,
    CODELET_VIEW_TOKENS,
    CODELET_WAVES,
    CSM_SELECTED_MEMORY_ACTIVATION,
)
from pm.context_render.turns import Role, Turn, format_content
from pm.llm.concurrency import ordered_parallel_map
from pm.llm.context_window import estimate_tokens
from pm.llm.prompt_encoding import TurnInput
from pm.records.mental_state import compute_attention_bias
from pm.records.situational_model import SituationalItem, affect_term, cosine
from pm.records.workspace import ShardKey

from workflows.inference_settings import CODELET_WORKERS
from workflows.main_cognition.affect import percept_affect
from workflows.main_cognition.codelet_percepts import (
    HEADER_FIELDS, CodeletPerceptType, container_model, percept_items, percept_text, sample_percept_types, schema_text,
)
from workflows.main_cognition.codelet_registry import (
    CodeletDefinition, activation_score, add_rating, apply_pathway_boost, fill,
    load_definitions, mark_fired, pathway_boosts, ranked, wave_sizes,
)
from workflows.main_cognition.csm_view import Embedder, csm_protected, render_story, story_line, story_tokens
from workflows.main_cognition.cycle_work import CycleWork, provisional_tick
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.appraise import partner_entity
from workflows.main_cognition.phases.model import complete, complete_prose, phase_prompts
from workflows.main_cognition.schemas import (
    ActivationScore, AgentCodeletJob, CodeletOutput, CodeletRatings, CodeletRun, CodeletsResult, CodeletVariant,
    RecordView, VariantPercept,
)
from workflows.main_cognition.story_prefix import StoryPrefix

#: Percept type name of the record that holds a variant recommendation.
RECOMMENDATION = "Recommendation"
#: The view text when the CSM holds nothing that the codelet can see.
EMPTY_VIEW = "(nothing yet)"


class CodeletJob(BaseModel):
    """One scheduled codelet run, prepared in the phase thread.

    ``percept_types`` holds one sampled list of percept types for each
    variant. ``view_ids`` are the CSM records of the view, and ``view`` is
    their story text with ids.
    """

    codelet: str = ""
    mode: str = ""
    wave: int = 0
    score: float = 0.0
    view_ids: list[int] = Field(default_factory=list)
    view: str = ""
    precursor_ids: list[int] = Field(default_factory=list)
    percept_types: list[list[CodeletPerceptType]] = Field(default_factory=list)


class JobResult(BaseModel):
    """The model output of one codelet run, before it changes any state."""

    variants: list[CodeletVariant] = Field(default_factory=list)
    records: list[RecordView] = Field(default_factory=list)
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def tokens(messages: Sequence[TurnInput]) -> int:
    """Return the estimated token count of one call's messages."""
    return estimate_tokens("", messages)[0]


class CodeletFrame:
    """The prompt frame of one cycle: names, shared prefix, and targets.

    The frame uses the shared story prefix of the cycle. Every call of the
    phase starts with :attr:`prefix`, so the bytes stay equal across calls.
    ``targets`` maps each shard label that the conversion call may name to
    its shard key: ``self`` and one label for each stimulus group.
    """

    def __init__(self, work: CycleWork, prompts: PhasePrompts):
        story = StoryPrefix(work)
        self.prompts = prompts
        self.companion = story.companion
        self.entity = story.entity
        self.prefix = story.turns
        #: The prefix as plain text, for agent sessions.
        self.story_text = story.story_text
        self.targets: dict[str, ShardKey] = story.targets

    def fill(self, text: str) -> str:
        """Replace the name placeholders of a definition text."""
        return fill(text, self.companion, self.entity)

    def view_text(self, view: str) -> str:
        """Return the view part of a codelet call."""
        return format_content(self.fill(self.prompts.view_template), view=view).text.strip()

    def job_text(self, definition: CodeletDefinition) -> str:
        """Return the codelet instruction. It comes last in call 1."""
        return format_content(self.fill(self.prompts.job_template), codelet_name=definition.name,
                              description=self.fill(definition.description),
                              instructions=self.fill(definition.prompt).strip()).text.strip()

    def ask(self, definition: CodeletDefinition, job: CodeletJob) -> list[Turn]:
        """Return the turns of call 1: the prefix, then the view and the instruction."""
        return [*self.prefix, Turn(Role.User, f"{self.view_text(job.view)}\n\n{self.job_text(definition)}")]

    def conversion_schema(self, types: list[CodeletPerceptType], allowed_ids: set[int]) -> type[BaseModel]:
        """Return the answer model of call 2 for the offered percept types.

        The answer holds ``target``, ``about_ids``, and one optional field
        for each offered percept type, all at the top level.
        """
        targets = self.targets

        def check(answer):
            if answer.target not in targets:
                raise ValueError(f"target must be one of {', '.join(targets)}.")
            unknown = sorted(set(answer.about_ids) - allowed_ids)
            if unknown:
                raise ValueError(f"about_ids {unknown} are not in the MIND list. Use only ids from that list.")
            return answer

        model = container_model(types, "PerceptConversion", target=(str, "self"), about_ids=(list[int], []))
        return create_model("PerceptConversion", __base__=model,
                            __validators__={"check": model_validator(mode="after")(check)})

    def conversion_text(self, types: list[CodeletPerceptType]) -> str:
        """Return the instruction of call 2."""
        return format_content(self.fill(self.prompts.conversion_template),
                              schema=schema_text(types, self.companion),
                              targets=", ".join(f'"{label}"' for label in self.targets)).text.strip()


def convert(context, work: CycleWork, frame: CodeletFrame, asked: list[Turn], prose: str,
            types: list[CodeletPerceptType], allowed_ids: set[int]) -> tuple[CodeletVariant, int, int]:
    """Convert one recommendation into typed percepts (call 2).

    Port of the second half of ``AgentSimpleCodelet``. The call continues
    the conversation of call 1: the recommendation is the assistant turn.
    Returns the variant and the estimated input and output tokens.
    """
    messages = [*asked, Turn(Role.Assistant, prose), Turn(Role.User, frame.conversion_text(types))]
    answer = complete(context, work, messages, frame.conversion_schema(types, allowed_ids))
    percepts = [VariantPercept(percept_type=kind.value, text=percept_text(kind, percept),
                               details={name: getattr(percept, name) for name in type(percept).model_fields
                                        if name not in HEADER_FIELDS},
                               salience=percept.salience, valence=percept.valence,
                               appraisal_general=percept.general(), appraisal_social=percept.social())
                for kind, percept in percept_items(answer)]
    variant = CodeletVariant(recommendation=prose, percepts=percepts, target_shard=frame.targets[answer.target],
                             about_ids=list(dict.fromkeys(answer.about_ids)))
    return variant, tokens(messages), tokens([Turn(Role.Assistant, answer.model_dump_json())])


def execute(context, work: CycleWork, frame: CodeletFrame, definition: CodeletDefinition, job: CodeletJob) -> JobResult:
    """Run one codelet and return its variants. Runs in a worker thread.

    The function reads ``work`` but never changes it. ``chain`` writes the
    recommendation with one prose call. ``agent`` runs one pm-coder session
    through the ``codelet_agent`` tool. The session may cite records outside
    the view; they become allowed ``about_ids`` and the result returns their
    views.
    """
    started = time.perf_counter()
    result = JobResult()
    asked = frame.ask(definition, job)
    for types in job.percept_types:
        allowed = set(job.view_ids)
        if job.mode == "agent":
            answer = context.runtime.tools["codelet_agent"](context, AgentCodeletJob(
                codelet=definition.name, story=frame.story_text,
                instruction=frame.job_text(definition),
                retrieval_plan=[frame.fill(item) for item in definition.data_aquisation],
                view=frame.view_text(job.view), view_ids=list(job.view_ids), known_ids=list(work.records)))
            prose = answer.recommendation.strip()
            allowed |= set(answer.evidence_ids)
            result.records.extend(answer.records)
            result.calls += 1
        else:
            prose = complete_prose(context, asked)
            result.calls += 1
            result.input_tokens += tokens(asked)
            result.output_tokens += tokens([Turn(Role.Assistant, prose)])
        variant, used, produced = convert(context, work, frame, asked, prose, types, allowed)
        if job.mode == "agent":
            variant.about_ids = list(dict.fromkeys([*variant.about_ids, *answer.evidence_ids]))
        result.variants.append(variant)
        result.calls += 1
        result.input_tokens += used
        result.output_tokens += produced
    result.seconds = round(time.perf_counter() - started, 3)
    return result


def select(context, work: CycleWork, frame: CodeletFrame, candidates: list[CodeletDefinition]) -> CodeletRatings:
    """Ask the selector to rate the candidates for this moment of the story.

    Port of ``AgentSelectCodelets``. A rating for a name outside the
    candidates is invalid output and gets a repair turn. A candidate without
    a rating gets nothing.
    """
    names = {item.name for item in candidates}
    listing = "\n".join(f"- {item.name}: {frame.fill(item.description)}" for item in candidates)
    text = format_content(frame.fill(frame.prompts.selector_template), candidates=listing).text.strip()

    def known(answer):
        unknown = sorted(set(answer.ratings) - names)
        if unknown:
            raise ValueError(f"Unknown mechanism names {', '.join(unknown)}. Use only: {', '.join(sorted(names))}.")
        return answer

    schema = create_model("SelectorRatings", __base__=CodeletRatings,
                          __validators__={"known": model_validator(mode="after")(known)})
    return complete(context, work, [*frame.prefix, Turn(Role.User, text)], schema)


def enter(work: CycleWork, embedder: Embedder, *, record_id: int, codelet: str, content: str, activation: float,
          valence: float, salience: float, shard: ShardKey) -> None:
    """Add one new percept or recommendation record to the CSM under its pending id."""
    tick = provisional_tick(work)
    work.records[record_id] = RecordView(record_id=record_id, kind="CodeletPercept", record_class="CodeletPercept",
                                         source=codelet, content=content, timestamp=work.request.cycle_time)
    work.state.csm.add_or_boost(SituationalItem(
        record_id=record_id, entered_tick=tick, last_boost_tick=tick, activation=activation,
        valence=valence, salience=salience, shard_hint=shard), codelet, embedder.record)


def integrate(work: CycleWork, embedder: Embedder, definition: CodeletDefinition, job: CodeletJob,
              result: JobResult, registered: set[str], codelets: CodeletsResult) -> None:
    """Apply one codelet run to the cycle. Runs in the phase thread, in run order.

    For each variant: mini-appraise each percept, enter the percepts and the
    recommendation into the CSM, and endorse the records that the variant is
    about. Then mark the codelet as fired and apply its pathway boosts.
    """
    state = work.state
    tick = provisional_tick(work)
    bias = compute_attention_bias(state.mental_state)
    entity = partner_entity(work)
    for record in result.records:
        work.records.setdefault(record.record_id, record)
    produced: list[int] = []
    for variant in result.variants:
        for percept in variant.percepts:
            appraisal_state, delta = percept_affect(state.mental_state, percept.appraisal_general,
                                                    percept.appraisal_social, entity_id=entity)
            percept.appraisal_vec = appraisal_state.to_list()
            percept.delta_vec = delta
            percept.record_id = work.allocate_pending_id()
            percept.activation = round(affect_term(percept.valence, percept.salience, bias), 4)
            enter(work, embedder, record_id=percept.record_id, codelet=definition.name, content=percept.text,
                  activation=percept.activation, valence=percept.valence, salience=percept.salience,
                  shard=variant.target_shard)
            produced.append(percept.record_id)
        # The recommendation record carries the mean appraisal state and the
        # summed delta of its percepts, like the old CodeletOutput feature.
        count = len(variant.percepts)
        variant.salience = round(math.fsum(item.salience for item in variant.percepts) / count, 4)
        variant.valence = round(math.fsum(item.valence for item in variant.percepts) / count, 4)
        variant.recommendation_vec = [math.fsum(values) / count for values in zip(*(item.appraisal_vec for item in variant.percepts))]
        variant.recommendation_delta_vec = [math.fsum(values) for values in zip(*(item.delta_vec for item in variant.percepts))]
        variant.recommendation_id = work.allocate_pending_id()
        enter(work, embedder, record_id=variant.recommendation_id, codelet=definition.name,
              content=variant.recommendation, activation=round(affect_term(variant.valence, variant.salience, bias), 4),
              valence=variant.valence, salience=variant.salience, shard=variant.target_shard)
        produced.append(variant.recommendation_id)
        for record_id in variant.about_ids:
            state.csm.add_or_boost(SituationalItem(record_id=record_id, entered_tick=tick, last_boost_tick=tick,
                                                   activation=CSM_SELECTED_MEMORY_ACTIVATION),
                                   definition.name, embedder.record)
    mark_fired(state.codelets.runtime(definition.name), tick, produced)
    for boost in pathway_boosts(definition.name, registered):
        boost.precursor_ids = list(produced)
        apply_pathway_boost(state.codelets, boost)
        codelets.pathway_boosts.append(boost)
    codelets.outputs.append(CodeletOutput(codelet=definition.name, mode=job.mode, wave=job.wave,
                                          variants=result.variants))
    codelets.runs.append(CodeletRun(codelet=definition.name, mode=job.mode, wave=job.wave, reason="scheduled",
                                    score=job.score, view_ids=job.view_ids, precursor_ids=job.precursor_ids,
                                    calls=result.calls, seconds=result.seconds))
    codelets.cost.llm_calls += result.calls
    codelets.cost.input_tokens_estimate += result.input_tokens
    codelets.cost.output_tokens_estimate += result.output_tokens
    codelets.cost.seconds = round(codelets.cost.seconds + result.seconds, 3)


def collect(context, work: CycleWork, prompts: PhasePrompts) -> CodeletsResult:
    """Score, select, and run the codelets of one cycle. Change the CSM and the codelet state."""
    request = work.request
    loaded = load_definitions(request.codelet_definitions or None)
    definitions = loaded.by_name()
    registered = set(definitions)
    frame = CodeletFrame(work, prompts)
    embedder = Embedder(context, work.records)
    rng = random.Random(f"{request.run_id}:codelets")
    tick = provisional_tick(work)
    registry = work.state.codelets
    stimulus_kinds = [item.kind for item in request.stimuli]
    situation = embedder.text(render_story(
        work.records, [*request.recent_dialogue_ids[-4:], *(item.knoxel_id for item in request.stimuli)],
        with_id=False) or "(nothing happens)")
    embedder.prefetch(item.signature_text(frame.companion, frame.entity) for item in definitions.values())
    signatures = {name: embedder.text(item.signature_text(frame.companion, frame.entity))
                  for name, item in definitions.items()}
    codelets = CodeletsResult(definitions_version=loaded.definitions.version, definitions_digest=loaded.digest,
                              budget=work.triage.codelet_budget)

    def score(names: list[str]) -> list[ActivationScore]:
        return ranked([activation_score(definitions[name], registry.runtime(name), tick=tick,
                                        stimulus_kinds=stimulus_kinds, state=work.state.mental_state,
                                        similarity=cosine(signatures[name], situation), rng=rng)
                       for name in names])

    codelets.scores = score(list(definitions))
    candidates = [definitions[item.codelet] for item in codelets.scores[:CODELET_SELECTOR_CANDIDATES]]
    codelets.ratings = select(context, work, frame, candidates).ratings
    codelets.cost.llm_calls += 1
    for name, rating in codelets.ratings.items():
        add_rating(registry.runtime(name), rating)

    ran: list[str] = []
    bias = compute_attention_bias(work.state.mental_state)
    for wave, size in enumerate(wave_sizes(codelets.budget, CODELET_WAVES), start=1):
        waiting = [name for name in definitions if name not in ran]
        chosen = score(waiting)[:size]
        embedder.prefetch_records(work.state.csm.items)
        jobs = []
        for item in chosen:
            definition = definitions[item.codelet]
            precursors = list(registry.runtime(item.codelet).precursor_ids)
            view_ids = work.state.csm.snapshot_for(
                signatures[item.codelet], embedder.record,
                lambda record_id: story_tokens(story_line(work.records[record_id], with_id=True)),
                attention_bias=bias, max_tokens=CODELET_VIEW_TOKENS, rng=rng, precursor_ids=precursors)
            jobs.append(CodeletJob(
                codelet=item.codelet, mode=definition.mode(), wave=wave, score=item.score,
                view_ids=view_ids, view=render_story(work.records, view_ids, with_id=True) or EMPTY_VIEW,
                precursor_ids=precursors,
                percept_types=[sample_percept_types(definition.families, CODELET_PERCEPT_TYPES,
                                                    CODELET_PERCEPT_TEMPERATURE, rng)
                               for _ in range(CODELET_OUTPUT_VARIANTS)]))
        results = ordered_parallel_map(
            lambda job: execute(context, work, frame, definitions[job.codelet], job), jobs,
            workers=CODELET_WORKERS, cancel=context.runtime.stop_event)
        for job, result in zip(jobs, results, strict=True):
            integrate(work, embedder, definitions[job.codelet], job, result, registered, codelets)
            ran.append(job.codelet)
    work.state.csm.prune(csm_protected(work))
    return codelets


def run(context, work: CycleWork) -> CycleWork:
    result = collect(context, work, phase_prompts("codelets"))
    work.codelets = result
    percepts = sum(len(variant.percepts) for output in result.outputs for variant in output.variants)
    waves = max((run.wave for run in result.runs), default=0)
    work.receipt("codelets", f"ran {len(result.runs)} of budget {result.budget} codelets in {waves} waves "
                             f"({', '.join(run.codelet for run in result.runs) or 'none'}), {percepts} percepts, "
                             f"{result.cost.llm_calls} model calls")
    return work


__all__ = [
    "CodeletFrame",
    "CodeletJob",
    "EMPTY_VIEW",
    "JobResult",
    "RECOMMENDATION",
    "collect",
    "convert",
    "enter",
    "execute",
    "integrate",
    "run",
    "select",
]
