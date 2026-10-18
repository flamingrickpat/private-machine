"""Model access for the phases of the main cognition cycle.

Every phase calls the model through :func:`complete`. It uses the
``llm_complete`` tool that the host grants to the run, and the repair loop of
:func:`workflows.main_cognition.model_json.complete_json`. The model tool
writes each attempt to the prompt log (``PM_PROMPT_LOG_DIR``).

Some phases read private grounding context: the public-context snapshot that
the host froze for this cycle. :func:`complete` inserts it as the second turn
of every attempt, or after the shared story prefix in a story-framed phase.
The context is data, so it stays outside the instruction turn, and the
current phase input stays last and authoritative.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from pm.constants import CODELET_TEXT_ATTEMPTS, Sampler
from pm.context_render import ContextKind, render_context
from pm.context_render.turns import Role, Turn
from pm.llm.prompt_encoding import TurnInput

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts, complete_json, complete_text, load_prompts

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

Result = TypeVar("Result", bound=BaseModel)


class EvidenceRow(BaseModel):
    """The two fields of one public-context evidence row that grounding reads."""

    knoxel_id: int | None = None
    text: str = ""


def phase_prompts(name: str) -> PhasePrompts:
    """Load the prompt file of one phase from ``phases/prompts/<name>.yaml``."""
    return load_prompts(PROMPTS_DIR / f"{name}.yaml")


def grounding_turn(work: CycleWork, *, emotion: bool) -> Turn | None:
    """Return the private grounding turn for one phase, or None.

    The result is None when the host supplied no public context. ``emotion``
    selects the emotion section. Phases after appraisal read the accepted
    affect in their own input, so they pass False. Evidence rows that repeat a
    current stimulus word for word are dropped. When evidence remains, the
    memory list is dropped, because the evidence already holds the sources.
    """
    public = work.request.public_context
    if public is None:
        return None
    kinds = ContextKind.All if emotion else ContextKind.All & ~ContextKind.Emotion
    stimuli = {item.knoxel_id: item.text for item in work.request.stimuli}
    evidence = []
    for row in public.evidence:
        parsed = EvidenceRow.model_validate(row)
        if parsed.knoxel_id not in stimuli or parsed.text != stimuli[parsed.knoxel_id]:
            evidence.append(row)
    selected = public.model_copy(update={"evidence": evidence,
                                         "memories": [] if public.evidence else public.memories})
    grounding = "Private context. Respect domain capabilities and source attribution. "
    grounding += "Private thoughts remain private. An emotion word cannot create an unimplemented state or sense.\n"
    return Turn(Role.User, grounding + render_context(selected, kinds))


def complete(context, work: CycleWork, messages: Sequence[TurnInput], schema: type[Result], *,
             grounded: bool = False, emotion: bool = True, grounding_at: int = 1) -> Result:
    """Call the model for one phase and return the validated answer.

    ``grounded`` inserts the private grounding turn at index ``grounding_at``
    of each attempt. A story-framed phase passes the length of the shared
    prefix, so the prefix bytes stay equal and the provider can reuse the
    cache. ``emotion`` is passed to :func:`grounding_turn`. Invalid output
    gets repair turns (see ``complete_json``). Exhausted repairs raise
    ``ModelFormatError``. Transport errors and cancellation propagate.
    """
    turn = grounding_turn(work, emotion=emotion) if grounded else None

    def insert(turns: list[Turn]) -> list[Turn]:
        if turn is not None:
            turns.insert(grounding_at, turn)
        return turns

    return complete_json(context, messages, schema, transform=insert,
                         temperature=Sampler.TempInstruct, enable_thinking=False)


def complete_prose(context, messages: Sequence[TurnInput]) -> str:
    """Call the model for prose and return the non-empty answer.

    The settings match :func:`complete`. Codelet calls get no grounding turn,
    so all codelet calls of one cycle share their prefix byte for byte. See
    ``complete_text`` for retries and errors.
    """
    return complete_text(context, messages, attempts=CODELET_TEXT_ATTEMPTS,
                         temperature=Sampler.TempInstruct, enable_thinking=False)


__all__ = ["EvidenceRow", "PROMPTS_DIR", "complete", "complete_prose", "grounding_turn", "phase_prompts"]
