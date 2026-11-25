"""Model access for the phases of the main cognition cycle.

Every phase calls the model through :func:`complete`. It uses the
``llm_complete`` tool that the host grants to the run, and the repair loop of
:func:`workflows.main_cognition.model_json.complete_json`. The model tool
writes each attempt to the configured prompt log.

Some phases read private grounding context: the public-context snapshot that
the host froze for this cycle. :func:`complete` inserts it as the second turn
of every attempt, or after the shared story prefix in a story-framed phase.
The context is data, so it stays outside the instruction turn, and the
current phase input stays last and authoritative.
"""
from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path
from typing import TypeVar

from pm_workflows.python_role import RoleContext
from pydantic import BaseModel

from pm.constants import Sampler
from pm.context_render.turns import Role, Turn
from pm.llm.prompt_encoding import TurnInput, normalize_turns

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.prompt_config import PhasePrompts, load_prompts
from pm.llm.structured import complete_json
from pm.llm.generation import GenerationSettings
from pm.llm.role_model import role_model
from pm.context_render.prompt import require_rendered_prompt

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

Result = TypeVar("Result", bound=BaseModel)


def phase_prompts(name: str) -> PhasePrompts:
    """Load the prompt file of one phase from ``phases/prompts/<name>.yaml``."""
    return load_prompts(PROMPTS_DIR / f"{name}.yaml")


def grounding_turn(work: CycleWork, *, emotion: bool, capabilities: bool = False) -> Turn | None:
    """Return the private grounding turn for one phase, or None.

    The result is None when the host supplied no public context. ``emotion``
    selects the emotion section. Phases after appraisal read the accepted
    affect in their own input, so they pass False. ``capabilities`` keeps the
    capability section and the capability note. Only the route phase passes
    True: capability limits belong to routing (plan section "Prompt frame").
    Evidence rows that repeat a current stimulus word for word are dropped.
    When evidence remains, the memory list is dropped, because the evidence
    already holds the sources.
    """
    public = work.request.public_context
    if public is None:
        return None
    lines = ["Private accepted state. Private thoughts remain private."]
    if emotion:
        lines.append("Feeling: " + public.feelings)
    if capabilities:
        lines.append("Actual available capabilities: " + "; ".join(work.request.capabilities))
    return Turn(Role.User, "\n".join(lines))


def complete(context: RoleContext, messages: Sequence[TurnInput], schema: type[Result], *,
             validation_context: BaseModel | None = None) -> Result:
    """Call the model for one phase and return the validated answer.

    ``grounded`` inserts the private grounding turn at index ``grounding_at``
    of each attempt. A story-framed phase passes the length of the shared
    prefix, so the prefix bytes stay equal and the provider can reuse the
    cache. ``emotion`` and ``capabilities`` are passed to :func:`grounding_turn`.
    ``validation_context`` is passed to the schema validation, so a rule that
    needs caller data, such as the ids a phase actually offered, can reject an
    answer inside the repair loop. Invalid output gets repair turns (see
    ``complete_json``). The loop has no attempt limit: it stops at the first
    answer that fits, and every rejection is logged with its attempt number.
    ``cancel`` comes from the model session, so stopping the run ends the loop
    between attempts. ``ModelFormatError`` covers the format failures the loop
    itself reports. Transport errors and cancellation propagate.
    """
    turns = normalize_turns(messages)
    session = role_model(context.runtime)
    return complete_json(session, turns, schema, validation_context=validation_context, cancel=session.cancel,
                         settings=GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False))



def complete_prose(context: RoleContext, messages: Sequence[TurnInput]) -> str:
    """Call the model for prose and return the non-empty answer.

    The codelet frame supplies the rendered story and view. Check their
    combined token budget before asking the model for a recommendation.
    """
    assert context.runtime is not None
    session = role_model(context.runtime)
    turns = tuple(normalize_turns(messages))
    require_rendered_prompt("main cognition prose", turns, session.render_settings())
    attempt = 0
    while True:
        attempt += 1
        session.check_cancelled()
        raw = session.complete(turns, settings=GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False))
        if raw.strip():
            return raw.strip()
        context.runtime.environment.write_text(f"scratch/model-output-empty-{attempt:02}.txt", raw)
        logging.getLogger(__name__).warning("Empty cognition prose, resampling attempt=%s", attempt)



__all__ = ["PROMPTS_DIR", "complete", "complete_prose", "grounding_turn", "phase_prompts"]
