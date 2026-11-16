"""Bridge serving measurements to the context renderer's complete-prompt selector.

The renderer owns selection. This adapter only applies connection configuration
and reserves wire-schema tokens that are absent from the rendered turns.
"""
from collections.abc import Sequence
from pm.context_render.budget import ContextBudget, PromptSelection, TokenEstimator, fit_turns
from pm.context_render.turns import Turn
from pm.llm.contracts import ContextMeasurement


def prompt_budget(capacity: ContextMeasurement, *, internal: bool = True,
                  rendered: Sequence[Turn] = ()) -> ContextBudget:
    """Build a budget from an empty-prompt measurement, including protocol schemas."""
    window = capacity["context_window"]
    if internal:
        window = min(window, capacity["cognition_context_window"])
    protocol_tokens = max(0, capacity["input_tokens"] - TokenEstimator().estimate(rendered))
    return ContextBudget(context_window=window, input_fraction=capacity["input_fraction"],
                         reserved_tokens=protocol_tokens)


def fit_measured_prompt(turns: Sequence[Turn], capacity: ContextMeasurement, *, internal: bool = False) -> PromptSelection:
    """Fit measured complete turns, retaining the measured protocol overhead."""
    return fit_turns(turns, prompt_budget(capacity, internal=internal, rendered=turns))
