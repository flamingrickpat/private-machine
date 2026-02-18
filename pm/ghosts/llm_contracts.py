import logging
from time import perf_counter
from typing import List, Optional, Sequence, Tuple, Type, TypeVar

from pydantic import BaseModel

from pm.ghosts.procedures.base import GhostProtocol
from pm.llm.llm_common import CommonCompSettings, LlmPreset

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Message = Tuple[str, str]


def _append_tool_quality_event(ghost: GhostProtocol, event: dict) -> None:
    if not hasattr(ghost, "llm_tool_quality_history"):
        ghost.llm_tool_quality_history = []
    ghost.llm_tool_quality_history.append(event)
    if len(ghost.llm_tool_quality_history) > 200:
        ghost.llm_tool_quality_history = ghost.llm_tool_quality_history[-200:]

    if not hasattr(ghost, "llm_tool_quality_last"):
        ghost.llm_tool_quality_last = {}
    ghost.llm_tool_quality_last[event["phase"]] = event


def _build_messages(system_prompt: str, user_prompt: str, examples: Optional[Sequence[Message]]) -> List[Message]:
    msgs: List[Message] = [("system", system_prompt)]
    if examples:
        msgs.extend(list(examples))
    msgs.append(("user", user_prompt))
    return msgs


def call_tool_with_contract(
    ghost: GhostProtocol,
    *,
    phase: str,
    schema: Type[T],
    system_prompt: str,
    user_prompt: str,
    examples: Optional[Sequence[Message]] = None,
    preset: LlmPreset = LlmPreset.Default,
    max_retries: int = 1,
    max_output_tokens: Optional[int] = None,
) -> tuple[Optional[T], dict]:
    """
    Robust schema call wrapper:
    - supports user/assistant few-shot examples
    - retries with stricter reminder for small local models
    - stores per-phase quality telemetry on ghost
    """
    start = perf_counter()
    attempts = 0
    fallback_reason = ""
    parsed = False
    model: Optional[T] = None
    raw_text = ""

    reminder = (
        "Return exactly one valid JSON object that conforms to the tool schema. "
        "Do not add prose, markdown, or extra keys."
    )

    msgs = _build_messages(system_prompt, user_prompt, examples)
    while attempts <= max_retries and model is None:
        attempts += 1
        use_msgs = list(msgs)
        if attempts > 1:
            use_msgs.append(("user", reminder))

        try:
            comp_settings = None
            if max_output_tokens is not None:
                comp_settings = CommonCompSettings(max_tokens=int(max_output_tokens))
            try:
                raw_text, calls = ghost.llm.completion_tool(
                    preset=preset,
                    inp=use_msgs,
                    tools=[schema],
                    comp_settings=comp_settings,
                )
            except TypeError:
                # Test doubles / legacy adapters may not support comp_settings.
                raw_text, calls = ghost.llm.completion_tool(
                    preset=preset,
                    inp=use_msgs,
                    tools=[schema],
                )
            if not calls:
                fallback_reason = "no_tool_calls"
                continue

            first = calls[0]
            if isinstance(first, schema):
                model = first
                parsed = True
                break

            try:
                model = schema.model_validate(first)
                parsed = True
                break
            except Exception:
                fallback_reason = f"type_mismatch:{type(first).__name__}"
        except Exception as e:
            fallback_reason = f"exception:{type(e).__name__}"
            logger.warning("Tool contract call failed phase=%s attempt=%d: %s", phase, attempts, e)

    duration_ms = round((perf_counter() - start) * 1000.0, 3)
    event = {
        "phase": phase,
        "schema": schema.__name__,
        "ok": model is not None,
        "attempts": attempts,
        "parsed": parsed,
        "fallback_reason": fallback_reason,
        "duration_ms": duration_ms,
        "example_turns": len(list(examples or [])),
        "raw_excerpt": (raw_text or "")[:160],
        "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
    }
    _append_tool_quality_event(ghost, event)
    return model, event
