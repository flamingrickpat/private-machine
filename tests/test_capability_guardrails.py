from types import SimpleNamespace

from pm.ghosts.capabilities import (
    capability_context_text,
    enforce_behavior_capability,
    resolve_capability_question,
)
from pm.ghosts.schemas import BehaviorOutput


def _ghost():
    return SimpleNamespace(
        config=SimpleNamespace(
            supported_capabilities=[
                "I can communicate with you via text in this chat.",
                "I can reason about my internal state and report simulated emotions.",
            ],
            unsupported_capabilities=[
                "I cannot create or host a literal virtual reality world you can physically join.",
                "I cannot directly act in the physical world.",
            ],
            capability_notes=[
                "Never claim capabilities beyond the current implementation.",
            ],
        )
    )


def test_capability_context_contains_supported_and_unsupported():
    txt = capability_context_text(_ghost())
    assert "Supported capabilities:" in txt
    assert "Unsupported capabilities:" in txt


def test_enforce_behavior_rewrites_impossible_action():
    ghost = _ghost()
    behavior = BehaviorOutput(
        action_description="Create a shared VR world and invite the user into it now.",
        speech="I'll open a VR world now.",
        internal_thought="Do it directly.",
        tool_call=None,
    )
    fixed, guard = enforce_behavior_capability(
        ghost,
        behavior,
        user_request="Please make a VR world for us.",
    )
    assert guard is not None
    assert guard["reason"] in {"unsupported_action", "capability_question"}
    assert "can't" in (fixed.speech or "").lower()
    assert "text" in (fixed.speech or "").lower()


def test_capability_question_resolution_yes_no():
    ghost = _ghost()
    yes = resolve_capability_question("Can you feel emotions?", ghost)
    no = resolve_capability_question("Can you create a virtual reality world I can join?", ghost)
    assert yes is not None and "yes" in yes.lower()
    assert no is not None and "can't" in no.lower()
