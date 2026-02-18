from types import SimpleNamespace

from pm.ghosts.llm_contracts import call_tool_with_contract
from pm.ghosts.schemas import IntentionMatchResult


class _RetryLlm:
    def __init__(self):
        self.calls = 0

    def completion_tool(self, preset, inp, tools):
        self.calls += 1
        if self.calls == 1:
            return "", []
        return "", [IntentionMatchResult(is_satisfied=True, confidence=0.9, explanation="ok")]


def test_call_tool_with_contract_retries_and_records_quality():
    ghost = SimpleNamespace(current_tick_id=11, llm=_RetryLlm())
    model, event = call_tool_with_contract(
        ghost,
        phase="test_phase",
        schema=IntentionMatchResult,
        system_prompt="sys",
        user_prompt="user",
        examples=[("user", "x"), ("assistant", '{"is_satisfied":false,"confidence":0.1,"explanation":"x"}')],
        max_retries=1,
    )

    assert model is not None
    assert model.is_satisfied is True
    assert event["attempts"] == 2
    assert event["ok"] is True
    assert ghost.llm_tool_quality_history
    assert ghost.llm_tool_quality_last["test_phase"]["ok"] is True
