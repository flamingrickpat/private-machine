import pytest


pytestmark = pytest.mark.skip(reason="Requires an external LLM runtime and is not part of the default automated suite.")


def test_llm_completion_manual_placeholder() -> None:
    pass
