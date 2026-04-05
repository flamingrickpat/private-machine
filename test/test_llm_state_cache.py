import pytest


pytestmark = pytest.mark.skip(reason="Manual llama.cpp cache experiment kept out of automated pytest collection.")


def test_llm_state_cache_manual_placeholder() -> None:
    pass
