import pytest


pytestmark = pytest.mark.skip(reason="Requires a persisted local memory database and live LLM wiring.")


def test_memquery_manual_placeholder() -> None:
    pass
