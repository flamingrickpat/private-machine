from types import SimpleNamespace

import pm.config_loader as cfg
from pm.ghosts.agent_context import compute_agent_context_budget
from pm.llm.llm_common import LlmPreset


class _DummyLlm:
    def get_max_tokens(self, preset):
        assert preset == LlmPreset.Default
        return int(cfg.model_map["Default"]["context"])


def test_multi_context_budget_uses_unit_config_limits():
    ghost = SimpleNamespace(
        llm=_DummyLlm(),
        config=SimpleNamespace(
            agent_context_target_ratio=float(cfg.agent_context_target_ratio),
            agent_context_safety_margin_tokens=int(cfg.agent_context_safety_margin_tokens),
        ),
    )

    out_tokens = 320
    budget = compute_agent_context_budget(
        ghost,
        output_tokens=out_tokens,
        min_budget=320,
    )

    model_ctx = int(cfg.model_map["Default"]["context"])
    hard_cap = model_ctx - out_tokens - int(cfg.agent_context_safety_margin_tokens)
    ratio_cap = int(model_ctx * float(cfg.agent_context_target_ratio))

    assert budget > 0
    assert budget <= hard_cap
    assert budget <= ratio_cap
