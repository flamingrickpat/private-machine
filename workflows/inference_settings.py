"""Shared inference constants for companion workflows.

Fractions apply after input and visible-answer reserves. Ceilings prevent a
large serving context from creating a large private reasoning allowance.
Changing these values requires new real evidence and conversation review.
"""
from types import MappingProxyType

from pm.llm.reasoning import ReasoningBudget

CODELET_WORKERS = 2
ATTENTION_WORKERS = 2
CHAT_CONCURRENCY = 4

REASONING = MappingProxyType({
    "off": ReasoningBudget(fraction=0),
    "notice": ReasoningBudget(fraction=0.02, ceiling_tokens=512),
    "remember": ReasoningBudget(fraction=0.03, ceiling_tokens=768),
    "interpret": ReasoningBudget(fraction=0.03, ceiling_tokens=768),
    "imagine": ReasoningBudget(fraction=0.04, ceiling_tokens=1024),
    "evaluate": ReasoningBudget(fraction=0.05, ceiling_tokens=1024),
    "inspect": ReasoningBudget(fraction=0.05, ceiling_tokens=1024),
    "plan": ReasoningBudget(fraction=0.05, ceiling_tokens=1536, answer_reserve_tokens=4096),
})
