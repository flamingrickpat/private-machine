from enum import Enum

import pytest
from pydantic import BaseModel, Field

from pm.utils.gbnf_utils import (
    better_generate_gbnf_grammar_and_documentation,
    fix_gbnf_grammar_generator,
)
from llama_cpp import LlamaGrammar


class NumericListModel(BaseModel):
    items: list[int] = Field(max_length=3)
    default_items: list[str] = Field(default_factory=list)
    timeout: int = Field(ge=5, le=7)
    even_bucket: int = Field(ge=0, le=10, multiple_of=2)
    score: float = Field(ge=0.0, le=1.0)
    signed: float = Field(ge=-1.0, le=1.0)
    maybe_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ToolAlpha(BaseModel):
    query: str
    top_k: int = Field(ge=1, le=5)


class ToolBeta(BaseModel):
    query: str
    threshold: float = Field(ge=0.0, le=1.0)


class ToolRouter(BaseModel):
    tool: ToolAlpha | ToolBeta


class Priority(Enum):
    LOW = "low"
    HIGH = "high"


class EnumAndDictModel(BaseModel):
    priority: Priority
    counters: dict[str, int]


def _compile_grammar(models: list[type[BaseModel]], default_max_list_length: int = 8) -> str:
    fix_gbnf_grammar_generator()
    gbnf_grammar, _ = better_generate_gbnf_grammar_and_documentation(
        models, default_max_list_length=default_max_list_length
    )
    assert isinstance(gbnf_grammar, str) and gbnf_grammar.strip()
    # Compile only; no text generation.
    res = LlamaGrammar(_grammar=gbnf_grammar)
    assert res is not None
    return gbnf_grammar


def test_numeric_and_list_constraints_compile_to_valid_llama_grammar():
    grammar = _compile_grammar([NumericListModel], default_max_list_length=8)

    # max_length on list field
    assert 'numeric-list-model-items ::= "[" ws integer ("," ws integer){0,3}  "]"' in grammar
    # default list cap fallback
    assert 'numeric-list-model-default-items ::= "[" ws string ("," ws string){0,8}  "]"' in grammar
    # bounded int expansion
    assert 'numeric-list-model-timeout-number ::= "5" | "6" | "7"' in grammar
    # bounded + multiple_of expansion
    assert 'numeric-list-model-even-bucket-number ::= "0" | "2" | "4" | "6" | "8" | "10"' in grammar
    # common bounded float patterns
    assert "numeric-list-model-score-number ::= (" in grammar
    assert "numeric-list-model-signed-number ::= (" in grammar
    assert "numeric-list-model-maybe-score-number ::= (" in grammar


def test_union_models_compile_to_valid_llama_grammar():
    grammar = _compile_grammar([ToolRouter])
    assert (
        "tool-router-tool-union ::= tool-alpha | tool-beta" in grammar
        or "tool-router-tool-union ::= tool-beta | tool-alpha" in grammar
    )
    assert 'tool-alpha-top-k-number ::= "1" | "2" | "3" | "4" | "5"' in grammar
    assert "tool-beta-threshold-number ::= (" in grammar


def test_enum_and_dict_constructs_compile_to_valid_llama_grammar():
    grammar = _compile_grammar([EnumAndDictModel])
    assert "enum-and-dict-model-priority ::= " in grammar
    assert "custom-dict-key-type-string-value-type-integer" in grammar
