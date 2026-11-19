from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict


class PublicOntologyTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["term"] = "term"
    record_id: int
    term: str
    term_kind: str
    definition: str
    usage_rule: str


class PublicOntologyBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["binding"] = "binding"
    record_id: int
    term_id: int
    fit_score: float
    explanation: str


type PublicOntologyEntry = PublicOntologyTerm | PublicOntologyBinding
