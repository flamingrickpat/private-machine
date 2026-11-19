from __future__ import annotations

from typing import Self
from pydantic import BaseModel, ValidationInfo, model_validator

from pm.constants import STRUCTURE_BINDER_MAX
from workflows.main_cognition.schemas import BinderAnswer, CodeletRatings, GoalsResult, StructureRatings


class StructureRatingCheck(BaseModel):
    labels: tuple[str, ...]


class CheckedStructureRatings(StructureRatings):
    @model_validator(mode="after")
    def check_labels(self, info: ValidationInfo) -> Self:
        check = info.context
        if not isinstance(check, StructureRatingCheck):
            raise TypeError("Attention ratings require StructureRatingCheck.")
        if sorted(item.structure for item in self.ratings) != sorted(check.labels):
            raise ValueError(f"Rate each label exactly once: {', '.join(check.labels)}.")
        return self


class BinderCheck(BaseModel):
    allowed_ids: frozenset[int]


class CheckedBinderAnswer(BinderAnswer):
    @model_validator(mode="after")
    def check_members(self, info: ValidationInfo) -> Self:
        check = info.context
        if not isinstance(check, BinderCheck):
            raise TypeError("Structure binding requires BinderCheck.")
        if len(self.structures) > STRUCTURE_BINDER_MAX:
            raise ValueError(f"Write at most {STRUCTURE_BINDER_MAX} structures.")
        for structure in self.structures:
            unknown = sorted(set(structure.record_ids) - check.allowed_ids)
            if unknown:
                raise ValueError(f"record_ids {unknown} are not in the MIND list. Use only ids from that list.")
            for relation in structure.relations:
                if {relation.source_id, relation.target_id} - set(structure.record_ids):
                    raise ValueError("A relation can link only ids of its own structure's record_ids.")
        return self


class CodeletRatingCheck(BaseModel):
    names: frozenset[str]


class CheckedCodeletRatings(CodeletRatings):
    @model_validator(mode="after")
    def check_names(self, info: ValidationInfo) -> Self:
        check = info.context
        if not isinstance(check, CodeletRatingCheck):
            raise TypeError("Codelet ratings require CodeletRatingCheck.")
        unknown = sorted(set(self.ratings) - check.names)
        if unknown:
            raise ValueError(f"Unknown mechanism names {', '.join(unknown)}. Use only: {', '.join(sorted(check.names))}.")
        return self


class CheckedGoals(GoalsResult):
    @model_validator(mode="after")
    def check_changes(self) -> Self:
        allowed = ("retain", "add", "complete", "reprioritize", "defer", "refuse")
        for change in self.changes:
            if change.change not in allowed:
                raise ValueError(f"Goal change {change.change!r} is not allowed. Use one of: {', '.join(allowed)}.")
        return self
