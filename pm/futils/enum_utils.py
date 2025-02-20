from enum import Enum
from typing import List, Type, TypeVar

T = TypeVar("T", bound=Enum)

def make_enum_subset(enum_type: Type[T], allowed_values: List[T]) -> Type[T]:
    """Create a new Enum subclass with only the allowed values."""
    subset_name = f"{enum_type.__name__}Subset"
    subset_dict = {e.name: e.value for e in allowed_values}
    return Enum(subset_name, subset_dict)

def make_enum_from_list(items: List[str]) -> Type[T]:
    subset_name = f"TempEnum"
    subset_dict = {e: e for e in items}
    return Enum(subset_name, subset_dict)
