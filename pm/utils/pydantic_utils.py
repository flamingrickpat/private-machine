import random
from copy import deepcopy
from datetime import timedelta
from enum import Enum
from typing import (
    Union,
    get_origin,
    Optional,
)
from typing import Type
from typing import Dict, Any
from datetime import datetime
from typing import get_args
from typing import List

from pydantic import BaseModel, Field, create_model


def _stringify_type(tp: Any) -> str:
    """
    Turn a typing object (e.g. List[int], Optional[str]) into a string.
    """
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is None:
        # a plain type
        return tp.__name__ if hasattr(tp, "__name__") else str(tp)
    # handle Optional[T] (which is Union[T, NoneType])
    if origin is Union and len(args) == 2 and type(None) in args:
        inner = args[0] if args[1] is type(None) else args[1]
        return f"Optional[{_stringify_type(inner)}]"
    # handle other Unions
    if origin is Union:
        return "Union[" + ", ".join(_stringify_type(a) for a in args) + "]"
    # handles List, Set, Tuple, etc.
    name = origin.__name__ if hasattr(origin, "__name__") else str(origin)
    if args:
        return f"{name}[{', '.join(_stringify_type(a) for a in args)}]"
    return name


def pydantic_to_type_dict(
        model: Union[Type[BaseModel], BaseModel]
) -> Dict[str, str]:
    """
    Given a Pydantic BaseModel class or instance, returns a dict
    { field_name: type_string } for each declared field.
    """
    # figure out the class
    model_cls = model if isinstance(model, type) else model.__class__
    if not issubclass(model_cls, BaseModel):
        raise TypeError("model must be a Pydantic BaseModel class or instance")

    out: Dict[str, str] = {}
    for name, field in model_cls.model_fields.items():
        # `outer_type_` preserves e.g. Optional[int] vs just int
        typ = field.outer_type_
        out[name] = _stringify_type(typ)
    return out

def create_basemodel(field_definitions, randomnize_entries=False):
    """
    Create a dynamic Pydantic BaseModel with randomized field order.

    Args:
        field_definitions (list): A list of dictionaries where each dict should have:
            - 'name': the field name (str)
            - 'type': the field type (e.g. str, int, etc.)
            - 'description': a description for the field (str)
        randomnize_entries (bool): If True, each field gets a random default value
            based on its type. Otherwise, the field is required.

    Returns:
        A dynamically created Pydantic BaseModel.
    """
    # Make a deep copy and randomize the order
    randomized_fields = deepcopy(field_definitions)
    if randomnize_entries:
        random.shuffle(randomized_fields)

    fields = {}
    for field in randomized_fields:
        name = field['name']
        f_type = field['type']
        description = field.get('description', "")

        # Use Ellipsis to indicate the field is required.
        fields[name] = (f_type, Field(..., description=description))

    # Create and return a dynamic model class
    return create_model('DynamicBaseModel', **fields)

_sample_words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "lambda"]

def _random_string(model_name: str, field_name: str) -> str:
    word = random.choice(_sample_words)
    num = random.randint(1, 999)
    return f"{model_name}_{field_name}_{word}_{num}"


def _random_int(field_info: Field) -> int:
    ge = -1000
    le = 1000
    for meta in field_info.metadata:
        if hasattr(meta, "ge"): ge = meta.ge
        if hasattr(meta, "le"): le = meta.le

    lo = ge if ge is not None else 0
    hi = le if le is not None else lo + 100
    return random.randint(int(lo), int(hi))


def _random_float(field_info: Field) -> float:
    ge = -float("inf")
    le = float("inf")
    for meta in field_info.metadata:
        if hasattr(meta, "ge"): ge = meta.ge
        if hasattr(meta, "le"): le = meta.le

    lo = ge if ge is not None else 0.0
    hi = le if le is not None else lo + 1.0
    return random.uniform(lo, hi)


def _random_datetime() -> datetime:
    # within ±1 day of now
    delta = timedelta(seconds=random.randint(-86_400, 86_400))
    return datetime.now() + delta


def create_random_pydantic_instance(t: Type[BaseModel]) -> BaseModel:
    """
    Instantiate and return a t(...) filled with random/test data,
    respecting any Field(..., ge=..., le=...) constraints.
    """
    values = {}
    model_name = t.__name__
    for name, field in t.__class__.model_fields.items():
        info = field
        ftype = field.annotation

        # If there's a default factory, just call it
        if info.default_factory is not None:
            values[name] = info.default_factory()
            continue

        # Handle Optional[...] by unwrapping
        origin = get_origin(ftype)
        if origin is Optional:
            (subtype,) = get_args(ftype)
            ftype = subtype
            origin = get_origin(ftype)

        # Primitives
        if ftype is int:
            values[name] = _random_int(info)
        elif ftype is float:
            values[name] = _random_float(info)
        elif ftype is bool:
            values[name] = random.choice([True, False])
        elif ftype is str:
            values[name] = _random_string(model_name, name)
        elif ftype is datetime:
            values[name] = _random_datetime()
        # Enums
        elif isinstance(ftype, type) and issubclass(ftype, Enum):
            values[name] = random.choice(list(ftype))
        # Lists
        elif origin in (list, List):
            (subtype,) = get_args(ftype)
            length = random.randint(1, 3)
            # e.g. List[float] or List[int] or List[SubModel]
            evs = []
            for _ in range(length):
                # simple primitives
                if subtype is int:
                    evs.append(_random_int(info))
                elif subtype is float:
                    evs.append(_random_float(info))
                elif subtype is str:
                    evs.append(_random_string(model_name, name))
                # nested BaseModel
                elif isinstance(subtype, type) and issubclass(subtype, BaseModel):
                    evs.append(create_random_pydantic_instance(subtype))
                else:
                    evs.append(None)
            values[name] = evs
        # Nested BaseModel
        elif isinstance(ftype, type) and issubclass(ftype, BaseModel):
            values[name] = create_random_pydantic_instance(ftype)
        else:
            # fallback to default or None
            values[name] = info.default if info.default is not None else None

    return t(**values)
