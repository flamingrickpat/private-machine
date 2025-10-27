from __future__ import annotations

import copy
import json
import random
from copy import deepcopy
from datetime import timedelta, date
from enum import Enum
from typing import (
    Union,
    get_origin,
    Optional, OrderedDict, get_type_hints, Callable, TypeVar,
)
from typing import Type
from typing import Dict, Any
from datetime import datetime
from typing import get_args
from typing import List
from itertools import groupby

from pydantic import BaseModel, Field, create_model

primitives = (bool, str, int, float, type(None))
T = TypeVar("T")

def is_primitive(obj):
    return isinstance(obj, primitives)

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

def create_basemodel(field_definitions, randomnize_entries=False, optional: bool = False):
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
        if optional:
            f_type = Optional[field['type']]
        else:
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

def create_delta_basemodel(
    original_basemodel: type[BaseModel],
    new_ge: float | int,
    new_le: float | int,
    add_reason: bool = True,
    reason_docstring: str = "Natural-language rationale for how/why these deltas were chosen.",
    shuffle_numeric_fields: bool = False,
) -> type[BaseModel]:
    """
    Create a 'delta' Pydantic model from an existing BaseModel by:
      - selecting only numeric fields (int/float),
      - setting their ge/le constraints to (new_ge, new_le),
      - copying per-field descriptions,
      - optionally adding a leading 'reason: str' field,
      - copying the original model's class docstring.

    Parameters
    ----------
    original_basemodel : type[BaseModel]
        The source Pydantic model class.
    new_ge : float | int
        Lower bound to apply to all numeric fields.
    new_le : float | int
        Upper bound to apply to all numeric fields.
    add_reason : bool, default True
        If True, add a leading 'reason: str' field.
    reason_docstring : str, default ...
        Description to attach to the 'reason' field (if added).
    shuffle_numeric_fields : bool, default False
        If True, randomize the order of numeric fields (reason stays first if present).

    Returns
    -------
    type[BaseModel]
        A new Pydantic model class named '<OriginalName>Delta'.
    """

    def _is_numeric_annotation(tp: Any) -> bool:
        """Return True if the annotation is (or includes) int/float."""
        origin = get_origin(tp)
        if origin is None:
            return tp in (int, float)
        if origin is Union:
            return any(_is_numeric_annotation(arg) for arg in get_args(tp))
        if str(origin) == "typing.Annotated":
            # Annotated[base, ...] -> check the base type
            args = get_args(tp)
            return _is_numeric_annotation(args[0]) if args else False
        return False

    def _base_numeric_type(tp: Any) -> type[int] | type[float] | None:
        """Return the concrete numeric type to use (float preferred if present)."""
        origin = get_origin(tp)
        if origin is None:
            if tp in (int, float):
                return tp
            return None
        if origin is Union:
            args = get_args(tp)
            # Prefer float if present, else int
            if any(a is float for a in args):
                return float
            if any(a is int for a in args):
                return int
            # handle Optional[float] etc. via recursion fallback
            for a in args:
                bt = _base_numeric_type(a)
                if bt in (float, int):
                    return bt
            return None
        if str(origin) == "typing.Annotated":
            args = get_args(tp)
            return _base_numeric_type(args[0]) if args else None
        return None

    # Collect numeric fields from the original model, preserving original order
    numeric_field_names: list[str] = []
    field_specs: list[tuple[str, tuple[type[Any], Any]]] = []

    for name, f in original_basemodel.model_fields.items():
        if not _is_numeric_annotation(f.annotation):
            continue

        # Decide the numeric type to expose in the delta model
        num_type = _base_numeric_type(f.annotation) or float
        # Sensible default delta is 0 (typed)
        default = 0.0 if num_type is float else 0

        # Preserve description if any
        desc = getattr(f, "description", None)

        # Build a new Field with updated bounds, preserving description
        new_field = Field(
            default=default,
            ge=new_ge,
            le=new_le,
            description=desc,
        )
        numeric_field_names.append(name)
        field_specs.append((name, (num_type, new_field)))

    if shuffle_numeric_fields:
        random.shuffle(field_specs)

    # Build ordered field definitions for create_model
    ordered_fields: "OrderedDict[str, tuple[type[Any], Any]]" = OrderedDict()

    # reason goes first if requested
    if add_reason:
        ordered_fields["reason"] = (str, Field(default="", description=reason_docstring))

    # Then add the (optionally shuffled) numeric fields
    for name, spec in field_specs:
        ordered_fields[name] = spec

    # Create the new model class
    new_model_name = f"{original_basemodel.__name__}Delta"
    DeltaModel = create_model(  # type: ignore[assignment]
        new_model_name,
        __base__=BaseModel,
        **ordered_fields,
    )

    # Copy class docstring exactly
    DeltaModel.__doc__ = original_basemodel.__doc__

    return DeltaModel


def flatten_pydantic_model(model_cls: Type[BaseModel]) -> Type[BaseModel]:
    annotations: Dict[str, Any] = {}
    field_definitions: Dict[str, Any] = {}

    # Walk MRO (BaseModel subclasses only)
    for base in reversed(model_cls.__mro__):
        if not issubclass(base, BaseModel):
            continue

        # Merge type hints
        base_annotations = get_type_hints(base, include_extras=True)
        annotations.update(base_annotations)

        # Merge field definitions using deep copies of FieldInfo
        for name, field_info in base.__fields__.items():
            field_info = copy.deepcopy(field_info)
            field_definitions[name] = (annotations[name], field_info)

    # Dynamically create flattened model
    flattened = create_model(
        f"{model_cls.__name__}Flat",
        __base__=BaseModel,
        __doc__=model_cls.__doc__,
        **field_definitions,
    )

    return flattened

def basemodel_to_text(instance: BaseModel) -> str:
    values = []
    for name, field in instance.model_fields.items():
        value = getattr(instance, name)
        if isinstance(value, list):
            values.append(f"{name}: [")
            values.append(",".join(str(v) for v in value))
            values.append("]")
        elif isinstance(value, BaseModel):
            continue
        elif value is not None:
            values.append(f"{name}: {value}")
        values.append("\n")
    return "".join(values)

def generate_pydantic_json_schema_llm(model: type[BaseModel] | BaseModel, beautify: bool = False) -> str:
    """Recursively resolve $refs and inline all $defs for compact LLM-friendly JSON schema."""
    if isinstance(model, BaseModel):
        model = model.__class__

    raw = model.model_json_schema()

    # attach docstring as top-level description if missing
    if not raw.get("description") and model.__doc__:
        raw["description"] = model.__doc__.strip()

    defs = raw.pop("$defs", {})  # extract definitions for manual deref

    def _clean(obj: Any) -> Any:
        """Recursively clean and inline schema fragments."""
        if isinstance(obj, dict):
            # if this node is a $ref -> replace with actual definition
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    # deep copy to avoid mutation
                    return _clean(copy.deepcopy(defs[ref_name]))
                else:
                    # unresolved ref, drop it
                    obj.pop("$ref")
                    return _clean(obj)

            # remove irrelevant keys
            for k in ["title", "$ref", "$defs", "examples", "default"]:
                obj.pop(k, None)

            # recursively process subkeys
            for k, v in list(obj.items()):
                obj[k] = _clean(v)
            return obj

        elif isinstance(obj, list):
            return [_clean(x) for x in obj]
        else:
            return obj

    cleaned = _clean(raw)

    if beautify:
        res = json.dumps(cleaned, indent=4)
    else:
        res = json.dumps(cleaned, separators=(",", ":"))
    return res

def pydandic_model_to_dict_jsonable(model: BaseModel) -> dict:
    """
    To dict with all fields that can't be serialized to JSON removed.
    """
    if model is None:
        return {}

    def make_safe(value: Any) -> Any:
        # primitives
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        # nested Pydantic model
        if isinstance(value, BaseModel):
            return pydandic_model_to_dict_jsonable(value)

        # list or tuple
        if isinstance(value, (list, tuple)):
            safe_list = [make_safe(v) for v in value]
            return [v for v in safe_list if v is not None]

        # dict
        if isinstance(value, dict):
            safe_dict = {k: make_safe(v) for k, v in value.items()}
            return {k: v for k, v in safe_dict.items() if v is not None}

        # discard known non-JSON types
        if isinstance(value, (datetime, date, Enum, object)):
            return None

        # anything else
        return None

    raw_dict = model.model_dump()  # for Pydantic v2
    return {k: v for k, v in ((k, make_safe(v)) for k, v in raw_dict.items()) if v is not None}

def group_by_int(items: List[T], key_func: Callable[[T], int]) -> List[List[T]]:
    # Sort items by the key first
    sorted_items = sorted(items, key=key_func)
    # Group them using itertools.groupby
    grouped = []
    for _, group in groupby(sorted_items, key=key_func):
        grouped.append(list(group))
    return grouped