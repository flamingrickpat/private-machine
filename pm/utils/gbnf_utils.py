import datetime
import inspect
import sys
from copy import copy
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, InvalidOperation
from enum import Enum
from typing import List, get_origin, Union, Any, Optional
from typing import get_args
from types import UnionType
from inspect import isclass

from lmformatenforcer.consts import WHITESPACE_CHARACTERS
import lmformatenforcer.jsonschemaparser
from pydantic import BaseModel, Field
from pydantic_gbnf_grammar_generator import generate_gbnf_grammar_and_documentation, generate_field_markdown, format_model_and_field_name, map_pydantic_type_to_gbnf, PydanticDataType, generate_markdown_documentation, generate_gbnf_grammar_from_pydantic_models, remove_empty_lines, get_primitive_grammar


MAX_ENUMERATED_INTEGER_VALUES = 2000
MAX_ENUMERATED_FLOAT_VALUES = 2000


def _to_decimal(v):
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _extract_numeric_constraints(field, schema_property: dict[str, Any] | None):
    constraints: dict[str, Any] = {"ge": None, "gt": None, "le": None, "lt": None, "multiple_of": None}

    # Prefer direct metadata extraction first.
    for meta in getattr(field, "metadata", []) or []:
        for key in constraints.keys():
            try:
                val = getattr(meta, key, None)
                if val is not None:
                    constraints[key] = val
            except Exception:
                pass

    # Fall back to JSON schema constraints when available.
    if schema_property:
        mapping = {
            "minimum": "ge",
            "exclusiveMinimum": "gt",
            "maximum": "le",
            "exclusiveMaximum": "lt",
            "multipleOf": "multiple_of",
        }
        for schema_key, target_key in mapping.items():
            val = schema_property.get(schema_key)
            if val is not None:
                constraints[target_key] = val

    return constraints


def _replace_field_type_in_model_rule(model_rule_line: str, field_name: str, old_rule: str, new_rule: str) -> str:
    needle = f' ws "\\"{field_name}\\"" ":" ws {old_rule}'
    repl = f' ws "\\"{field_name}\\"" ":" ws {new_rule}'
    return model_rule_line.replace(needle, repl)


def _unwrap_optional_type(t):
    try:
        if t == UnionType or isinstance(t, UnionType):
            args = get_args(t)
            non_none = [x for x in args if x is not type(None)]
            if len(non_none) == 1 and len(args) != len(non_none):
                return non_none[0]
            return t
    except Exception:
        pass

    origin = get_origin(t)
    if origin is Union:
        args = [x for x in get_args(t) if x is not type(None)]
        if len(args) == 1 and len(get_args(t)) != len(args):
            return args[0]
    return t


def _iter_embedded_basemodel_types(t):
    t = _unwrap_optional_type(t)

    if isclass(t) and issubclass(t, BaseModel):
        yield t
        return

    origin = get_origin(t)
    if origin is None:
        return

    for arg in get_args(t):
        arg = _unwrap_optional_type(arg)
        if isclass(arg) and issubclass(arg, BaseModel):
            yield arg
            continue
        # recurse nested containers/unions
        yield from _iter_embedded_basemodel_types(arg)


def _collect_referenced_models(models: list[type[BaseModel]]) -> list[type[BaseModel]]:
    out: list[type[BaseModel]] = []
    seen: set[type[BaseModel]] = set()
    queue = list(models)

    while queue:
        model = queue.pop(0)
        if model in seen:
            continue
        seen.add(model)
        out.append(model)

        try:
            fields = model.model_fields.values()
        except Exception:
            fields = []
        for field in fields:
            for nested in _iter_embedded_basemodel_types(field.annotation):
                if nested not in seen:
                    queue.append(nested)

    return out


def _build_integer_rule_from_constraints(rule_name: str, constraints: dict[str, Any]) -> str | None:
    ge = _to_decimal(constraints.get("ge"))
    gt = _to_decimal(constraints.get("gt"))
    le = _to_decimal(constraints.get("le"))
    lt = _to_decimal(constraints.get("lt"))
    multiple_of = _to_decimal(constraints.get("multiple_of"))

    lower = None
    upper = None

    if ge is not None:
        lower = int(ge.to_integral_value(rounding=ROUND_CEILING))
    if gt is not None:
        gt_lower = int(gt.to_integral_value(rounding=ROUND_FLOOR)) + 1
        lower = gt_lower if lower is None else max(lower, gt_lower)

    if le is not None:
        upper = int(le.to_integral_value(rounding=ROUND_FLOOR))
    if lt is not None:
        lt_upper = int(lt.to_integral_value(rounding=ROUND_CEILING)) - 1
        upper = lt_upper if upper is None else min(upper, lt_upper)

    if lower is not None and upper is not None and lower > upper:
        return None

    has_int_constraints = any(x is not None for x in [ge, gt, le, lt, multiple_of])
    if not has_int_constraints:
        return None

    if lower is not None and upper is not None:
        count = upper - lower + 1
        if count <= 0:
            return None
        if count <= MAX_ENUMERATED_INTEGER_VALUES:
            values = list(range(lower, upper + 1))
            if multiple_of is not None and multiple_of != 0 and multiple_of == multiple_of.to_integral_value():
                step = int(abs(multiple_of))
                if step > 1:
                    values = [v for v in values if v % step == 0]
            if not values:
                return None
            return f"{rule_name} ::= " + " | ".join(f'"{v}"' for v in values)

    # Fallback sign constraints for very wide or single-sided ranges.
    if lower is not None and lower >= 0:
        return f'{rule_name} ::= "0" | [1-9] [0-9]*'
    if upper is not None and upper < 0:
        return f'{rule_name} ::= "-" [1-9] [0-9]*'
    return None


def _build_float_rule_from_constraints(rule_name: str, constraints: dict[str, Any]) -> str | None:
    ge = _to_decimal(constraints.get("ge"))
    gt = _to_decimal(constraints.get("gt"))
    le = _to_decimal(constraints.get("le"))
    lt = _to_decimal(constraints.get("lt"))
    multiple_of = _to_decimal(constraints.get("multiple_of"))

    has_float_constraints = any(x is not None for x in [ge, gt, le, lt, multiple_of])
    if not has_float_constraints:
        return None

    lower = ge if ge is not None else gt
    upper = le if le is not None else lt
    lower_exclusive = gt is not None and ge is None
    upper_exclusive = lt is not None and le is None

    # Exact finite enumeration when bounded + multiple_of is available.
    if lower is not None and upper is not None and multiple_of is not None and multiple_of > 0:
        start = (lower / multiple_of).to_integral_value(rounding=ROUND_CEILING) * multiple_of
        if lower_exclusive and start <= lower:
            start += multiple_of
        end = (upper / multiple_of).to_integral_value(rounding=ROUND_FLOOR) * multiple_of
        if upper_exclusive and end >= upper:
            end -= multiple_of
        if end >= start:
            count = int(((end - start) / multiple_of).to_integral_value(rounding=ROUND_FLOOR)) + 1
            if 0 < count <= MAX_ENUMERATED_FLOAT_VALUES:
                vals: list[str] = []
                cur = start
                for _ in range(count):
                    s = format(cur.normalize(), "f")
                    if "." in s:
                        s = s.rstrip("0").rstrip(".")
                    vals.append(s if s != "-0" else "0")
                    cur += multiple_of
                if vals:
                    return f"{rule_name} ::= (" + " | ".join(f'"{v}"' for v in vals) + ") ws"

    # Common exact bounded score ranges used heavily in this codebase.
    if ge == Decimal("0") and le == Decimal("1"):
        return (
            f'{rule_name} ::= ('
            f'"0" ("." [0-9]+)?'
            f' | "1" ("." "0"+)?'
            f') ws'
        )
    if ge == Decimal("-1") and le == Decimal("1"):
        return (
            f'{rule_name} ::= ('
            f'"-"? "0" ("." [0-9]+)?'
            f' | "-" "1" ("." "0"+)?'
            f' | "1" ("." "0"+)?'
            f') ws'
        )

    # Practical fallback: sign-constrained float.
    if ge is not None and ge >= 0:
        return f'{rule_name} ::= ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws'
    if gt is not None and ge is None and gt >= 0:
        return f'{rule_name} ::= [1-9] [0-9]* ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws'
    if le is not None and le <= 0:
        return (
            f'{rule_name} ::= ("0" ("." [0-9]+)? | "-" ([0-9] | [1-9] [0-9]*) ("." [0-9]+)?) '
            f'([eE] [-+]? [0-9]+)? ws'
        )
    if lt is not None and le is None and lt <= 0:
        return f'{rule_name} ::= "-" ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws'
    return None


def first_non_none_type(u: UnionType) -> type:
    try:
        args = get_args(u)
    except Exception:
        raise TypeError(f"Expected a UnionType, got {type(u).__name__!r}")

    # Make sure it really was a union
    if not args:
        raise TypeError(f"Type {u!r} has no arguments—expected a Union")

    for t in args:
        if t is not type(None):
            return t

    raise ValueError(f"No non-NoneType member found in {u!r}")

def get_pydantic_options(u: UnionType):
    try:
        res = []
        args = get_args(u)
        for t in args:
            if isclass(t) and issubclass(t, BaseModel):
                res.append(t)
            else:
                return None
        return res
    except:
        return None


def new_generate_field_markdown(
    field_name: str, field_type: type[Any], model: type[BaseModel], depth=1, documentation_with_field_description=True
) -> str:
    indent = "    " * depth

    field_info = model.model_fields.get(field_name)
    field_description = field_info.description if field_info and field_info.description else ""

    if get_origin(field_type) == list:
        element_type = get_args(field_type)[0]
        field_text = f"{indent}{field_name} ({format_model_and_field_name(field_type.__name__)} of {format_model_and_field_name(element_type.__name__)})"
        if field_description != "":
            field_text += ":\n"
        else:
            field_text += "\n"
    elif get_origin(field_type) == Union:
        element_types = get_args(field_type)
        types = []
        for element_type in element_types:
            types.append(format_model_and_field_name(element_type.__name__))
        field_text = f"{indent}{field_name} ({' or '.join(types)})"
        if field_description != "":
            field_text += ":\n"
        else:
            field_text += "\n"
    else:
        # flamingrickpat: check for UnionType
        if field_type == UnionType or isinstance(field_type, UnionType):
            field_type = first_non_none_type(field_type)

        field_text = f"{indent}{field_name} ({format_model_and_field_name(field_type.__name__)})"
        if field_description != "":
            field_text += ":\n"
        else:
            field_text += "\n"

    if not documentation_with_field_description:
        return field_text

    if field_description != "":
        field_text += f"        Description: " + field_description + "\n"

    # Check for and include field-specific examples if available
    if (
        hasattr(model, "Config")
        and hasattr(model.Config, "json_schema_extra")
        and "example" in model.Config.json_schema_extra
    ):
        field_example = model.Config.json_schema_extra["example"].get(field_name)
        if field_example is not None:
            example_text = f"'{field_example}'" if isinstance(field_example, str) else field_example
            field_text += f"{indent}  Example: {example_text}\n"

    if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
        field_text += f"{indent}  Details:\n"
        for name, type_ in field_type.__annotations__.items():
            field_text += generate_field_markdown(name, type_, field_type, depth + 2)

    return field_text

def new_get_members_structure(cls, rule_name):
    if issubclass(cls, Enum):
        # Handle Enum types
        members = [f'"\\"{member.value}\\""' for name, member in cls.__members__.items()]
        return f"{cls.__name__.lower()} ::= " + " | ".join(members)
    # flamingrickpat: check if annotations even exist!
    if hasattr(cls, "__annotations__") and cls.__annotations__ and cls.__annotations__ != {}:
        result = f'{rule_name} ::= "{{"'
        # Modify this comprehension
        members = [
            f'  "\\"{name}\\"" ":"  {map_pydantic_type_to_gbnf(param_type)}'
            for name, param_type in cls.__annotations__.items()
            if name != "self"
        ]

        result += '"," '.join(members)
        result += '  "}"'
        return result
    if rule_name == "custom-class-any":
        result = f"{rule_name} ::= "
        result += "value"
        return result

    init_signature = inspect.signature(cls.__init__)
    parameters = init_signature.parameters
    result = f'{rule_name} ::=  "{{"'
    # Modify this comprehension too
    members = [
        f'  "\\"{name}\\"" ":"  {map_pydantic_type_to_gbnf(param.annotation)}'
        for name, param in parameters.items()
        if name != "self" and param.annotation != inspect.Parameter.empty
    ]

    result += '", "'.join(members)
    result += '  "}"'
    return result

def new_map_pydantic_type_to_gbnf(pydantic_type: type[Any]) -> str:
    if isclass(pydantic_type) and issubclass(pydantic_type, str):
        return PydanticDataType.STRING.value
    elif isclass(pydantic_type) and issubclass(pydantic_type, bool):
        return PydanticDataType.BOOLEAN.value
    elif isclass(pydantic_type) and issubclass(pydantic_type, int):
        return PydanticDataType.INTEGER.value
    elif isclass(pydantic_type) and issubclass(pydantic_type, float):
        return PydanticDataType.FLOAT.value
    elif isclass(pydantic_type) and issubclass(pydantic_type, Enum):
        return PydanticDataType.ENUM.value
    elif isclass(pydantic_type) and issubclass(pydantic_type, BaseModel):
        return format_model_and_field_name(pydantic_type.__name__)
    elif get_origin(pydantic_type) is list:
        element_type = get_args(pydantic_type)[0]
        return f"{map_pydantic_type_to_gbnf(element_type)}-list"
    elif get_origin(pydantic_type) is set:
        element_type = get_args(pydantic_type)[0]
        return f"{map_pydantic_type_to_gbnf(element_type)}-set"
    elif get_origin(pydantic_type) is Union:
        union_types = get_args(pydantic_type)
        union_rules = [map_pydantic_type_to_gbnf(ut) for ut in union_types]
        return f"union-{'-or-'.join(union_rules)}"
    elif get_origin(pydantic_type) is Optional:
        element_type = get_args(pydantic_type)[0]
        return f"optional-{map_pydantic_type_to_gbnf(element_type)}"
    elif isclass(pydantic_type):
        return f"{PydanticDataType.CUSTOM_CLASS.value}-{format_model_and_field_name(pydantic_type.__name__)}"
    elif get_origin(pydantic_type) is dict:
        key_type, value_type = get_args(pydantic_type)
        return f"custom-dict-key-type-{format_model_and_field_name(map_pydantic_type_to_gbnf(key_type))}-value-type-{format_model_and_field_name(map_pydantic_type_to_gbnf(value_type))}"
    else:
        # flamingrickpat: check for UnionType
        if pydantic_type == UnionType or isinstance(pydantic_type, UnionType):
            pds = get_pydantic_options(pydantic_type)
            if pds is not None:
                res = []
                for pd in pds:
                    res.append(format_model_and_field_name(pd.__name__))
                return "union-" + "-".join(res) #"\n".join(res)
            else:
                t = first_non_none_type(pydantic_type)
                return map_pydantic_type_to_gbnf(t)
        return "unknown"

def get_allowed_characters_fixed(self) -> str:
    if self.seen_whitespace_after_digits:
        return WHITESPACE_CHARACTERS
    if len(self.parsed_string) > 12:
        self.seen_whitespace_after_digits = True
        return WHITESPACE_CHARACTERS
    else:
        allowed_characters = "0123456789"
    if not self.parsed_string:
        allowed_characters += "-" + WHITESPACE_CHARACTERS
    if self.parsed_string and len(self.parsed_string) == 1 and self.parsed_string[0] == "0":
        allowed_characters = WHITESPACE_CHARACTERS
    if self.parsed_string and len(self.parsed_string) == 2 and self.parsed_string == "-0":
        allowed_characters = "." + WHITESPACE_CHARACTERS
    if self.parsed_string and self.parsed_string[-1] in "eE":
        allowed_characters += "-+"
    if self.seen_digit and not self.seen_exponent:
        allowed_characters += "eE"
    if self.allow_floating_point and not self.seen_decimal_point and self.seen_digit and not self.seen_exponent:
        allowed_characters += "."
    if self.parsed_string and self.parsed_string[-1].isdigit():
        allowed_characters += WHITESPACE_CHARACTERS
    return allowed_characters

lmformatenforcer.jsonschemaparser.NumberParsingState.get_allowed_characters = get_allowed_characters_fixed

def fix_gbnf_grammar_generator():
    module = sys.modules["pydantic_gbnf_grammar_generator.main"]
    module.generate_field_markdown = new_generate_field_markdown
    module.get_members_structure = new_get_members_structure
    module.map_pydantic_type_to_gbnf = new_map_pydantic_type_to_gbnf
    sys.modules["pydantic_gbnf_grammar_generator.main"] = module

def better_generate_gbnf_grammar_and_documentation(pydantic_model_list, default_max_list_length: int | None = 8):
    #gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(pydantic_model_list)

    documentation = ""
    try:
        documentation = generate_markdown_documentation(
            copy(pydantic_model_list),
            "Output Model",
            "Output Field",
            documentation_with_field_description=True,
        )
    except:
        pass

    grammar = generate_gbnf_grammar_from_pydantic_models(
        pydantic_model_list, None, None, False
    )
    grammar = remove_empty_lines(grammar + get_primitive_grammar(grammar))

    # remove default dt rules
    grammar_lines = [x for x in grammar.split("\n") if not x.startswith("custom-class-datetime") and not x.startswith("custom-class-date") and not x.startswith("ws ::=")]

    # get all rules for list types; set max length to max_length or default_max_list_length
    # to prevent endless list generation for when the llm adds garbage items until max tokens
    # also limit ws count to max 8 consecutive
    all_models = _collect_referenced_models(pydantic_model_list)
    for tool in all_models:
        tool_name = format_model_and_field_name(tool.__name__)
        tool_schema_properties = {}
        try:
            tool_schema_properties = (tool.model_json_schema() or {}).get("properties", {}) or {}
        except Exception:
            tool_schema_properties = {}

        model_rule_idx = None
        for i, line in enumerate(grammar_lines):
            if line.startswith(tool_name + " ::= "):
                model_rule_idx = i
                break

        extra_number_rules: list[str] = []
        for field_name, field in tool.model_fields.items():
            schema_property = tool_schema_properties.get(field_name, {})

            if get_origin(field.annotation) == list:
                le = default_max_list_length
                for i in range(len(field.metadata)):
                    try:
                        le = field.metadata[0].max_length
                        break
                    except:
                        pass
                if le is not None:
                    for i in range(len(grammar_lines)):
                        fn = field_name.replace("_", "-")
                        if grammar_lines[i].startswith(tool_name + "-" + fn):
                            grammar_lines[i] = grammar_lines[i].replace(')*  "]"', f'){{0,{le}}}  "]"')

            # Add ge/le/gt/lt/multiple_of aware numeric rules.
            if model_rule_idx is not None:
                constraints = _extract_numeric_constraints(field, schema_property)
                field_type = _unwrap_optional_type(field.annotation)
                number_rule_name = f"{tool_name}-{field_name.replace('_', '-')}-number"

                int_rule = None
                float_rule = None
                if isclass(field_type) and issubclass(field_type, int):
                    int_rule = _build_integer_rule_from_constraints(number_rule_name, constraints)
                    if int_rule is not None:
                        grammar_lines[model_rule_idx] = _replace_field_type_in_model_rule(
                            grammar_lines[model_rule_idx], field_name, "integer", number_rule_name
                        )
                        extra_number_rules.append(int_rule)
                elif isclass(field_type) and issubclass(field_type, float):
                    float_rule = _build_float_rule_from_constraints(number_rule_name, constraints)
                    if float_rule is not None:
                        grammar_lines[model_rule_idx] = _replace_field_type_in_model_rule(
                            grammar_lines[model_rule_idx], field_name, "float", number_rule_name
                        )
                        extra_number_rules.append(float_rule)

        if extra_number_rules:
            grammar_lines.extend(extra_number_rules)

    gbnf_grammar = "\n".join(grammar_lines)

    rules_datetime_unknown = r"""
HEX   ::= [0-9a-fA-F]
DIGIT ::= [0-9]

ws ::= [ \t\n]{0,8}

unknown ::= string

custom-class-date     ::= date-literal
custom-class-datetime ::= datetime-literal

date-literal ::= "\"" date-part "\"" ws

date-part  ::= YYYY "-" MM "-" DD

datetime-literal ::= "\"" date-part "T" time-part timezone? "\"" ws

time-part ::= hh ":" mm ":" ss frac? 

timezone ::= "Z" | (("+" | "-") hh ":" mm)

frac ::= "." DIGIT{1,9} 

YYYY ::= DIGIT DIGIT DIGIT DIGIT
MM   ::= "0" DIGIT | "1" [0-2]
DD   ::= "0" DIGIT | [12] DIGIT | "3"[0-1]
hh   ::= [01] DIGIT | "2"[0-3]
mm   ::= [0-5] DIGIT
ss   ::= [0-5] DIGIT
"""

    gbnf_grammar += rules_datetime_unknown
    return gbnf_grammar, documentation


if __name__ == '__main__':
    class TestBm(BaseModel):
        a: str | None
        b: datetime.datetime
        d: datetime.date
        c: List[str] = Field()
        d: List[str] = Field(max_length=12)
    tools = [TestBm]

    fix_gbnf_grammar_generator()
    gram, doc = better_generate_gbnf_grammar_and_documentation(tools)
    print(gram)
    print(doc)
