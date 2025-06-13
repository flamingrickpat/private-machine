import datetime
import inspect
import sys
from enum import Enum
from typing import List, get_origin, Union, Any, Optional
from typing import get_args
from types import UnionType
from inspect import isclass

from pydantic import BaseModel, Field
from pydantic_gbnf_grammar_generator import generate_gbnf_grammar_and_documentation, generate_field_markdown, format_model_and_field_name, map_pydantic_type_to_gbnf, PydanticDataType

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
            t = first_non_none_type(pydantic_type)
            return map_pydantic_type_to_gbnf(t)
        return "unknown"

def fix_gbnf_grammar_generator():
    module = sys.modules["pydantic_gbnf_grammar_generator.main"]
    module.generate_field_markdown = new_generate_field_markdown
    module.get_members_structure = new_get_members_structure
    module.map_pydantic_type_to_gbnf = new_map_pydantic_type_to_gbnf
    sys.modules["pydantic_gbnf_grammar_generator.main"] = module

def better_generate_gbnf_grammar_and_documentation(pydantic_model_list, default_max_list_length: int | None = 16):
    gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(pydantic_model_list)

    # remove default dt rules
    grammar_lines = [x for x in gbnf_grammar.split("\n") if not x.startswith("custom-class-datetime") and not x.startswith("custom-class-date")]

    # get all rules for list types; set max length to max_length or default_max_list_length
    # to prevent endless list generation for when the llm adds garbage items until max tokens
    for tool in pydantic_model_list:
        tool_name = format_model_and_field_name(tool.__name__)
        for field_name, field in tool.model_fields.items():
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
                        if grammar_lines[i].startswith(tool_name + "-" + field_name):
                            grammar_lines[i] = grammar_lines[i].replace(')*  "]"', f'){{0,{le}}}  "]"')

    gbnf_grammar = "\n".join(grammar_lines)

    rules_datetime_unknown = r"""
HEX   ::= [0-9a-fA-F]
DIGIT ::= [0-9]

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