import inspect
import logging
from enum import Enum
from typing import Callable, Dict, Any

from jinja2 import Template

logger = logging.getLogger(__name__)

from typing import Optional, List


class DynamicLiteral:
    @staticmethod
    def get_literal() -> Optional[List[str]]:
        return None

    @staticmethod
    def get_literal_hint() -> Optional[List[str]]:
        return None

    @staticmethod
    def get_literal_documentation() -> Optional[List[str]]:
        return None

class ArgLiteral(str):
    pass

class Grammar:
    def __init__(self, gbnf: str = ""):
        self.gbnf: str = gbnf


# emoji causes error
# emoji ::= [\U0001F600-\U0001F64F]

# newlines are annoying
# ws ::= ([ \\t\\n] ws)?

# finishFunction
# ("\\n" functionCall)*

TEMPLATE_BASE = """
# Defines the overall structure: think function, optional additional calls, finish function
root ::= {internal_thought} functionLine (functionLine(functionLine(functionLine)?)?)?

# Specific rule for the think function with an internal thought as a parameter
thinkFunction ::= "explain_reasoning" ws "(" ws thinkParameters ws ")"
thinkParameters ::= "reasoning""="thinkValue
thinkValue ::= validString ws
# Specific rule for the finish function with no parameters
finishFunction ::= "finish()"

functionLine ::= functionCall ws

alphanumeric ::= [a-zA-Z0-9 ]
hyphen ::= "-"
underscore ::= "_"


validCharacter ::= alphanumeric | hyphen | underscore
validString ::= "\\"" validCharacter+ "\\"" ws

functionCall ::= {function_rules}

{function_definitions}

ws ::= ([ \\t] ws)?
"""


TEMPLATE_STRUCTURE_BASE = """
# Defines the overall structure: think function, optional additional calls, finish function
root ::= {ruleset}

alphanumeric ::= [a-zA-Z0-9 ]
hyphen ::= "-"
underscore ::= "_"

validCharacter ::= alphanumeric | hyphen
validString ::= validCharacter+

anything ::= (validString ws)?
thought ::= "Thought:" ws validString ws
consideration ::= "Consideration:" ws validString ws
observation ::= "Observation:" ws validString ws
response ::= "Response:" ws validString ws
action ::= "<|python_tag|>" anything "<|eom_id|>"

ws ::= ([ \\t\\n] ws)?
"""


def type_to_gbnf_rule(param_type: type) -> str | None | Any:
    """Maps Python types and enums to GBNF rules."""
    # Check if the type is an Enum
    if inspect.isclass(param_type) and issubclass(param_type, Enum):
        # Generate a rule with all enum values as alternatives
        enum_values = '|'.join([f'"{member.value}"' for name, member in param_type.__members__.items()])
        return f'({enum_values})'

    # The rest of the mapping remains the same as before
    type_rules = {
        "int": '[0-9]+',
        "float": '([0-9]*[.])?[0-9]+',
        "bool": '("True"|"False")',
        "str": 'validString',  # Simplistic representation, adjust as needed
        "datetime": '\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}',  # ISO 8601 format
    }

    possible_values = None
    if issubclass(param_type, DynamicLiteral):
        tmp = param_type()
        possible_values = tmp.get_literal()
    if issubclass(param_type, ArgLiteral):
        return None

    if hasattr(param_type, '__args__'):
        possible_values = param_type.__args__

    if possible_values is not None:
        lst = []
        for pv in possible_values:
            lst.append(f'"\\"{pv}\\""')
        tmp = "|".join(lst)
        return f'({tmp})'
    else:
        tmp = type_rules[param_type.__name__]
        return tmp


def function_metadata_to_dict(func: Callable) -> Dict[str, Any]:
    if isinstance(func, Callable):

        """Extracts and returns function metadata including its name, parameters, types and defaults."""
        sig = inspect.signature(func)
        metadata = {
            "name": func.__name__,
            "docstr": func.__doc__,
            "params": [],
        }
        for name, param in sig.parameters.items():
            if param.name == "self":
                continue
            d = {
                "name": param.name,
                "type": param.annotation,
                "default": None
            }
            if param.default is not inspect.Parameter.empty:
                d["default"] = param.default
            metadata["params"].append(d)
        return metadata
    else:
        raise Exception(f"Not a callable!")


def generate_gbnf_structure_grammar(ruleset: str) -> Grammar:
    f = TEMPLATE_STRUCTURE_BASE.format(ruleset=ruleset)
    return Grammar(gbnf=f)

# Function to generate GBNF grammar
def generate_gbnf_grammar(functions: List[Callable], internal_thought: bool = False) -> Grammar:
    gbnf_template = TEMPLATE_BASE
    function_rules = []
    function_definitions = []

    if isinstance(functions, Callable):
        functions = [functions]

    for func in functions:
        if func.__name__ == "explain_reasoning":
            internal_thought = True
            continue

        metadata = function_metadata_to_dict(func)
        func_name = metadata["name"]
        # gbnf
        if func_name == "think" or func_name == "finish":
            continue

        normalized_func_name = func_name.replace('_', '')

        if len(metadata["params"]) > 0:
            function_rules.append(f'"{func_name}" ws "(" ws {normalized_func_name}Params ws ")"')
        else:
            function_rules.append(f'"{func_name}" ws "(" ws ")"')

        # Generating parameters grammar
        params_rules = []
        for param in metadata["params"]:
            name = param["name"]
            _type = param["type"]
            #default = param["default"]

            normalized_name = name.capitalize().replace('_', '')
            type_grammar = type_to_gbnf_rule(_type)
            if type_grammar is None:
                params_rules.append(f'{normalized_name} ::= "{name}"')
            else:
                # Simplify rule names by removing underscores and capitalizing
                params_rules.append(f'{normalized_name}Value ::= {type_grammar} ws')

                # Handling optional parameters based on the presence of a default
                optional_indicator = ''# '?' if default != inspect.Parameter.empty else ''
                params_rules.append(f'{normalized_name} ::= "{name}""="{normalized_name}Value{optional_indicator}')

        params_sequence = " ws \",\" ws ".join([f'{p["name"].capitalize().replace("_", "")}' for p in metadata["params"]])

        if len(metadata["params"]) > 0:
            function_definitions.append(f'{normalized_func_name}Params ::= {params_sequence}\n' + "\n".join(params_rules))

    f = gbnf_template.format(
        function_rules=" | ".join(function_rules),
        function_definitions="\n\n".join(function_definitions),
        internal_thought='thinkFunction "\\n"' if internal_thought else ""
    )

    assert "None" not in f
    return Grammar(gbnf=f)


def generate_func_description(functions: List[Callable]) -> str:
    lst = []
    for func in functions:
        if isinstance(func, Callable):
            # Getting the signature of the function
            sig = inspect.signature(func)
            # Getting the docstring of the function
            doc = inspect.getdoc(func)

            format_args = []
            sig_args = []
            args = list(sig.parameters.items())
            for arg in args:
                name = arg[0]
                type_class = arg[1].annotation

                arg = {
                    "name": name,
                    "type": type_class.__name__,
                    "literals": []
                }
                if issubclass(type_class, DynamicLiteral):
                    tmp = type_class()
                    lit_hints = tmp.get_literal_hint()
                    if lit_hints is not None:

                        lit_hints = [f'"{x}"' for x in lit_hints]
                        lit_str = ", ".join(lit_hints)
                        sig_args.append(f"{name}: Literal[{lit_str}]")
                    else:
                        sig_args.append(f"{name}: str")

                    docs = tmp.get_literal_documentation()
                    if docs is not None:
                        arg["literals"] = docs
                elif issubclass(type_class, ArgLiteral):
                    sig_args.append(f"{name}: str")
                else:
                    sig_args.append(f"{name}: {type_class.__name__}")

                format_args.append(arg)

            # Formatting the function definition with the signature
            sig_str = ", ".join(sig_args)
            sig_str = sig_str.replace("self, ", "").replace("self", "").replace("'", "")
            try:
                sig_ret = sig.return_annotation.__name__
            except:
                sig_ret = "None"

            template = Template(doc)
            doc = template.render(args=format_args)

            function_def = f"def {func.__name__}({sig_str}) -> {sig_ret}:"
            # Preparing the docstring to be included if it exists
            formatted_doc = '    """\n    ' + (doc.replace('\n', '\n    ') if doc else '') + '\n    """'



            # Combine the function definition with the docstring
            lst.append(function_def + '\n' + formatted_doc + '\n')
    return "\n".join(lst)
