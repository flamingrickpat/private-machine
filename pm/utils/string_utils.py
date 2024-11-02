import re
from typing import Dict, Any

def same_string(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()

def list_ends_with(full_list: list[int], ending: list[int]) -> bool:
    if len(ending) > len(full_list):
        return False
    return full_list[-len(ending):] == ending

def truncate_after_last_period(long_string):
    last_dot_index = long_string.rfind('.')
    if last_dot_index == -1:  # No dot found in the string
        return long_string
    return long_string[:last_dot_index + 1]


def find_python_functions_in_text(text, functions):
    regex_pattern = "|".join(f'{func}\(.*?\)' for func in functions)
    return re.findall(regex_pattern, text)

def join_memory_paths(*args):
    paths = []
    for arg in args:
        paths.append(arg.strip("/"))
    return "/".join(paths)


class StringConstantMeta(type):
    def __getattr__(cls, name):
        if name in cls._values:
            return cls._values[name]
        raise AttributeError(f"{cls.__name__} has no attribute '{name}'")

    def __iter__(cls):
        return iter(cls._values.values())

    def __contains__(cls, item):
        return item in cls._values.values()

class StringConstant(metaclass=StringConstantMeta):
    _values: Dict[str, str] = {}

    @classmethod
    def add(cls, name: str, value: str = None):
        if value is None:
            value = name
        cls._values[name] = value
        return value

    def __class_getitem__(cls, item):
        if isinstance(item, str):
            return cls.add(item)
        elif isinstance(item, tuple) and len(item) == 2:
            return cls.add(*item)
        raise ValueError("Invalid item format")

