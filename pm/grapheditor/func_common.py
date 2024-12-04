import enum
from typing import Any, Optional
import logging
from pydantic import BaseModel

from pm.grapheditor.decorator import graph_function_pure, graph_function, graph_datamodel

logger = logging.getLogger(__name__)


@graph_function()
def log(a: Any) -> None:
    logger.info(f"{a}")

@graph_function_pure()
def log_pure(a: Any) -> None:
    logger.info(f"{a}")

@graph_function_pure()
def concat_string(a: str, b: str) -> str:
    return a + b

@graph_function_pure()
def concat_string_semicolon(a: str, b: str) -> str:
    return a + ";" + b

@graph_function_pure()
def add(a: float, b: float) -> float:
    return a + b

@graph_function_pure()
def sub(a: float, b: float) -> float:
    return a - b

@graph_function_pure()
def mul(a: float, b: float) -> float:
    return a * b

@graph_function_pure()
def div(a: float, b: float) -> float:
    return a / b

@graph_function_pure()
def round(a: float) -> float:
    return round(a)

@graph_function_pure()
def trunc(a: float) -> float:
    return float(int(a))

@graph_function_pure()
def is_none(a: Any) -> bool:
    return a is None

@graph_function_pure()
def length(a: Any) -> int:
    return len(a)

@graph_function_pure()
def append(*args) -> Any:
    res = []
    for a in args:
        if a is not None:
            try:
                for tmp in a:
                    res.append(tmp)
            except:
                res.append(a)
    return res

#@graph_function_pure()
#def union(*args) -> Any:
#    res = args[0].__class__()
#    for a in args:
#        if a not in res:
#            res.append(a)
#    return res

@graph_function_pure()
def to_list(*args) -> Any:
    res = list()
    for a in args:
        if a not in res and a is not None:
            res.append(a)
    return res

@graph_function_pure()
def greater_than(a: Any, b: Any) -> bool:
    return a > b

@graph_function_pure()
def greater_equal_than(a: Any, b: Any) -> bool:
    return a >= b

@graph_function_pure()
def lesser_than(a: Any, b: Any) -> bool:
    return a < b

@graph_function_pure()
def lesser_equal_than(a: Any, b: Any) -> bool:
    return a <= b

@graph_function_pure()
def equal(a: Any, b: Any) -> bool:
    return a == b

@graph_function_pure()
def slice_enumerable(enumerable: Any, start: int, stop: Optional[int] = None, step: Optional[int] = None):
    index = start
    if stop is None:
        end = start + 1
    else:
        end = stop
    if step is None:
        stride = 1
    else:
        stride = step
    return enumerable[index:end:stride]

@graph_function_pure()
def make_key_value_pair(key: str, value: Any) -> Any:
    if key is None:
        return {}
    return {
        key: value
    }

@graph_function_pure()
def make_dict(*args: Any) -> Any:
    res = {}
    for arg in args:
        if isinstance(arg, dict):
            res.update(arg)
    return res

@graph_function_pure()
def to_string(value: Any) -> str:
    if isinstance(value, enum.Enum):
        return value.value
    return str(value)

@graph_function_pure()
def to_semicolon_string(lst: Any) -> str:
    return ";".join(lst)
