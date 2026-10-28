"""Fast YAML helpers that use the C loader and dumper from PyYAML.

The C classes parse the same safe subset as ``yaml.safe_load`` and
``yaml.safe_dump``, but much faster on large ghost snapshots.
"""
import yaml
from yaml import YAMLError
from typing import BinaryIO, TextIO, TypedDict, Unpack, overload

__all__ = ["YAMLError", "safe_dump", "safe_load"]


class YamlDumpOptions(TypedDict, total=False):
    """Supported text-dump options at the external YAML codec boundary."""

    allow_unicode: bool
    sort_keys: bool
    default_flow_style: bool | None
    default_style: str | None
    indent: int
    width: int
    canonical: bool
    explicit_start: bool
    explicit_end: bool
    version: tuple[int, int]
    tags: dict[str, str]
    line_break: str


def safe_load(stream: str | bytes | TextIO | BinaryIO) -> object:
    """Parse one YAML document with the safe C loader.

    Raises ``YAMLError`` for invalid YAML.
    """
    return yaml.load(stream, Loader=yaml.CSafeLoader)


@overload
def safe_dump(data: object, stream: None = None, **kwargs: Unpack[YamlDumpOptions]) -> str: ...


@overload
def safe_dump(data: object, stream: TextIO, **kwargs: Unpack[YamlDumpOptions]) -> None: ...


def safe_dump(data: object, stream: TextIO | None = None, **kwargs: Unpack[YamlDumpOptions]) -> str | None:
    """Serialize ``data`` with the safe C dumper and return the text when ``stream`` is None."""
    return yaml.dump(data, stream, Dumper=yaml.CSafeDumper, **kwargs)
