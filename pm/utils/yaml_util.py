"""Fast YAML helpers that use the C loader and dumper from PyYAML.

The C classes parse the same safe subset as ``yaml.safe_load`` and
``yaml.safe_dump``, but much faster on large ghost snapshots.
"""
import yaml
from yaml import YAMLError

__all__ = ["YAMLError", "safe_dump", "safe_load"]


def safe_load(stream):
    """Parse one YAML document with the safe C loader.

    Raises ``YAMLError`` for invalid YAML.
    """
    return yaml.load(stream, Loader=yaml.CSafeLoader)


def safe_dump(data, stream=None, **kwargs):
    """Serialize ``data`` with the safe C dumper and return the text when ``stream`` is None."""
    return yaml.dump(data, stream, Dumper=yaml.CSafeDumper, **kwargs)
