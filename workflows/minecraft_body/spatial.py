import re
from pydantic import BaseModel, ConfigDict


class WorldPosition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    x: float
    y: float
    z: float


_NUMBER = r"(-?\d+(?:\.\d+)?)"
_SEPARATOR = r"(?:\s*,\s*|\s+)"
_COORDINATES = re.compile(
    r"(?<![\w.])(?:x\s*[:=]?\s*)?" + _NUMBER + _SEPARATOR
    + r"(?:y\s*[:=]?\s*)?" + _NUMBER + _SEPARATOR + r"(?:z\s*[:=]?\s*)?" + _NUMBER
    + r"(?!\w)", re.IGNORECASE)


def mentions_coordinates(text: str) -> bool:
    return _COORDINATES.search(text) is not None


def coordinates_in_answer(text: str, position: WorldPosition) -> bool:
    for match in _COORDINATES.finditer(text):
        x, y, z = (float(value) for value in match.groups())
        if abs(x - position.x) <= 0.01 and abs(y - position.y) <= 0.01 and abs(z - position.z) <= 0.01:
            return True
    return False
