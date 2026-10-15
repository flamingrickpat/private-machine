"""Keep detached narrative setup outside a generated public utterance."""
import re


def strip_narrator_setup(text: str, user_name: str) -> str:
    """Remove a detached question setup before the companion's speech.

    Captured Story output described the user asking a question in a separate
    leading paragraph. That paragraph is outside the spoken continuation.
    Single-paragraph jokes and ordinary mentions of the user stay unchanged.
    The caller retains the raw draft and rejects an empty spoken result.
    """
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    if len(parts) == 2 and re.match(rf"^\s*{re.escape(user_name)}\s+(?:asks|asked)\b", parts[0]):
        return parts[1].strip()
    return text
