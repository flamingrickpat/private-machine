"""Separate generated speech from narrator setup and provider output."""
import re


CLASSIFIER_OUTPUT_ERROR = "The model returned a classification report without public speech."


class SpeechOutputError(ValueError):
    """The current sample cannot supply a public utterance."""


def contains_tool_protocol(text: str) -> bool:
    """Identify serialized function calls emitted as text by a speech model."""
    return re.search(r"<(?:/?[｜|]DSML[｜|]\s*(?:calls|invoke|parameter)\b|tool_call>|function=)",
                     text, flags=re.IGNORECASE) is not None


def validate_public_speech(text: str) -> None:
    if contains_tool_protocol(text):
        raise SpeechOutputError("The model returned tool protocol. Answer the partner in ordinary speech using the supplied evidence.")


def strip_classifier_report(text: str) -> str:
    """Read speech following a leading DeepSeek report in the assistant content.

    An envelope alone supplies no speech. A verdict or a line break can
    separate it from a spoken continuation. Keep inline examples intact.
    """
    envelope = re.match(r"\s*<ds_safety>.*?</ds_safety>", text, flags=re.IGNORECASE | re.DOTALL)
    if envelope is None:
        return text
    remainder = text[envelope.end():]
    if not remainder.strip():
        return ""
    verdict = re.match(r"\s*(?:safe|unsafe)\b[ \t]*(?:[.!?。！？]+|\r?\n|\Z)", remainder, flags=re.IGNORECASE)
    if verdict is not None:
        return remainder[verdict.end():].strip()
    if remainder.lstrip(" \t").startswith(("\r", "\n")):
        return remainder.strip()
    return text


def reject_classifier_report(text: str) -> None:
    """Reject a provider report before it can become a public dialogue record."""
    if strip_classifier_report(text) != text:
        raise SpeechOutputError("A classification report is still present in the public reply.")


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
