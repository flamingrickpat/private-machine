"""Count the tokens of plain text: the unit that every context budget spends.

Cognition spends these counts on the CSM, the shard items, and the shard
tails. The memory inspector reports the same counts, so a maintainer sees the
price of the text the model reads. :mod:`pm.llm.token_estimate` re-exports
this counter and adds the chat framing on top, so two counters never disagree.

The encoding is ``o200k_base``, the one the local model servers use. The count
is an estimate: the provider stays authoritative for real usage.
"""
from __future__ import annotations

import tiktoken

_ENCODING = tiktoken.get_encoding("o200k_base")


def text_tokens(text: str) -> int:
    """Return the tiktoken token count for one text string.

    Empty text costs zero. Special marker names inside the text are ordinary
    text here, so a record that quotes a marker cannot raise.
    """
    if not text:
        return 0
    return len(_ENCODING.encode(text, disallowed_special=()))


__all__ = ["text_tokens"]
