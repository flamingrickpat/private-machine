"""Make identity allocation repeatable for one scenario run.

The runtime draws ``uuid.uuid4`` for invocation ids, commit ids, artifact
directory names, and prompt-log file names. Those identities reach the model
through prompts, so a replay that drew different ones would ask a different
question and the recording would be useless.

:func:`deterministic_identities` replaces the process-wide identity source with
a counter for the duration of one scenario run. Recording and replay use the
same seed and the same code path, so both draw the same identities in the same
order. This is the identity counterpart of the virtual clock in
:mod:`pm.vclock`: it changes *when* and *which* identities exist, never what
the code does with them.

The harness is single-threaded, which is what makes a counter sufficient. A
scenario that started real worker threads would draw identities in an
unrepeatable order and must not rely on this.

Reading map: :func:`deterministic_identities` is the context manager,
:class:`IdentitySequence` the counter it installs.
"""
from __future__ import annotations

import hashlib
import uuid


class IdentitySequence:
    """Draw repeatable UUID values from one seed, in call order."""

    def __init__(self, seed: str) -> None:
        self.seed = seed
        self.count = 0

    def next_uuid(self) -> uuid.UUID:
        """Return the next identity for this seed."""
        self.count += 1
        digest = hashlib.sha256(f"{self.seed}:{self.count}".encode("utf-8")).digest()
        return uuid.UUID(bytes=digest[:16], version=4)


class deterministic_identities:
    """Install a repeatable ``uuid.uuid4`` for one scenario run.

    The original function is restored even when the scenario raises, so a failed
    recording cannot leave the process allocating synthetic identities.
    """

    def __init__(self, seed: str) -> None:
        self.sequence = IdentitySequence(seed)
        self._original = uuid.uuid4

    def __enter__(self) -> IdentitySequence:
        """Install the counter and return it for inspection."""
        uuid.uuid4 = self.sequence.next_uuid
        return self.sequence

    def __exit__(self, *_: object) -> None:
        """Restore the real identity source."""
        uuid.uuid4 = self._original


__all__ = ["IdentitySequence", "deterministic_identities"]
