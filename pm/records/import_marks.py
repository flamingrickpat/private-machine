"""Typed marks for records that did not come from a lived cycle.

A character import writes records from a chat log, and it writes a few
synthetic records about the move. These records look like lived records,
so the live loop reads them without special cases. The mark in
``record.metadata[IMPORT_MARK_KEY]`` lets later code and the owner tell them
apart. Provenance stays as the live path sets it (``SourceKind.User`` for the
user, ``SourceKind.Self`` for the companion), so evidence roles do not change.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from pm.records.base import RecordBase

#: The metadata key of the mark.
IMPORT_MARK_KEY = "import_mark"


class ImportOrigin(StrEnum):
    """What wrote the record."""

    #: ``character init`` wrote it: the entities of the user and the companion.
    CharacterInit = "character_init"
    #: One row of an imported chat log.
    ChatLog = "chat_log"
    #: A synthetic record about the memory transfer. No log row backs it.
    Transfer = "transfer"


class ImportMark(BaseModel):
    """The import mark of one record.

    ``import_id`` names the import run, and the receipt in the home's
    ``import/`` folder has the same id. ``source_sha256`` and ``source_row``
    name the CSV file and the 1-based data row (the header is row 0).
    ``synthetic`` is True when no log row states the content.
    ``imported_at`` is the naive UTC time of the import.
    """

    origin: ImportOrigin = ImportOrigin.ChatLog
    import_id: str = ""
    source_sha256: str = ""
    source_row: int = 0
    synthetic: bool = False
    imported_at: datetime = Field(default_factory=datetime.utcnow)


def import_mark(record: RecordBase) -> ImportMark | None:
    """Return the import mark of ``record``, or None for a lived record."""
    if IMPORT_MARK_KEY not in record.metadata:
        return None
    return ImportMark.model_validate(record.metadata[IMPORT_MARK_KEY])


def mark_metadata(mark: ImportMark) -> dict:
    """Return the metadata entry that stores ``mark``."""
    return {IMPORT_MARK_KEY: mark.model_dump(mode="json")}


__all__ = ["IMPORT_MARK_KEY", "ImportMark", "ImportOrigin", "import_mark", "mark_metadata"]
