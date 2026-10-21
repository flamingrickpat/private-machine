"""The character home folder: one folder holds one companion.

A home folder has this layout::

    <home>/
      character.yaml    # PersonaConfig fields without personality_profile
      personality.json  # {"<PersonalityProfile field>": 0..1, ...}
      ghost.pickle      # complete accepted state, including embeddings
      ghost_*.pickle    # complete backup after each successful main cognition commit
      logs/             # process and thread diagnostics
      runtime/          # runtime root of the app
      import/           # copies of imported CSV files and import receipts

``character init`` creates the folder. The app opens it with
``CharacterHome(root).open_store()`` and saves to ``ghost_pickle``.
The persona files are the source of truth for the card text and the traits.
The checkpoint is the source of truth for memory.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.mental_state import PersonalityProfile
from pm.utils import yaml_util


class CharacterFile(BaseModel):
    """The fields of ``character.yaml``.

    These are the ``PersonaConfig`` fields without ``personality_profile``.
    The traits live in ``personality.json``. Unknown keys raise a
    validation error, so a typo does not silently drop a setting.
    """

    model_config = ConfigDict(extra="forbid")

    companion_name: str = ""
    user_name: str = ""
    timestamp_format: str = "%Y-%m-%d %H:%M"
    character_card_story: str = ""
    character_card_assistant: str | None = None
    supported_capabilities: list[str] = Field(default_factory=list)
    unsupported_capabilities: list = Field(default_factory=list)
    capability_notes: list = Field(default_factory=list)


def parse_personality(data: dict) -> PersonalityProfile:
    """Validate user-written trait values and return the profile.

    Unknown keys raise ``ValueError`` with the allowed names. Missing keys
    keep the profile default 0.5. Values outside 0..1 raise a pydantic
    ``ValidationError``.
    """
    allowed = sorted(PersonalityProfile.model_fields)
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown personality keys {unknown}. Allowed keys: {allowed}")
    return PersonalityProfile.model_validate(data)


class CharacterHome(BaseModel):
    """The paths and the loaders of one character home folder.

    Construct it with ``CharacterHome(root=Path(...))`` or
    ``CharacterHome(Path(...))``. The constructor reads no file.
    """

    model_config = ConfigDict(frozen=True)

    root: Path = Path(".")

    def __init__(self, root: Path | str = Path("."), **data):
        super().__init__(root=Path(root), **data)

    @property
    def character_yaml(self) -> Path:
        return self.root / "character.yaml"

    @property
    def personality_json(self) -> Path:
        return self.root / "personality.json"

    @property
    def ghost_json(self) -> Path:
        return self.root / "ghost.json"

    @property
    def ghost_pickle(self) -> Path:
        """Return the complete binary checkpoint path."""
        return self.root / "ghost.pickle"

    @property
    def runtime_root(self) -> Path:
        return self.root / "runtime"

    @property
    def import_dir(self) -> Path:
        return self.root / "import"

    def load_character(self) -> CharacterFile:
        """Read ``character.yaml``. A missing file raises FileNotFoundError."""
        return CharacterFile.model_validate(yaml_util.safe_load(self.character_yaml.read_text(encoding="utf-8")) or {})

    def load_personality(self) -> PersonalityProfile:
        """Read ``personality.json``. A missing file gives the neutral profile."""
        if not self.personality_json.exists():
            return PersonalityProfile()
        return parse_personality(json.loads(self.personality_json.read_text(encoding="utf-8-sig")))

    def load_persona(self) -> PersonaConfig:
        """Return the persona of ``character.yaml`` with the traits of ``personality.json``."""
        character = self.load_character()
        return PersonaConfig(**character.model_dump(),
                             personality_profile=self.load_personality().to_vector())

    def open_store(self) -> MemoryStore:
        """Load the accepted checkpoint and apply the persona files.

        The store saves later commits to ``ghost_pickle``. Legacy JSON remains readable. The card text,
        the capabilities, and the traits come from the persona files, so
        an edit there takes effect at the next start. A changed companion
        or user name raises ValueError, because the stored entities and
        dialogue name the old speakers.
        """
        checkpoint = self.ghost_pickle if self.ghost_pickle.exists() else self.ghost_json
        if not checkpoint.exists():
            raise FileNotFoundError(f"{self.ghost_pickle} does not exist. Run `character init` first.")
        store = MemoryStore.from_checkpoint(checkpoint)
        persona = self.load_persona()
        stored = store.ghost_config
        if (persona.companion_name, persona.user_name) != (stored.companion_name, stored.user_name):
            raise ValueError(
                f"character.yaml names {persona.companion_name!r}/{persona.user_name!r}, "
                f"but the checkpoint holds {stored.companion_name!r}/{stored.user_name!r}.")
        store.ghost_config = persona
        return store

    def create_store(self) -> MemoryStore:
        """Write a new checkpoint for the persona and return its store.

        The store has no history: no dialogue and no cycles. It holds only
        the entities of the user and the companion, because the live loop
        finds both speakers by entity. The commit writes the checkpoint
        through the normal commit path.
        """
        from pm.app.chat_import import ensure_persona_entities
        from pm.records.import_marks import ImportMark, ImportOrigin

        store = MemoryStore(self.load_persona(), checkpoint=self.ghost_pickle)
        store.commit(store.begin_batch("character-home-init"))
        ensure_persona_entities(store, ImportMark(origin=ImportOrigin.CharacterInit, import_id="character-init",
                                                  synthetic=True))
        return store
