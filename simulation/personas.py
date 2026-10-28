"""Discover persona homes and fork trusted checkpoints without changing sources."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import yaml

from pm.app.character_home import CharacterFile, CharacterHome
from pm.host.durable_files import atomic_json
from pm.memory.store import MemoryStore
from pm.paths import REPO_ROOT
from pm.records.mental_state import PersonalityProfile


DEFAULT_HOMES = Path.home() / ".pm" / "private-machine"
DEFAULT_RUNS = REPO_ROOT / ".runtime" / "simulation"
CANON = REPO_ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"


def homes(root: Path) -> list[dict]:
    """List character homes. Cache, run, and log folders are not personas."""
    result = []
    for card in sorted(root.rglob("character.yaml")) if root.exists() else []:
        home = CharacterHome(card.parent)
        if home.ghost_pickle.exists():
            persona = home.load_character()
            result.append({"id": str(home.root.resolve()), "name": persona.companion_name,
                           "folder": home.root.name, "user": persona.user_name})
    return result


def write_home(store: MemoryStore, destination: Path) -> CharacterHome:
    """Create a new home from a complete store. Existing homes raise FileExistsError."""
    destination.mkdir(parents=True, exist_ok=False)
    home = CharacterHome(destination)
    config = store.ghost_config.model_dump(mode="json")
    card = CharacterFile.model_validate({key: config[key] for key in CharacterFile.model_fields})
    home.character_yaml.write_text(yaml.safe_dump(card.model_dump(mode="json"), allow_unicode=True, sort_keys=False), encoding="utf-8")
    traits = PersonalityProfile.init_from_vector(store.ghost_config.personality_profile)
    atomic_json(home.personality_json, traits.model_dump())
    store.checkpoint = home.ghost_pickle
    store.save()
    return home


def canon_home(destination: Path, *, cache_root: Path = DEFAULT_RUNS / "seeds") -> CharacterHome:
    """Load the complete synthetic Spamuel canon and cache its trusted conversion."""
    digest = hashlib.sha256(CANON.read_bytes()).hexdigest()
    cached = cache_root / f"spamuel-{digest}.pickle"
    if cached.exists():
        store = MemoryStore.from_checkpoint(cached)
    else:
        store = MemoryStore.from_yaml(CANON, checkpoint=cached)
        store.save()
    return write_home(store, destination)


def fork_home(source: Path, destination: Path, *, snapshot: Path | None = None) -> CharacterHome:
    """Fork the clean causal view. Preserve the original home and raw history."""
    home = CharacterHome(source)
    checkpoint = snapshot or home.ghost_pickle
    store = MemoryStore.from_checkpoint(checkpoint)
    created = write_home(store, destination)
    shutil.copyfile(home.character_yaml, created.character_yaml)
    if home.personality_json.exists():
        shutil.copyfile(home.personality_json, created.personality_json)
    clock = (home.root / "runtime/clock.json" if checkpoint.resolve() == home.ghost_pickle.resolve() else
             checkpoint.with_suffix(".json") if checkpoint.parent.name == "snapshots" else None)
    if clock is not None and clock.exists():
        target = created.root / "runtime/clock.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clock, target)
    return created
