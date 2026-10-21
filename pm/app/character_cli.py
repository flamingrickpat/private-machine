"""Quick start of an existing character: home folder, chat import, reconsolidation.

Commands (run with ``uv run --locked python -m pm.app.character_cli <command>``):

- ``init``: create the home folder, ``character.yaml``, ``personality.json``,
  and a ``ghost.pickle`` without history.
- ``import``: import a ``name,message,timestamp`` chat log, run memory
  reconsolidation, add the transfer records, save ``ghost.pickle``, and write
  a receipt into ``import/``.
- ``reconsolidate``: run memory reconsolidation on the saved home.

``import`` works on an in-memory copy of the store. Only a completed import
saves ``ghost.pickle``. Any error writes a failed receipt and then raises, so
no half-built checkpoint replaces the old one.

The module is runtime code. It never imports ``workflows``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from pm.app.character_home import CharacterFile, CharacterHome, parse_personality
from pm.constants import DEFAULT_SUPPORTED_CAPABILITIES, DEFAULT_UNSUPPORTED_CAPABILITIES, RECONSOLIDATION_WORKERS
from pm.memory.reconsolidation import ReconsolidationReport, reconsolidate
from pm.records.enums import FeatureType
from pm.utils import yaml_util


class CapabilityFile(BaseModel):
    """The optional ``--capabilities`` YAML file of ``init``."""

    model_config = ConfigDict(extra="forbid")

    supported_capabilities: list[str] = Field(default_factory=list)
    unsupported_capabilities: list[str] = Field(default_factory=list)
    capability_notes: list[str] = Field(default_factory=list)


class StoredImportReceipt(BaseModel):
    """The fields of a stored consolidation receipt that the duplicate check reads."""

    model_config = ConfigDict(extra="ignore")

    kind: str = ""
    source_sha256: str = ""


class ImportReceipt(BaseModel):
    """The receipt of one ``import`` run, written to ``<home>/import/<import_id>.json``.

    ``model_requests`` counts the HTTP chat requests of the run, transport
    retries included. ``reconsolidation.model_calls`` counts the validated
    calls with their repairs.
    """

    schema_id: str = "pm.character-import.v1"
    import_id: str = ""
    status: Literal["running", "completed", "failed"] = "running"
    home: str = ""
    csv_source: str = ""
    csv_copy: str = ""
    source_sha256: str = ""
    limit: int | None = None
    rows_read: int = 0
    rows_imported: int = 0
    synthesized_timestamps: bool = False
    exchanges: int = 0
    first_tick: int = 0
    last_tick: int = 0
    imported_feature_ids: list[int] = Field(default_factory=list)
    transfer_ids: list[int] = Field(default_factory=list)
    reconsolidation: ReconsolidationReport | None = None
    model_requests: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    wall_seconds: float = 0.0
    error: str = ""
    traceback: str = ""


class _CardDumper(yaml.SafeDumper):
    """Write multi-line texts as YAML block literals, so the card stays readable and editable."""


def _represent_text(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_CardDumper.add_representer(str, _represent_text)


def fill_names(text: str, companion: str, user: str) -> str:
    """Replace the ``{companion_name}`` and ``{user_name}`` placeholders of a card.

    Trailing spaces go, because YAML cannot write them in a block literal.
    """
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.replace("{companion_name}", companion).replace("{user_name}", user)


def init_home(home: CharacterHome, *, companion_name: str, user_name: str, card: Path,
              assistant_card: Path | None = None, personality: Path | None = None,
              timestamp_format: str | None = None, capabilities: Path | None = None, force: bool = False) -> None:
    """Create a character home. An existing non-empty folder raises unless ``force``.

    ``force`` deletes the old folder first, with its memory and receipts.
    The card texts may use the ``{companion_name}`` and ``{user_name}``
    placeholders. Without ``capabilities``, the home gets the defaults of
    ``pm.constants``.
    """
    root = home.root
    if root.exists() and any(root.iterdir()):
        if not force:
            raise FileExistsError(f"{root} exists and is not empty. Use --force to replace it.")
        shutil.rmtree(root)
    caps = (CapabilityFile.model_validate(yaml_util.safe_load(capabilities.read_text(encoding="utf-8")) or {})
            if capabilities is not None else
            CapabilityFile(supported_capabilities=list(DEFAULT_SUPPORTED_CAPABILITIES),
                           unsupported_capabilities=list(DEFAULT_UNSUPPORTED_CAPABILITIES)))
    character = CharacterFile(
        companion_name=companion_name, user_name=user_name,
        character_card_story=fill_names(card.read_text(encoding="utf-8-sig"), companion_name, user_name).strip() + "\n",
        character_card_assistant=(fill_names(assistant_card.read_text(encoding="utf-8-sig"), companion_name,
                                             user_name).strip() + "\n") if assistant_card is not None else None,
        supported_capabilities=[fill_names(item, companion_name, user_name) for item in caps.supported_capabilities],
        unsupported_capabilities=[fill_names(item, companion_name, user_name) for item in caps.unsupported_capabilities],
        capability_notes=[fill_names(item, companion_name, user_name) for item in caps.capability_notes],
        **({"timestamp_format": timestamp_format} if timestamp_format else {}))
    traits = parse_personality(json.loads(personality.read_text(encoding="utf-8-sig"))) if personality else None
    root.mkdir(parents=True, exist_ok=True)
    home.runtime_root.mkdir(exist_ok=True)
    home.import_dir.mkdir(exist_ok=True)
    home.character_yaml.write_text(yaml.dump(character.model_dump(mode="json"), Dumper=_CardDumper, sort_keys=False,
                                             allow_unicode=True, width=120), encoding="utf-8")
    if traits is not None:
        home.personality_json.write_text(json.dumps(traits.model_dump(), indent=2), encoding="utf-8")
    home.create_store()


def build_models(config_path: str):
    """Return ``(stage_model, embeddings)`` for a model config file.

    The stage model activates on its first call. Embedding vectors stay on
    the records and in the complete pickle checkpoint.
    """
    from pm.llm.activation import ModelActivation
    from pm.llm.config import load_config, resolve_connection
    from pm.llm.stage_model import LazyStageModel
    from pm.memory.embeddings import EmbeddingService

    config = load_config(config_path)
    stage = LazyStageModel(resolve_connection(config), ModelActivation(), max_concurrency=RECONSOLIDATION_WORKERS)
    return stage, EmbeddingService(config.embedding)


def _print(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def import_chat(home: CharacterHome, csv_path: Path, config_path: str, *, limit: int | None = None,
                do_reconsolidate: bool = True, now: datetime | None = None) -> ImportReceipt:
    """Import one chat log into the home. Return the completed receipt.

    The steps: copy the CSV into ``import/``, read and check it, write the
    exchanges, run reconsolidation, add the transfer records, embed the new
    records, and save ``ghost.pickle``. Any error writes the failed receipt
    and raises again; ``ghost.pickle`` keeps its old content then.
    """
    from pm.app.chat_import import (add_transfer_records, file_sha256, group_exchanges, import_exchanges,
                                    read_chat_csv, synthesize_times)
    started = time.monotonic()
    now = now or datetime.utcnow()
    sha = file_sha256(csv_path)
    import_id = f"import-{now.strftime('%Y%m%dT%H%M%S')}-{sha[:8]}"
    home.import_dir.mkdir(parents=True, exist_ok=True)
    copy = home.import_dir / f"{import_id}-{csv_path.name}"
    shutil.copyfile(csv_path, copy)
    receipt = ImportReceipt(import_id=import_id, home=str(home.root.resolve()), csv_source=str(csv_path.resolve()),
                            csv_copy=str(copy), source_sha256=sha, limit=limit, started_at=now)
    path = home.import_dir / f"{import_id}.json"
    # A killed process (for example by a timeout) runs no handler below.
    # The early "running" receipt then stays as evidence of the attempt.
    path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    stage = None
    try:
        store = home.open_store()
        for value in store.consolidations.values():
            stored = StoredImportReceipt.model_validate(value)
            if stored.kind == "chat_import" and stored.source_sha256 == sha:
                raise ValueError(f"{csv_path} was already imported into this home.")
        persona = store.ghost_config
        rows = read_chat_csv(csv_path, companion_name=persona.companion_name, user_name=persona.user_name)
        receipt.rows_read = len(rows)
        rows = rows[:limit] if limit is not None else rows
        receipt.synthesized_timestamps = bool(rows) and rows[0].timestamp is None
        rows = synthesize_times(rows, now)
        receipt.rows_imported = len(rows)
        with store.lock:
            lived = [item.timestamp_world_begin for item in store.all_features if item.feature_type == FeatureType.Dialogue]
        if rows and lived and rows[0].timestamp < max(lived):
            raise ValueError("The log starts before the newest dialogue of this home. Import older logs first.")
        # Work in memory. Only the final commit below writes ghost.pickle.
        store.checkpoint = None
        exchanges = group_exchanges(rows)
        _print(f"{len(rows)} rows, {len(exchanges)} exchanges")
        imported = import_exchanges(store, exchanges, import_id=import_id, source_sha256=sha,
                                    personality=home.load_personality(), imported_at=now)
        receipt.exchanges = len(exchanges)
        receipt.first_tick, receipt.last_tick = (imported.ticks[0], imported.ticks[-1]) if imported.ticks else (0, 0)
        receipt.imported_feature_ids = imported.feature_ids
        stage, embeddings = build_models(config_path)
        store.initialize_embeddings(embeddings)
        if do_reconsolidate:
            receipt.reconsolidation = reconsolidate(store, complete=stage.complete, embeddings=embeddings, force=True,
                                                    now=now, log=_print)
        receipt.transfer_ids = add_transfer_records(store, import_id=import_id, imported_at=now)
        _embed_records(store, embeddings, receipt.transfer_ids)
        receipt.status = "completed"
        receipt.model_requests = stage.request_count
        receipt.finished_at = datetime.utcnow()
        receipt.wall_seconds = round(time.monotonic() - started, 1)
        store.checkpoint = home.ghost_pickle
        batch = store.begin_batch(f"{import_id}-save")
        batch.set_consolidation_receipt(f"chat_import:{import_id}", {
            "run_id": import_id, "kind": "chat_import", "source_sha256": sha, "rows": len(rows),
            "exchanges": len(exchanges), "first_tick": receipt.first_tick, "last_tick": receipt.last_tick,
            "imported_at": now.isoformat(), "transfer_ids": receipt.transfer_ids})
        store.commit(batch)
    except BaseException as error:
        receipt.status = "failed"
        receipt.error = f"{type(error).__name__}: {error}"
        receipt.traceback = traceback.format_exc()
        receipt.model_requests = stage.request_count if stage is not None else 0
        receipt.finished_at = datetime.utcnow()
        receipt.wall_seconds = round(time.monotonic() - started, 1)
        path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        raise
    path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return receipt


def _embed_records(store, embeddings, ids: list[int]) -> None:
    """Store every missing vector on its record before the final import save."""
    import asyncio
    asyncio.run(embeddings.prepare_index(store.snapshot(set(ids)).all_knoxels))


def reconsolidate_home(home: CharacterHome, config_path: str) -> ReconsolidationReport:
    """Run reconsolidation on the saved home. Each committed group saves ``ghost.pickle``.

    The report goes to ``runtime/reconsolidation-<run_id>.json``.
    """
    store = home.open_store()
    stage, embeddings = build_models(config_path)
    store.initialize_embeddings(embeddings)
    report = reconsolidate(store, complete=stage.complete, embeddings=embeddings, force=True, log=_print)
    home.runtime_root.mkdir(parents=True, exist_ok=True)
    (home.runtime_root / f"reconsolidation-{report.run_id}.json").write_text(report.model_dump_json(indent=2),
                                                                             encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pm.app.character_cli",
                                     description="Quick start of an existing character in pm.")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a character home folder.")
    init.add_argument("--home", type=Path, required=True)
    init.add_argument("--companion-name", required=True)
    init.add_argument("--user-name", required=True)
    init.add_argument("--card", type=Path, required=True, help="Character card text (Markdown).")
    init.add_argument("--assistant-card", type=Path, default=None)
    init.add_argument("--personality", type=Path, default=None, help="personality.json with trait values 0..1.")
    init.add_argument("--timestamp-format", default=None)
    init.add_argument("--capabilities", type=Path, default=None, help="YAML with the capability lists.")
    init.add_argument("--force", action="store_true", help="Delete and replace an existing home.")
    chat = commands.add_parser("import", help="Import a chat log CSV into a home.")
    chat.add_argument("--home", type=Path, required=True)
    chat.add_argument("--csv", type=Path, required=True)
    chat.add_argument("--config", required=True, help="Model config YAML.")
    chat.add_argument("--limit", type=int, default=None, help="Import only the first N rows.")
    chat.add_argument("--no-reconsolidate", action="store_true")
    again = commands.add_parser("reconsolidate", help="Run memory reconsolidation on a home.")
    again.add_argument("--home", type=Path, required=True)
    again.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pm.host.diagnostics import configure_logging, close_logging
    from pm.host.processes import ensure_process_cleanup
    # Initialization must check the destination before creating its log folder.
    if args.command == "init":
        return _run_command(args)
    configure_logging(args.home)
    ensure_process_cleanup()
    try:
        return _run_command(args)
    except BaseException:
        import logging
        logging.getLogger(__name__).exception("Character command failed")
        raise
    finally:
        close_logging()


def _run_command(args) -> int:
    """Run the parsed character command under the caller's diagnostics."""
    home = CharacterHome(args.home)
    if args.command == "init":
        init_home(home, companion_name=args.companion_name, user_name=args.user_name, card=args.card,
                  assistant_card=args.assistant_card, personality=args.personality,
                  timestamp_format=args.timestamp_format, capabilities=args.capabilities, force=args.force)
        _print(f"Created {home.root}. Next: import a chat log, then start the app with --home {home.root}.")
    elif args.command == "import":
        receipt = import_chat(home, args.csv, args.config, limit=args.limit,
                              do_reconsolidate=not args.no_reconsolidate)
        report = receipt.reconsolidation
        _print(f"Imported {receipt.rows_imported} rows in {receipt.exchanges} exchanges"
               + (f"; {len(report.topical_ids)} topical, {len(report.temporal_ids)} temporal, "
                  f"{len(report.fact_ids)} facts, {len(report.narrative_ids)} narratives, "
                  f"{report.model_calls} model calls" if report is not None else "")
               + f"; {receipt.wall_seconds} s. Receipt: {home.import_dir / (receipt.import_id + '.json')}")
    else:
        report = reconsolidate_home(home, args.config)
        _print(f"Reconsolidation {report.run_id}: {len(report.topical_ids)} topical, {len(report.temporal_ids)} "
               f"temporal, {len(report.fact_ids)} facts, {len(report.narrative_ids)} narratives, "
               f"{report.model_calls} model calls. Notes: {report.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
