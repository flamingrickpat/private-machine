import inspect
import json
import logging
import os
import re
import shutil
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from pm.ghost.ghost_base import BaseGhost
from pm.ghost.ghost_config import GhostConfig
from pm.ghost.ghost_state import GhostState
from pm.model.knoxel_common import Stimulus, Narrative, DeclarativeFactKnoxel, MemoryClusterKnoxel, CauseEffectKnoxel, Action, Intention
from pm.model.knoxel_core import KnoxelBase
from pm.model.knoxel_feature import Feature
from pm.model.knoxel_graph import ConceptNode, GraphNode, GraphEdge
from pm.model.knoxel_list import KnoxelList
from pm.utils.serialize_utils import deserialize_embedding

logger = logging.getLogger(__name__)


def _is_optional(tp: Any) -> bool:
    origin = get_origin(tp)
    if origin is Union:
        return type(None) in get_args(tp)
    return False


def _unwrap_optional(tp: Any) -> Any:
    if not _is_optional(tp):
        return tp
    args = [x for x in get_args(tp) if x is not type(None)]
    return args[0] if args else Any


def _as_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, KnoxelBase):
        return int(value.id)
    if isinstance(value, KnoxelList):
        return [int(x.id) for x in value.to_list()]
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_as_jsonable(v) for v in value]
    return str(value)


class PersistSqlite:
    """
    Dynamic + tolerant SQLite persistence.

    Goals:
    - Schema evolves with Pydantic models (additive ALTER TABLE for new fields).
    - Lists/dicts/enums/embeddings serialized as readable JSON TEXT.
    - Knoxel links serialized as knoxel IDs.
    - Loading tolerates unknown/missing columns and parse errors.
    """

    def __init__(self, ghost: BaseGhost):
        self.ghost = ghost

    def _commit_enabled(self) -> bool:
        return self.ghost.system_config.commit

    def _snake_name(self, name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def _iter_knoxel_subclasses(self) -> Dict[str, Type[KnoxelBase]]:
        discovered: Dict[str, Type[KnoxelBase]] = {}

        def walk(cls: Type[KnoxelBase]) -> None:
            for sub in cls.__subclasses__():
                discovered[sub.__name__] = sub
                walk(sub)

        walk(KnoxelBase)

        if not discovered:
            fallback = [
                Stimulus,
                Intention,
                Action,
                MemoryClusterKnoxel,
                DeclarativeFactKnoxel,
                CauseEffectKnoxel,
                Narrative,
                Feature,
                ConceptNode,
                GraphNode,
                GraphEdge,
            ]
            return {cls.__name__: cls for cls in fallback}
        return discovered

    def _sqlite_type_for_annotation(self, annotation: Any) -> str:
        tp = _unwrap_optional(annotation)
        origin = get_origin(tp)

        if tp in (int, bool):
            return "INTEGER"
        if tp is float:
            return "REAL"
        if tp is datetime:
            return "TEXT"
        if tp is str:
            return "TEXT"
        if inspect.isclass(tp) and issubclass(tp, Enum):
            return "TEXT"
        if tp is KnoxelList:
            return "TEXT"
        if inspect.isclass(tp) and issubclass(tp, KnoxelBase):
            return "INTEGER"
        if inspect.isclass(tp) and issubclass(tp, BaseModel):
            return "TEXT"
        if origin in (list, dict, tuple, set):
            return "TEXT"
        return "TEXT"

    def _serialize(self, value: Any, annotation: Any) -> Any:
        if value is None:
            return None

        tp = _unwrap_optional(annotation)

        if tp is int:
            return int(value)
        if tp is float:
            return float(value)
        if tp is str:
            if isinstance(value, str):
                return value
            # Be tolerant to schema drifts where a field changed shape.
            if isinstance(value, (list, dict, tuple, set, BaseModel, Enum, datetime)):
                return json.dumps(_as_jsonable(value), ensure_ascii=False)
            return str(value)
        if tp is bool:
            return int(bool(value))
        if tp is datetime and isinstance(value, datetime):
            return value.isoformat()
        if inspect.isclass(tp) and issubclass(tp, Enum):
            return value.value if isinstance(value, Enum) else str(value)
        if inspect.isclass(tp) and issubclass(tp, KnoxelBase):
            return int(value.id) if isinstance(value, KnoxelBase) else int(value)

        # JSON-backed storage for all structured containers/models.
        return json.dumps(_as_jsonable(value), ensure_ascii=False)

    def _deserialize(self, db_value: Any, annotation: Any) -> Any:
        if db_value is None:
            return None

        tp = _unwrap_optional(annotation)
        origin = get_origin(tp)

        try:
            if tp is int:
                return int(db_value)
            if tp is float:
                return float(db_value)
            if tp is str:
                return str(db_value)
            if tp is bool:
                return bool(int(db_value)) if not isinstance(db_value, bool) else db_value
            if tp is datetime:
                if isinstance(db_value, datetime):
                    return db_value
                return datetime.fromisoformat(str(db_value))
            if inspect.isclass(tp) and issubclass(tp, Enum):
                return tp(db_value)
            if inspect.isclass(tp) and issubclass(tp, KnoxelBase):
                try:
                    return self.ghost.get_knoxel_by_id(int(db_value))
                except Exception:
                    return None

            if isinstance(db_value, bytes):
                return deserialize_embedding(db_value)

            if tp is KnoxelList:
                raw = self._load_json_or_fallback(db_value)
                if not isinstance(raw, list):
                    return KnoxelList([])
                items = []
                for item in raw:
                    if isinstance(item, int):
                        knx = self.ghost.get_knoxel_by_id(item)
                        if knx is not None:
                            items.append(knx)
                    elif isinstance(item, KnoxelBase):
                        items.append(item)
                return KnoxelList(items)

            if inspect.isclass(tp) and issubclass(tp, BaseModel):
                raw = self._load_json_or_fallback(db_value)
                if isinstance(raw, dict):
                    try:
                        return tp.model_validate(raw)
                    except Exception:
                        return tp()
                return tp()

            if origin in (list, tuple, set, dict):
                return self._coerce_json_container(db_value, tp)
        except Exception:
            logger.debug("Deserialize fallback for type %s with value %r", tp, db_value, exc_info=True)

        return db_value

    def _load_json_or_fallback(self, value: Any) -> Any:
        if isinstance(value, (list, dict, int, float, bool)) or value is None:
            return value
        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:
            # Backward compatibility: comma-separated int IDs.
            if "," in text and all(x.strip().lstrip("-").isdigit() for x in text.split(",") if x.strip()):
                return [int(x.strip()) for x in text.split(",") if x.strip()]
            return text

    def _coerce_json_container(self, db_value: Any, target_type: Any) -> Any:
        raw = self._load_json_or_fallback(db_value)
        origin = get_origin(target_type)
        args = get_args(target_type)

        if origin is dict:
            if not isinstance(raw, dict):
                return {}
            if len(args) == 2:
                _, vtype = args
                return {k: self._coerce_value(v, vtype) for k, v in raw.items()}
            return raw

        if origin in (list, tuple, set):
            if not isinstance(raw, list):
                return [] if origin is list else tuple()
            if args:
                item_t = args[0]
                coerced = [self._coerce_value(x, item_t) for x in raw]
            else:
                coerced = raw
            if origin is tuple:
                return tuple(coerced)
            if origin is set:
                return set(coerced)
            return coerced

        return raw

    def _coerce_value(self, value: Any, expected_type: Any) -> Any:
        tp = _unwrap_optional(expected_type)
        origin = get_origin(tp)

        try:
            if tp is int:
                return int(value)
            if tp is float:
                return float(value)
            if tp is bool:
                return bool(value)
            if tp is str:
                return str(value)
            if tp is datetime:
                return datetime.fromisoformat(str(value))
            if inspect.isclass(tp) and issubclass(tp, Enum):
                return tp(value)
            if inspect.isclass(tp) and issubclass(tp, KnoxelBase):
                if isinstance(value, int):
                    return self.ghost.get_knoxel_by_id(value)
                return None
            if inspect.isclass(tp) and issubclass(tp, BaseModel):
                if isinstance(value, dict):
                    return tp.model_validate(value)
                return tp()
            if origin in (list, tuple, set, dict):
                return self._coerce_json_container(value, tp)
        except Exception:
            return None

        return value

    def _ensure_table(self, cursor: sqlite3.Cursor, table_name: str, columns: Dict[str, str], primary_key: Optional[str] = None) -> None:
        cursor.execute(f"PRAGMA table_info({table_name});")
        info = cursor.fetchall()
        existing_cols = {row[1] for row in info}

        if not info:
            col_defs = []
            for name, ctype in columns.items():
                suffix = ""
                if primary_key and name == primary_key:
                    suffix = " PRIMARY KEY"
                col_defs.append(f"{name} {ctype}{suffix}")
            sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)});"
            cursor.execute(sql)
            return

        for col, ctype in columns.items():
            if col in existing_cols:
                continue
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {ctype};")

    def _upsert_key_value_table(self, cursor: sqlite3.Cursor, table: str, values: Dict[str, str]) -> None:
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} (key TEXT PRIMARY KEY, value TEXT);")
        cursor.execute(f"DELETE FROM {table};")
        for k, v in values.items():
            cursor.execute(f"INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?);", (k, v))

    def _model_columns(self, model_cls: Type[BaseModel], primary_key: Optional[str] = None) -> Dict[str, str]:
        cols = {}
        for name, field in model_cls.model_fields.items():
            cols[name] = self._sqlite_type_for_annotation(field.annotation)
        if primary_key and primary_key not in cols:
            cols[primary_key] = "INTEGER"
        return cols

    def _table_name_candidates(self, knoxel_cls: Type[KnoxelBase]) -> List[str]:
        # New canonical snake_case first, then legacy lower name.
        snake = self._snake_name(knoxel_cls.__name__)
        legacy = knoxel_cls.__name__.lower()
        if snake == legacy:
            return [snake]
        return [snake, legacy]

    def save_state_sqlite(self, filename: str) -> None:
        if not self._commit_enabled():
            logger.info("Persistence disabled via config.commit=False")
            return

        logger.info("Saving ghost state to SQLite: %s", filename)

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except Exception:
            pass

        try:
            bak_path = filename + f".tick{self.ghost.current_tick_id - 1}.db"
            if os.path.isfile(bak_path):
                os.remove(bak_path)
            if os.path.isfile(filename):
                shutil.copyfile(filename, bak_path)
        except Exception:
            logger.debug("Backup creation failed", exc_info=True)

        conn = sqlite3.connect(filename)
        try:
            cursor = conn.cursor()

            # Metadata/config tables.
            meta = {
                "current_tick_id": str(getattr(self.ghost, "current_tick_id", 0)),
                "current_knoxel_id": str(getattr(self.ghost, "current_knoxel_id", 0)),
            }
            self._upsert_key_value_table(cursor, "metadata", meta)

            cfg = {}
            for key, value in self.ghost.ghost_config.model_dump(mode="json").items():
                cfg[key] = json.dumps(_as_jsonable(value), ensure_ascii=False)
            self._upsert_key_value_table(cursor, "config", cfg)

            # Dynamic knoxel tables.
            subclasses = self._iter_knoxel_subclasses()
            schema_by_cls: Dict[Type[KnoxelBase], Dict[str, str]] = {}
            table_for_cls: Dict[Type[KnoxelBase], str] = {}
            for cls in subclasses.values():
                table_name = self._snake_name(cls.__name__)
                table_for_cls[cls] = table_name
                cols = self._model_columns(cls, primary_key="id")
                schema_by_cls[cls] = cols
                self._ensure_table(cursor, table_name, cols, primary_key="id")

            # Dynamic ghost_states table.
            state_table = "ghost_states"
            state_cols = self._model_columns(GhostState, primary_key="tick_id")
            self._ensure_table(cursor, state_table, state_cols, primary_key="tick_id")

            # Replace all managed rows with current in-memory state.
            for table_name in set(table_for_cls.values()):
                cursor.execute(f"DELETE FROM {table_name};")
            cursor.execute(f"DELETE FROM {state_table};")

            for knx in self.ghost.all_knoxels.values():
                cls = type(knx)
                if cls not in table_for_cls:
                    logger.warning("Skipping unknown knoxel type during save: %s", cls.__name__)
                    continue

                table = table_for_cls[cls]
                cols = []
                vals = []
                payload = knx.model_dump(mode="python")
                for name, field in cls.model_fields.items():
                    cols.append(name)
                    vals.append(self._serialize(payload.get(name), field.annotation))

                placeholders = ", ".join(["?"] * len(cols))
                sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders});"
                cursor.execute(sql, tuple(vals))

            for state in self.ghost.states:
                cols = []
                vals = []
                payload = state.model_dump(mode="python")
                for name, field in GhostState.model_fields.items():
                    cols.append(name)
                    vals.append(self._serialize(payload.get(name), field.annotation))
                placeholders = ", ".join(["?"] * len(cols))
                sql = f"INSERT OR REPLACE INTO {state_table} ({', '.join(cols)}) VALUES ({placeholders});"
                cursor.execute(sql, tuple(vals))

            conn.commit()
            self.ghost.current_db_path = os.path.abspath(filename)
            logger.info("Saved ghost SQLite state with dynamic schema migration.")
        finally:
            conn.close()

    def load_state_sqlite(self, filename: str) -> bool:
        logger.info("Loading ghost state from SQLite: %s", filename)

        if not os.path.exists(filename):
            logger.warning("SQLite file does not exist: %s", filename)
            return False

        try:
            conn = sqlite3.connect(filename)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            logger.exception("Could not open SQLite DB")
            return False

        try:
            cursor = conn.cursor()
            self.ghost._reset_internal_state()

            # Metadata
            try:
                cursor.execute("SELECT key, value FROM metadata;")
                meta = {row["key"]: row["value"] for row in cursor.fetchall()}
                self.ghost.current_tick_id = int(meta.get("current_tick_id", 0) or 0)
                self.ghost.current_knoxel_id = int(meta.get("current_knoxel_id", 0) or 0)
            except Exception:
                logger.warning("Metadata table missing or malformed; using defaults.", exc_info=True)

            # Config (ignore unknown keys, fallback to defaults on parse/validation errors)
            config_data: Dict[str, Any] = {}
            cursor.execute("SELECT key, value FROM config;")
            cfg_rows = cursor.fetchall()
            for row in cfg_rows:
                key = row["key"]
                raw = row["value"]
                if key not in GhostConfig.model_fields:
                    continue
                ann = GhostConfig.model_fields[key].annotation
                try:
                    decoded = self._load_json_or_fallback(raw)
                    if isinstance(decoded, str) and ann is not str:
                        # scalar fallback path
                        decoded = self._deserialize(decoded, ann)
                    else:
                        decoded = self._coerce_value(decoded, ann)
                        if decoded is None:
                            decoded = self._deserialize(raw, ann)
                    config_data[key] = decoded
                except Exception:
                    logger.debug("Failed to load config field %s; using default.", key, exc_info=True)
            self.ghost.ghost_config = GhostConfig(**config_data)

            # Load knoxels.
            subclasses = self._iter_knoxel_subclasses()
            total_loaded = 0
            for cls in subclasses.values():
                table_name = None
                for candidate in self._table_name_candidates(cls):
                    try:
                        cursor.execute(f"SELECT 1 FROM {candidate} LIMIT 1;")
                        table_name = candidate
                        break
                    except sqlite3.Error:
                        continue

                if table_name is None:
                    continue

                try:
                    cursor.execute(f"SELECT * FROM {table_name};")
                    rows = cursor.fetchall()
                except sqlite3.Error:
                    logger.warning("Failed reading table %s", table_name, exc_info=True)
                    continue

                for row in rows:
                    row_map = dict(row)
                    payload: Dict[str, Any] = {}
                    for fname, field in cls.model_fields.items():
                        if fname not in row_map:
                            continue
                        try:
                            payload[fname] = self._deserialize(row_map[fname], field.annotation)
                        except Exception:
                            logger.debug(
                                "Field deserialize failed for %s.%s id=%s",
                                cls.__name__,
                                fname,
                                row_map.get("id", "?"),
                                exc_info=True,
                            )

                    if "id" not in payload:
                        logger.warning("Skipping row without id in table %s", table_name)
                        continue

                    try:
                        obj = cls(**payload)
                        self.ghost.all_knoxels[obj.id] = obj
                        total_loaded += 1
                    except ValidationError:
                        logger.warning(
                            "Skipping invalid %s row id=%s; payload did not validate.",
                            cls.__name__,
                            payload.get("id"),
                        )
                    except Exception:
                        logger.warning(
                            "Skipping broken %s row id=%s due to unexpected error.",
                            cls.__name__,
                            payload.get("id"),
                            exc_info=True,
                        )

            self.ghost._rebuild_specific_lists()
            logger.info("Loaded %d knoxels.", total_loaded)

            # Load ghost states.
            self.ghost.states = []
            try:
                cursor.execute("SELECT * FROM ghost_states ORDER BY tick_id ASC;")
                for row in cursor.fetchall():
                    row_map = dict(row)
                    payload: Dict[str, Any] = {}
                    for fname, field in GhostState.model_fields.items():
                        if fname not in row_map:
                            continue
                        try:
                            payload[fname] = self._deserialize(row_map[fname], field.annotation)
                        except Exception:
                            logger.debug("GhostState field deserialize failed: %s", fname, exc_info=True)

                    try:
                        state = GhostState(**payload)
                        self.ghost.states.append(state)
                    except ValidationError:
                        logger.warning(
                            "Skipping invalid GhostState tick=%s", payload.get("tick_id", "?"),
                        )
                    except Exception:
                        logger.warning(
                            "Skipping broken GhostState tick=%s", payload.get("tick_id", "?"),
                            exc_info=True,
                        )
            except sqlite3.Error:
                logger.warning("ghost_states table missing or unreadable; continuing without historical states.")

            self.ghost.states.sort(key=lambda x: x.tick_id)
            self.ghost.current_db_path = os.path.abspath(filename)
            return True
        except Exception:
            logger.exception("Unexpected error during SQLite load")
            self.ghost._reset_internal_state()
            return False
        finally:
            conn.close()

    # Compatibility utility for legacy callers/tests.
    def _get_knoxel_type_map(self) -> Dict[str, Type[KnoxelBase]]:
        return {
            cls.__name__: cls
            for cls in [
                KnoxelBase,
                Stimulus,
                Intention,
                Action,
                MemoryClusterKnoxel,
                DeclarativeFactKnoxel,
                Narrative,
                Feature,
                ConceptNode,
                GraphNode,
                GraphEdge,
            ]
        }
