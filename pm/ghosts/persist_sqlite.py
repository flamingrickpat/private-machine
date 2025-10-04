import inspect
import json
import logging
import os
import re
import shutil
import sqlite3
from datetime import datetime
from enum import Enum
from enum import StrEnum, IntEnum
from json import JSONEncoder
from typing import Dict, Any
from typing import List
from typing import Type
from typing import (
    Union,
    get_origin,
)
from typing import get_args

from pydantic import ValidationError

from pm.config_loader import commit
from pm.data_structures import KnoxelBase, Feature, Stimulus, Intention, Narrative, Action, MemoryClusterKnoxel, DeclarativeFactKnoxel, KnoxelList, ConceptNode, GraphNode, GraphEdge
from pm.ghosts.base_ghost import BaseGhost, GhostState, GhostConfig
from pm.mental_states import EmotionalAxesModel, CognitionAxesModel, NeedsAxesModel, ClampedModel
from pm.utils.serialize_utils import deserialize_embedding, serialize_embedding
from pm.utils.string_utils import to_camel_case


def _default(self, obj):
    try:
        return obj.to_json()
    except:
        try:
            return str(obj.id)
        except:
            return f"{obj}"


_default.default = JSONEncoder().default
JSONEncoder.default = _default

logger = logging.getLogger(__name__)


class PersistSqlite:
    def __init__(self, ghost: BaseGhost):
        self.ghost = ghost

    def _get_knoxel_subclasses(self) -> Dict[str, Type[KnoxelBase]]:
        """Finds all subclasses of KnoxelBase defined in the current scope."""
        # This might need adjustment based on where KnoxelBase and its subclasses are defined.
        # Assuming they are in the same module or imported.
        subclasses = {}
        # Check classes defined in the same module as Ghost
        for name, obj in inspect.getmembers(inspect.getmodule(inspect.currentframe())):
            if inspect.isclass(obj) and issubclass(obj, KnoxelBase) and obj is not KnoxelBase:
                # Use class name as key, map to the actual class type
                subclasses[obj.__name__] = obj
        # Add KnoxelBase itself if needed for some reason (usually not for tables)
        # subclasses[KnoxelBase.__name__] = KnoxelBase
        if not subclasses:
            logging.warning("Could not dynamically find any KnoxelBase subclasses. DB operations might fail.")
            # Fallback to known types if inspection fails
            return {
                cls.__name__: cls for cls in
                [Stimulus, Intention, Action, MemoryClusterKnoxel, DeclarativeFactKnoxel, Narrative, Feature, ConceptNode, GraphNode, GraphEdge]
            }

        return subclasses

    def _pydantic_type_to_sqlite_type(self, field_info: Any) -> str:
        """Maps a Pydantic field's type annotation to an SQLite type string."""
        field_type = field_info.annotation
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional types
        if origin is Union and type(None) in args:
            # Get the non-None type
            field_type = next(t for t in args if t is not type(None))
            origin = get_origin(field_type)  # Re-evaluate origin for List[Optional[...]] etc.
            args = get_args(field_type)
            # Nullability is handled by default in SQLite columns unless PRIMARY KEY or NOT NULL specified

        # Basic types
        if field_type is int: return "INTEGER"
        if field_type is float: return "REAL"
        if field_type is str: return "TEXT"
        if field_type is bool: return "INTEGER"  # Store as 0 or 1
        if field_type is datetime: return "TEXT"  # Store as ISO format string

        # Enums
        if inspect.isclass(field_type) and issubclass(field_type, (StrEnum, IntEnum)):
            return "TEXT" if issubclass(field_type, StrEnum) else "INTEGER"

        if inspect.isclass(field_type) and issubclass(field_type, ClampedModel):
            return "TEXT"

        # Lists
        if origin is list or field_type is List or "KnoxelList" in str(field_type):
            if args:
                list_item_type = args[0]
                # Special case for embeddings
                if list_item_type is float:
                    return "BLOB"  # Store List[float] embeddings as BLOB
            # Store other lists (like list of int IDs) as JSON strings
            return "TEXT"

        # Dictionaries
        if origin is dict or field_type is Dict:
            return "TEXT"  # Store dicts as JSON strings

        # References to other Knoxels (by ID) - These shouldn't be direct fields in Knoxel tables itself usually
        # but might appear in GhostState. We store the ID.
        if inspect.isclass(field_type) and issubclass(field_type, KnoxelBase):
            return "INTEGER"

        # Default / Fallback
        logging.warning(f"Unknown Pydantic type for SQLite mapping: {field_type}. Using TEXT as fallback.")
        return "TEXT"

    def _serialize_value_for_db(self, value: Any, field_type: Type) -> Any:
        """Converts Python value to DB storable format based on Pydantic type."""
        if value is None:
            return None

        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional correctly
        if origin is Union and type(None) in args:
            if value is None: return None
            field_type = next(t for t in args if t is not type(None))
            origin = get_origin(field_type)  # Re-evaluate origin
            args = get_args(field_type)

        # Basic types that are already compatible or need simple conversion
        if isinstance(value, (int, float, str)): return value
        if isinstance(value, bool): return 1 if value else 0
        if isinstance(value, datetime): return value.isoformat()

        # Enums
        if isinstance(value, (StrEnum, IntEnum)): return value.value

        # Embeddings (List[float])
        if (origin is list or isinstance(value, list)) and args and args[0] is float:
            # Check if it looks like an embedding before serializing
            if all(isinstance(x, float) for x in value):
                return serialize_embedding(value)
            else:
                logging.warning(
                    f"Expected List[float] for embedding, got list with other types: {type(value[0]) if value else 'empty'}. Serializing as JSON.")
                # Fallback to JSON for non-float lists
                return json.dumps(value)

        # Other Lists or Dicts
        if isinstance(value, (list, dict)):
            return json.dumps(value)

        # Knoxel References (store ID) - relevant for GhostState, not Knoxel tables usually
        if isinstance(value, KnoxelBase): return value.id

        # id list
        if isinstance(value, KnoxelList):
            return ",".join([str(x.id) for x in value])

        logging.warning(f"Could not serialize type {type(value)} for DB. Returning str(value).")
        return str(value)  # Fallback

    def _deserialize_value_from_db(self, db_value: Any, target_type: Type) -> Any:
        """Converts DB value back to Python type based on Pydantic annotation."""
        if db_value is None:
            return None

        origin = get_origin(target_type)
        args = get_args(target_type)
        is_optional = False

        if origin is Union and type(None) in args:
            is_optional = True
            if db_value is None: return None
            target_type = next(t for t in args if t is not type(None))  # Get the actual type
            origin = get_origin(target_type)  # Re-evaluate origin
            args = get_args(target_type)

        # id knoxel ref
        if issubclass(target_type, KnoxelBase):
            _id = db_value
            if isinstance(db_value, str):
                _id = int(db_value)
            return self.ghost.get_knoxel_by_id(_id)

        # Basic types
        if target_type is int: return int(db_value) if db_value is not None else (None if is_optional else 0)
        if target_type is float: return float(db_value) if db_value is not None else (None if is_optional else 0.0)
        if target_type is str: return str(db_value) if db_value is not None else (None if is_optional else "")
        if target_type is bool: return bool(db_value) if db_value is not None else (None if is_optional else False)
        if target_type is datetime:
            try:
                return datetime.fromisoformat(db_value) if isinstance(db_value, str) else (
                    None if is_optional else datetime.now())  # Provide default or handle error?
            except (ValueError, TypeError):
                return None if is_optional else datetime.now()  # Fallback

        # Enums
        if inspect.isclass(target_type) and issubclass(target_type, Enum):
            try:
                return target_type(db_value)
            except ValueError:
                return None if is_optional else list(target_type)[0]  # Fallback to first enum member or None

        # Embeddings (List[float])
        if (origin is list or target_type is List) and args and args[0] is float:
            if isinstance(db_value, bytes):
                return deserialize_embedding(db_value)
            else:
                logging.warning(f"Expected bytes for embedding BLOB, got {type(db_value)}. Returning empty list.")
                return []

        # Other Lists or Dicts (from JSON text)
        if (origin is list or target_type is List or target_type is KnoxelList) and isinstance(db_value, str):
            try:
                res = []
                if db_value != "":
                    try:
                        tmp = json.loads(db_value)
                    except:
                        tmp = list(map(int, db_value.split(',')))
                    for val in tmp:
                        if isinstance(val, int) and (len(args) == 0 or args[0] != int):
                            res.append(self.ghost.get_knoxel_by_id(val))
                        else:
                            res.append(val)
                if target_type is KnoxelList:
                    return KnoxelList(res)
                else:
                    return res
            except json.JSONDecodeError:
                logging.warning(f"Failed to decode JSON for {target_type}: {db_value[:100]}...")
                return None if is_optional else ([] if origin is list else {})  # Fallback

        if (origin is dict or target_type is Dict) and isinstance(db_value, str):
            try:
                tmp = json.loads(db_value)
                res_dict = {}
                for key, val in tmp.items():
                    if isinstance(val, list):
                        res = []
                        for sub_val in val:
                            if isinstance(sub_val, int):
                                res.append(self.ghost.get_knoxel_by_id(sub_val))
                            else:
                                res.append(sub_val)
                        res_dict[key] = res
                    else:
                        res_dict[key] = val
                return res_dict

            except json.JSONDecodeError:
                logging.warning(f"Failed to decode JSON for {target_type}: {db_value[:100]}...")
                return None if is_optional else ([] if origin is list else {})  # Fallback

        # Knoxel references (return ID - reconstruction happens later)
        if inspect.isclass(target_type) and issubclass(target_type, KnoxelBase):
            return int(db_value) if db_value is not None else None

        logging.warning(f"Could not deserialize DB value '{db_value}' to type {target_type}. Returning raw value.")
        return db_value  # Fallback

    # --- Dynamic SQLite Persistence ---

    def save_state_sqlite(self, filename: str):
        """Saves the complete Ghost state to an SQLite database dynamically."""
        if not commit:
            logging.info(f"Rolling back...")
            return

        logging.info(f"Dynamically saving state to SQLite database: {filename}...")

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except:
            pass

        try:
            bak_path = filename + f".tick{self.ghost.current_tick_id - 1}.db"
            if os.path.isfile(bak_path):
                os.remove(bak_path)

            if os.path.isfile(filename):
                shutil.copyfile(filename, bak_path)
        except:
            pass

        conn = None
        if True:
            conn = sqlite3.connect(filename)
            cursor = conn.cursor()

            # --- Define Tables Dynamically ---
            knoxel_subclasses = self._get_knoxel_subclasses()
            knoxel_schemas = {}  # Store {Type: (table_name, columns)}
            ghost_state_columns = {}  # Store {field_name: sqlite_type}

            # 1. Knoxel Tables
            for knoxel_cls in knoxel_subclasses.values():
                table_name = re.sub(r'([a-z])([A-Z])', r'\1_\2', knoxel_cls.__name__.lower()).lower()
                columns = {}
                for field_name, field_info in knoxel_cls.model_fields.items():
                    sqlite_type = self._pydantic_type_to_sqlite_type(field_info)
                    columns[field_name] = sqlite_type
                knoxel_schemas[knoxel_cls] = (table_name, columns)

            # 2. GhostState Table (Flatten nested models)
            state_table_name = "ghost_states"
            base_state_columns = {}
            nested_state_models = {
                'state_emotions': EmotionalAxesModel,
                'state_needs': NeedsAxesModel,
                'state_cognition': CognitionAxesModel
            }
            for field_name, field_info in GhostState.model_fields.items():
                if field_name not in nested_state_models:  # Skip nested model fields themselves
                    base_state_columns[field_name] = self._pydantic_type_to_sqlite_type(field_info)

            # Add flattened columns from nested models
            for prefix, model_cls in nested_state_models.items():
                for field_name, field_info in model_cls.model_fields.items():
                    col_name = f"{prefix.split('_')[-1]}_{field_name}"  # e.g., emotion_valence
                    base_state_columns[col_name] = self._pydantic_type_to_sqlite_type(field_info)
            ghost_state_columns = base_state_columns

            # --- Drop Existing Tables ---
            all_tables_to_drop = ['metadata', 'config'] + [name for name, _ in knoxel_schemas.values()] + [
                state_table_name]
            for table_name in all_tables_to_drop:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
            logging.debug("Dropped existing tables.")

            # --- Create New Tables ---
            # Metadata table
            cursor.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);")
            # Config table
            cursor.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);")
            # Knoxel tables
            for _, (table_name, columns) in knoxel_schemas.items():
                cols_sql = ",\n    ".join(
                    [f"{name} {ctype}" + (" PRIMARY KEY" if name == "id" else "") for name, ctype in columns.items()])
                create_sql = f"CREATE TABLE {table_name} (\n    {cols_sql}\n);"
                logging.debug(f"Creating table {table_name}:\n{create_sql}")
                cursor.execute(create_sql)
            # Ghost state table
            cols_sql = ",\n    ".join(
                [f"{name} {ctype}" + (" PRIMARY KEY" if name == "tick_id" else "") for name, ctype in
                 ghost_state_columns.items()])
            create_state_sql = f"CREATE TABLE {state_table_name} (\n    {cols_sql}\n);"
            logging.debug(f"Creating table {state_table_name}:\n{create_state_sql}")
            cursor.execute(create_state_sql)

            logging.info("Created tables dynamically based on Pydantic models.")

            # --- Insert Metadata & Config (same as before) ---
            cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                           ('current_tick_id', str(self.ghost.current_tick_id)))
            cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                           ('current_knoxel_id', str(self.ghost.current_knoxel_id)))
            config_data = self.ghost.config.model_dump()
            for key, value in config_data.items():
                value_str = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
                cursor.execute("INSERT INTO config (key, value) VALUES (?, ?)", (key, value_str))

            # --- Insert Knoxels Dynamically ---
            for knoxel in self.ghost.all_knoxels.values():
                knoxel_type = type(knoxel)
                if knoxel_type not in knoxel_schemas:
                    logging.warning(f"Skipping knoxel {knoxel.id} - No schema found for type {knoxel_type.__name__}.")
                    continue

                table_name, columns_schema = knoxel_schemas[knoxel_type]
                data_dict = knoxel.model_dump()

                # Prepare data and map field names to column names (they should match)
                insert_data = {}
                values_list = []
                cols_list = []

                for field_name, field_info in knoxel_type.model_fields.items():
                    if field_name in columns_schema:  # Ensure the field exists in our table schema
                        raw_value = data_dict.get(field_name)
                        serialized_value = self._serialize_value_for_db(raw_value, field_info.annotation)
                        insert_data[field_name] = serialized_value
                        cols_list.append(field_name)
                        values_list.append(serialized_value)

                if not cols_list: continue  # Skip if no columns to insert

                cols_str = ', '.join(cols_list)
                placeholders = ', '.join(['?'] * len(cols_list))
                sql = f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders});"

                try:
                    cursor.execute(sql, tuple(values_list))
                except sqlite3.Error as e:
                    logging.error(f"SQLite error inserting into {table_name} (Knoxel {knoxel.id}): {e}")
                    logging.error(f"SQL: {sql}")
                    logging.error(f"Values: {values_list}")

            # --- Insert Ghost States Dynamically ---
            for state in self.ghost.states:
                state_insert_data = {}
                values_list = []
                cols_list = []

                # Handle base GhostState fields
                for field_name, field_info in GhostState.model_fields.items():
                    col_name = field_name
                    value = getattr(state, field_name, None)

                    if field_name in nested_state_models:  # Skip the direct nested model fields
                        continue

                    # Ensure column exists in our dynamic schema before adding
                    if col_name in ghost_state_columns:
                        serialized_value = self._serialize_value_for_db(value, field_info.annotation)
                        state_insert_data[col_name] = serialized_value
                        cols_list.append(col_name)
                        values_list.append(serialized_value)

                # Handle flattened nested models
                for prefix, model_cls in nested_state_models.items():
                    model_instance = getattr(state, prefix, None)
                    if model_instance:
                        for field_name, field_info in model_cls.model_fields.items():
                            col_name = f"{prefix.split('_')[-1]}_{field_name}"  # e.g., emotion_valence
                            if col_name in ghost_state_columns:  # Check if column exists
                                raw_value = getattr(model_instance, field_name, None)
                                serialized_value = self._serialize_value_for_db(raw_value, field_info.annotation)
                                state_insert_data[col_name] = serialized_value
                                cols_list.append(col_name)
                                values_list.append(serialized_value)

                cols_str = ', '.join(cols_list)
                placeholders = ', '.join(['?'] * len(cols_list))
                sql = f"INSERT INTO {state_table_name} ({cols_str}) VALUES ({placeholders});"

                try:
                    cursor.execute(sql, tuple(values_list))
                except sqlite3.Error as e:
                    logging.error(f"SQLite error inserting into {state_table_name} (Tick {state.tick_id}): {e}")
                    logging.error(f"SQL: {sql}")
                    logging.error(f"Values: {values_list}")

            # --- Commit ---
            conn.commit()
            self.ghost.current_db_path = os.path.abspath(filename)
            logging.info(f"State dynamically saved to SQLite database: {filename}")


    def load_state_sqlite(self, filename: str) -> False:
        """Loads the complete Ghost state from an SQLite database dynamically."""
        logging.info(f"Dynamically loading state from SQLite database: {filename}...")
        conn = None
        try:
            # Check if file exists before connecting
            if not os.path.exists(filename):
                raise Exception(f"SQLite save file {filename} not found. Starting with fresh state.")

            conn = sqlite3.connect(filename)
            conn.row_factory = sqlite3.Row  # Access columns by name
            cursor = conn.cursor()

            # --- Reset internal state before loading ---
            self.ghost._reset_internal_state()

            # --- Load Metadata & Config ---
            try:
                cursor.execute("SELECT value FROM metadata WHERE key = 'current_tick_id';")
                self.ghost.current_tick_id = int(cursor.fetchone()['value'])
                cursor.execute("SELECT value FROM metadata WHERE key = 'current_knoxel_id';")
                self.ghost.current_knoxel_id = int(cursor.fetchone()['value'])
            except (TypeError, sqlite3.Error, ValueError) as e:  # Handle missing table/keys or non-integer values
                logging.warning(f"Could not load metadata (tick/knoxel IDs): {e}. Using defaults.")
                self.ghost.current_tick_id = 0
                self.ghost.current_knoxel_id = 0

            try:
                cursor.execute("SELECT key, value FROM config;")
                config_data = {}
                loaded_config = cursor.fetchall()
                config_fields = GhostConfig.model_fields
                for row in loaded_config:
                    key, value_str = row['key'], row['value']
                    if key in config_fields:
                        target_type = config_fields[key].annotation
                        # Attempt to deserialize complex types (e.g., lists from JSON)
                        try:
                            # Basic check if it looks like JSON list/dict
                            if (isinstance(target_type, type) and issubclass(target_type, (List, Dict))) or \
                                    (get_origin(target_type) in [list, dict, List, Dict]):
                                config_data[key] = json.loads(value_str)
                            else:  # Otherwise, try casting basic types or keep as string
                                if target_type is int:
                                    config_data[key] = int(value_str)
                                elif target_type is float:
                                    config_data[key] = float(value_str)
                                elif target_type is bool:
                                    config_data[key] = value_str.lower() in ['true', '1']
                                else:
                                    config_data[key] = value_str  # Keep as string if other type
                        except (json.JSONDecodeError, ValueError, TypeError) as e:
                            logging.warning(
                                f"Error parsing config value for '{key}': {e}. Value: '{value_str}'. Using raw string.")
                            config_data[key] = value_str
                    else:
                        logging.warning(f"Ignoring unknown config key from DB: {key}")

                self.ghost.config = GhostConfig(**config_data)
                logging.info(f"Loaded config: {self.ghost.config.model_dump_json(indent=1)}")
            except (sqlite3.Error, ValidationError) as e:
                logging.error(f"Could not load or validate config: {e}. Using default config.")
                self.ghost.config = GhostConfig()  # Use defaults

            # --- Load Knoxels Dynamically ---
            knoxel_subclasses = self._get_knoxel_subclasses()
            loaded_knoxels = {}  # Temp dict to hold loaded knoxels {id: instance}

            for knoxel_cls in knoxel_subclasses.values():
                table_name = knoxel_cls.__name__.lower()
                logging.debug(f"Attempting to load from table: {table_name}")
                try:
                    cursor.execute(f"SELECT * FROM {table_name};")
                    rows = cursor.fetchall()
                    if not rows:
                        logging.debug(f"Table {table_name} is empty or does not exist.")
                        continue

                    column_names = [desc[0] for desc in cursor.description]
                    model_fields = knoxel_cls.model_fields

                    for row in rows:
                        knoxel_data = {}
                        row_dict = dict(row)  # Convert sqlite3.Row to dict
                        knoxel_id = row_dict.get('id', -1)  # Get ID for logging

                        for field_name, field_info in model_fields.items():
                            if field_name in column_names:
                                db_value = row_dict[field_name]
                                try:
                                    py_value = self._deserialize_value_from_db(db_value, field_info.annotation)
                                    knoxel_data[field_name] = py_value
                                except Exception as e:
                                    logging.error(
                                        f"Error deserializing field '{field_name}' for {table_name} ID {knoxel_id}: {e}. DB Value: {db_value}. Skipping field.")
                            else:
                                # Field exists in model but not in DB table (schema evolution)
                                # Pydantic will use default value if available, otherwise raise error if required
                                logging.debug(
                                    f"Field '{field_name}' not found in DB table '{table_name}' for knoxel ID {knoxel_id}. Using model default if available.")

                        # Add fields present in DB but not in model? Pydantic ignores extra by default.
                        if 'id' not in knoxel_data:  # Should always be present if table exists
                            logging.error(f"Missing 'id' column data for row in {table_name}. Skipping row: {row_dict}")
                            continue

                        try:
                            knoxel_instance = knoxel_cls(**knoxel_data)
                            self.ghost.all_knoxels[knoxel_instance.id] = knoxel_instance
                        except ValidationError as e:
                            logging.error(f"Pydantic validation failed for {knoxel_cls.__name__} ID {knoxel_id}: {e}")
                            logging.error(f"Data causing error: {knoxel_data}")
                        except Exception as e:
                            logging.error(f"Unexpected error instantiating {knoxel_cls.__name__} ID {knoxel_id}: {e}")
                            logging.error(f"Data causing error: {knoxel_data}")

                except sqlite3.OperationalError as e:
                    logging.warning(f"Could not read from table {table_name} (might not exist in this DB): {e}")
                except Exception as e:
                    logging.error(f"Unexpected error loading from table {table_name}: {e}", exc_info=True)

            logging.info(f"Loaded {len(self.ghost.all_knoxels)} knoxels from database.")

            # Rebuild specific lists from the loaded knoxels
            self.ghost._rebuild_specific_lists()

            # --- Load Ghost States Dynamically ---
            state_table_name = "ghost_states"
            self.ghost.states = []
            nested_state_models = {
                'state_emotions': EmotionalAxesModel,
                'state_needs': NeedsAxesModel,
                'state_cognition': CognitionAxesModel
            }

            try:
                cursor.execute(f"SELECT * FROM {state_table_name} ORDER BY tick_id ASC;")
                rows = cursor.fetchall()
                if not rows:
                    logging.warning(f"Ghost states table '{state_table_name}' is empty or does not exist.")

                column_names = [desc[0] for desc in cursor.description]
                base_state_fields = GhostState.model_fields

                for row in rows:
                    state_data = {}
                    nested_models_data = {prefix: {} for prefix in nested_state_models}
                    row_dict = dict(row)
                    tick_id = row_dict.get('tick_id', -1)

                    # Process columns from DB row
                    for col_name in column_names:
                        db_value = row_dict[col_name]
                        processed = False

                        # Check if it's a flattened nested model field
                        for prefix, model_cls in nested_state_models.items():
                            short_prefix = prefix.split('_')[-1]  # e.g., emotion
                            if col_name.startswith(f"{short_prefix}_"):
                                field_name = col_name[len(short_prefix) + 1:]
                                if field_name in model_cls.model_fields:
                                    target_type = model_cls.model_fields[field_name].annotation
                                    try:
                                        py_value = self._deserialize_value_from_db(db_value, target_type)
                                        nested_models_data[prefix][field_name] = py_value
                                    except Exception as e:
                                        logging.error(
                                            f"Error deserializing nested field '{col_name}' for state tick {tick_id}: {e}. DB Value: {db_value}. Skipping field.")
                                    processed = True
                                    break  # Move to next column once matched

                        if processed:
                            continue

                        # Check if it's a base GhostState field (or reference ID)
                        if col_name not in base_state_fields and "_" in col_name:
                            field_name = to_camel_case(col_name)
                        else:
                            field_name = col_name

                        try:
                            if field_name in base_state_fields:
                                target_type = base_state_fields[field_name].annotation
                                py_value = self._deserialize_value_from_db(db_value, target_type)
                                state_data[field_name] = py_value
                            else:
                                raise Exception(f"Unhandled column: {col_name}")
                        except Exception as e:
                            logger.critical(f"Bad column in GhostState: {col_name}")

                    # Instantiate nested models
                    for prefix, model_cls in nested_state_models.items():
                        try:
                            instance = model_cls(**nested_models_data[prefix])
                            state_data[prefix] = instance
                        except ValidationError as e:
                            logging.error(
                                f"Validation failed for {model_cls.__name__} in state tick {tick_id}: {e}. Data: {nested_models_data[prefix]}")
                            state_data[prefix] = model_cls()  # Use default instance
                        except Exception as e:
                            logging.error(
                                f"Error instantiating {model_cls.__name__} in state tick {tick_id}: {e}. Data: {nested_models_data[prefix]}")
                            state_data[prefix] = model_cls()
                    ghost_state_instance = GhostState(**state_data)
                    self.ghost.states.append(ghost_state_instance)
            except sqlite3.OperationalError as e:
                logging.warning(f"Could not read from table {state_table_name}: {e}")
                #raise e
            except Exception as e:
                logging.error(f"Unexpected error loading from table {state_table_name}: {e}", exc_info=True)
                #raise e
            logging.info(f"Loaded {len(self.ghost.states)} Ghost states from database.")
            self.ghost.states.sort(key=lambda s: s.tick_id)  # Ensure states are ordered
            self.ghost.current_state = self.ghost.states[-1] if self.ghost.states else None
            self.ghost.current_db_path = os.path.abspath(filename)
            return True

        except sqlite3.Error as e:
            logging.error(f"SQLite Error during load: {e}", exc_info=True)
            self.ghost._reset_internal_state()  # Reset on error
            raise e
        except Exception as e:
            logging.error(f"Unexpected Error during load: {e}", exc_info=True)
            self.ghost._reset_internal_state()
            raise e
        finally:
            if conn: conn.close()

    # --- Utility to prepare data for older saves (used by original load_state) ---
    def _get_knoxel_type_map(self):  # Helper for original load_state
        return {
            cls.__name__: cls for cls in
            [KnoxelBase, Stimulus, Intention, Action, MemoryClusterKnoxel, DeclarativeFactKnoxel, Narrative, Feature]
        }

    def _prepare_knoxel_data_for_load(self, knoxel_cls: Type[KnoxelBase],
                                      data: Dict) -> Dict:  # Helper for original load_state
        # Add default values for fields potentially missing in older JSON saves
        if knoxel_cls == Intention:
            data.setdefault('internal', True)  # Default old intentions to internal goals
            data.setdefault('originating_action_id', None)
            data.setdefault('affective_valence', 0.0)
            data.setdefault('incentive_salience', 0.5)  # Default salience if missing
            data.setdefault('urgency', 0.5)  # Default urgency if missing
            data.setdefault('fulfilment', 0.0)  # Default fulfilment if missing
        elif knoxel_cls == Action:
            data.setdefault('generated_expectation_ids', [])
            data.setdefault('content', data.get('content', ''))  # Handle old action format maybe
        elif knoxel_cls == Feature:
            data.setdefault('incentive_salience', None)  # Added later
            data.setdefault('causal', False)  # Ensure default
            data.setdefault('interlocus', 0.0)  # Ensure default

        # Ensure basic KnoxelBase fields have defaults if missing
        data.setdefault('tick_id', -1)
        data.setdefault('content', '')
        data.setdefault('embedding', [])  # Embeddings not saved in JSON anyway
        data.setdefault('timestamp_creation', datetime.now().isoformat())
        data.setdefault('timestamp_world_begin', datetime.now().isoformat())
        data.setdefault('timestamp_world_end', datetime.now().isoformat())
        return data
