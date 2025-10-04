from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Callable, TypeVar
from enum import Enum
import duckdb
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance  # you use this everywhere
import datetime as dt

from pm.data_structures import KnoxelBase
from pm.ghosts.base_ghost import BaseGhost
from pm.llm.llm_proxy import LlmManagerProxy
from pm.mental_state_vectors import VectorModelReservedSize, FullMentalState

T = TypeVar("T")

# --------------------- small helpers ---------------------

_SCALAR_TYPES = (str, int, float, bool, dt.datetime, dt.date, dt.time)

def _is_scalar(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, _SCALAR_TYPES):
        return True
    if isinstance(v, Enum):
        return True
    return False

def _row_dict(obj: KnoxelBase,
              whitelist: Optional[Sequence[str]] = None,
              blacklist: Optional[Sequence[str]] = None,
              custom_props: Optional[Dict[str, Callable[[Any], Any]]] = None) -> Dict[str, Any]:
    """
    Reflect object to a row dict:
    - include every non-iterable (scalar) attribute (strings allowed) from __dict__
    - apply whitelist/blacklist
    - add custom properties (computed columns)
    """
    d = obj.__dict__  # direct access as requested
    row: Dict[str, Any] = {}

    if whitelist:
        cols = whitelist
    else:
        cols = [k for k in d.keys() if not (blacklist and k in blacklist)]

    for k in cols:
        v = d.get(k)
        if _is_scalar(v):
            # normalize Enums to their value/name
            if isinstance(v, Enum):
                row[k] = v.value if hasattr(v, "value") else v.name
            elif isinstance(v, (dt.datetime, dt.date, dt.time)):
                row[k] = v  # pandas/duckdb handle these
            else:
                row[k] = v
        else:
            # keep list-like columns only if explicitly whitelisted
            #if whitelist and k in whitelist:
            row[k] = v
            # otherwise skip (e.g., embedding, metadata) unless whitelisted deliberately
    if custom_props:
        for name, fn in custom_props.items():
            row[name] = fn(obj)
    return row

def _to_dataframe(items: Iterable[KnoxelBase],
                  whitelist: Optional[Sequence[str]],
                  blacklist: Optional[Sequence[str]],
                  custom_props: Optional[Dict[str, Callable[[Any], Any]]]) -> pd.DataFrame:
    rows = [_row_dict(o, whitelist=whitelist, blacklist=blacklist, custom_props=custom_props) for o in items]
    return pd.DataFrame(rows)

def _is_number_dtype(dtype) -> bool:
    return str(dtype) in (
        "int8","int16","int32","int64",
        "uint8","uint16","uint32","uint64",
        "float16","float32","float64","Float64","Int64"
    )

def _clip(s: str, width: int) -> str:
    if width <= 0 or len(s) <= width:
        return s
    return s[: max(1, width - 1)] + "…"

def _fmt_scalar(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

def _infer_alignments(df: pd.DataFrame) -> list[str]:
    aligns = []
    for c in df.columns:
        aligns.append("right" if _is_number_dtype(df[c].dtype) else "left")
    return aligns

def _col_widths(df: pd.DataFrame, max_col_width: int, max_rows: int) -> list[int]:
    widths = [len(str(c)) for c in df.columns]
    n = min(len(df), max_rows)
    for j, c in enumerate(df.columns):
        col = df[c].iloc[:n]
        for v in col:
            s = _fmt_scalar(v)
            w = len(s)
            if w > widths[j]:
                widths[j] = w
    return [min(w, max_col_width) for w in widths]

def _is_sequence_like(v) -> bool:
    if v is None or isinstance(v, (str, bytes)):  # strings aren't "sequence-like" here
        return False
    try:
        iter(v)
        return True
    except Exception:
        return False

def _mask_value(col: str, v, max_cell_len: int) -> str:
    """
    Replace large/opaque values (vectors, dicts, long text) with compact placeholders.
    We still show that the column exists, but not its raw payload.
    """
    # vectors / numpy arrays / lists / tuples
    if isinstance(v, (list, tuple)):
        return f"⟨{col}:{len(v)}d⟩"
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            return f"⟨{col}:{int(v.size)}d⟩"
    except Exception:
        pass

    # dicts or other objects
    if isinstance(v, dict):
        return f"⟨{col}:map⟩"
    if _is_sequence_like(v):
        return f"⟨{col}:seq⟩"

    # scalar-ish → pretty print with clipping
    s = _fmt_scalar(v)
    if max_cell_len > 0 and len(s) > max_cell_len:
        return s[: max(1, max_cell_len - 1)] + "…"
    return s

# --------------------- main class ---------------------

@dataclass
class MemQuery:
    """
    In-RAM SQL facade for knoxels with:
      - dynamic reflection to tables
      - DuckDB UDFs for cosine distance and on-the-fly query embeddings via your LLM
      - dynamic view builders for datetime, ranges, enums
    """
    ghost: BaseGhost
    llm: LlmManagerProxy

    conn: duckdb.DuckDBPyConnection = field(init=False)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    redact_columns: set[str] = field(default_factory=list)

    def set_redact_columns(self, names: list[str]) -> None:
        self.redact_columns = set(names)

    def __post_init__(self):
        self.redact_columns = {"embedding", "emb", "vec", "vector", "metadata", "emotion_embedding", "mental_state_delta"}
        self.conn = duckdb.connect(database=":memory:")
        self._register_udfs()

    # ---------- UDFs (SQL-callable) ----------

    def _udf_query_embedding(self, text: str) -> list[float]:
        vec = self.llm.get_embedding(text)           # your global embedding endpoint
        if isinstance(vec, np.ndarray):
            return vec.tolist()
        return list(vec)

    def _get_embedding_by_id(self, _id: int) -> list[float]:
        return self.ghost.get_knoxel_by_id(_id).embedding

    def _udf_cosine_distance(self, a: list[float], b: list[float]) -> float:
        # scipy returns distance in [0..2] if vectors aren't normalized; typically [0..2], but cosine distance is 1 - cosine similarity.
        # you use it everywhere; leave as-is.
        return float(cosine_distance(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))

    # 1) get_emotional_state(when, ema) -> LIST<FLOAT>
    def _get_emotional_state(self, when: str, ema: str) -> list[float]:
        v = np.random.rand(VectorModelReservedSize)
        v_hat = v / np.linalg.norm(v)
        return v_hat.tolist()

    # 2) explain_mental_state(vec) -> TEXT
    def _explain_mental_state(self, v: list[float]) -> str:
        fms = FullMentalState.init_from_list(v)
        return fms.model_dump_json(indent=2)

    # 3) vec_mean(vs LIST<LIST<FLOAT>>) -> LIST<FLOAT>
    def _vec_mean(self, vs: list[list[float]]) -> list[float]:
        arr = np.asarray(vs, dtype=np.float32)
        return arr.mean(axis=0).astype(np.float32).tolist()

    def _register_udfs(self) -> None:
        self.conn.create_function("query_embedding", self._udf_query_embedding)      # TEXT -> LIST<FLOAT>
        self.conn.create_function("cosine_distance", self._udf_cosine_distance)     # LIST, LIST -> FLOAT
        self.conn.create_function("vec_mean", self._vec_mean)
        self.conn.create_function("get_emotional_state", self._get_emotional_state)
        self.conn.create_function("explain_mental_state", self._explain_mental_state)
        self.conn.create_function("get_embedding_by_id", self._get_embedding_by_id)

    # ---------- Table ingest ----------

    def add_list(self,
                 items: Sequence[KnoxelBase],
                 name: str,
                 whitelist: Optional[Sequence[str]] = None,
                 blacklist: Optional[Sequence[str]] = None,
                 custom_props: Optional[Dict[str, Callable[[T], Any]]] = None) -> None:
        """
        Reflect a list of knoxels into a DuckDB table:
        - every non-iterable field becomes a column (strings ok)
        - lists/dicts ignored unless explicitly whitelisted
        - custom_props: dict of {column_name: fn(obj)->value} to display derived data (e.g., token_count)
        """
        df = _to_dataframe(items, whitelist, blacklist, custom_props)
        self.tables[name] = df
        self.conn.register(name, df)

    # ---------- Dynamic view builders ----------

    def add_view_datetime(self,
                          source_table: str,
                          view_name: str,
                          field_name: str) -> None:
        """
        Build a hierarchical-friendly datetime projection:
          - SELECT *, year, month, day, hour
        """
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
              *,
              strftime({field_name}, '%Y') AS year,
              strftime({field_name}, '%m') AS month,
              strftime({field_name}, '%d') AS day,
              strftime({field_name}, '%H') AS hour
            FROM {source_table};
        """)

    def add_view_range(self,
                       source_table: str,
                       view_name: str,
                       field_name: str,
                       min_value: Optional[Any] = None,
                       max_value: Optional[Any] = None,
                       include_null: bool = False) -> None:
        """
        Build a simple range-sliced view over numeric/datetime fields.
        Examples:
          add_view_range("features", "features_recent", "timestamp_world_begin", min_value="2025-01-01")
          add_view_range("facts", "facts_high_imp", "importance", min_value=0.8)
        """
        conds: List[str] = []
        if min_value is not None:
            if isinstance(min_value, (dt.datetime, dt.date)):
                conds.append(f"{field_name} >= TIMESTAMP '{pd.to_datetime(min_value)}'")
            else:
                conds.append(f"{field_name} >= {repr(min_value)}")
        if max_value is not None:
            if isinstance(max_value, (dt.datetime, dt.date)):
                conds.append(f"{field_name} <= TIMESTAMP '{pd.to_datetime(max_value)}'")
            else:
                conds.append(f"{field_name} <= {repr(max_value)}")
        if include_null:
            conds.append(f"{field_name} IS NULL")
        where = " OR ".join(conds) if include_null and conds else " AND ".join(conds) if conds else "TRUE"
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT * FROM {source_table}
            WHERE {where};
        """)

    def add_view_enum(self,
                      source_table: str,
                      view_name: str,
                      field_name: str,
                      only_values: Optional[Sequence[Any]] = None) -> None:
        """
        Build an enum-sliced view. If `only_values` is provided, keep those; else just expose the field.
        """
        if only_values:
            in_list = ", ".join(repr(x.value if isinstance(x, Enum) else x) for x in only_values)
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM {source_table}
                WHERE {field_name} IN ({in_list});
            """)
        else:
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM {source_table};
            """)

    # ---------- Convenience: list views / tables ----------

    def list_views(self) -> pd.DataFrame:
        return self.conn.execute("SELECT view_name AS view FROM duckdb_views() ORDER BY 1").fetchdf()

    def list_tables(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY 1
        """).fetchdf()

    # ---------- Raw SQL ----------

    def query(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).fetchdf()

    def to_pretty_table(
        self,
        df: pd.DataFrame,
        max_rows: int = 40,
        max_col_width: int = 40,
        *,
        redact: Optional[list[str]] = None,
        max_cell_len: int = 120,
    ) -> str:
        """
        Fixed-width, aligned table for humans.
        - Masks columns listed in `redact` (defaults to self.redact_columns)
        - Truncates strings to max_cell_len
        """
        redact_set = set(redact or self.redact_columns)

        n = min(len(df), max_rows)
        df2 = df.iloc[:n].copy()

        # build a *display* frame where values are masked
        disp = pd.DataFrame(index=df2.index)
        for c in df2.columns:
            if c in redact_set:
                disp[c] = df2[c].map(lambda v: _mask_value(c, v, max_cell_len))
            else:
                disp[c] = df2[c].map(lambda v: _mask_value(c, v, max_cell_len))

        headers = [str(c) for c in disp.columns]
        widths = _col_widths(disp, max_col_width=max_col_width, max_rows=max_rows)
        aligns = _infer_alignments(disp)  # after masking, most cols are strings → left-align

        def fmt_cell(j, v):
            s = str(v)
            s = _clip(s, widths[j])
            return s.rjust(widths[j]) if aligns[j] == "right" else s.ljust(widths[j])

        header_line = "  ".join(
            (h if len(h) <= widths[j] else _clip(h, widths[j])).ljust(widths[j]) for j, h in enumerate(headers)
        )
        sep_line = "  ".join("-" * w for w in widths)
        body_lines = []
        for i in range(n):
            row = disp.iloc[i]
            body_lines.append("  ".join(fmt_cell(j, row[c]) for j, c in enumerate(disp.columns)))

        footer = ""
        if len(df) > n:
            footer = f"\n… ({len(df) - n} more rows)"

        return "\n".join([header_line, sep_line] + body_lines) + footer

    def to_csv(
        self,
        df: pd.DataFrame,
        max_rows: int = 2000,
        sep: str = ";",
        *,
        redact: Optional[list[str]] = None,
        max_cell_len: int = 120,
    ) -> str:
        """
        CSV for LLMs (semicolon-separated).
        - Masks columns in `redact` with compact placeholders so the column still exists.
        """
        redact_set = set(redact or self.redact_columns)

        n = min(len(df), max_rows)
        df2 = df.iloc[:n].copy()

        disp = pd.DataFrame(index=df2.index)
        for c in df2.columns:
            if c in redact_set:
                disp[c] = df2[c].map(lambda v: _mask_value(c, v, max_cell_len))
            else:
                disp[c] = df2[c].map(lambda v: _mask_value(c, v, max_cell_len))

        return disp.to_csv(index=False, sep=sep)
    # ---------- Introspection ----------

    def help_info(self) -> Dict[str, Any]:
        """
        Dynamic snapshot: tables, views, columns (dtypes), counts, UDFs, sample queries.
        """
        # tables (registered)
        table_meta = []
        for name, df in self.tables.items():
            cols = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]
            table_meta.append({
                "name": name,
                "rows": int(len(df)),
                "columns": cols,
            })

        # views (from duckdb)
        views_df = self.list_views()
        views = views_df["view"].tolist()

        # UDFs
        udfs = [
            {"name": "query_embedding", "sig": "TEXT -> LIST<FLOAT>", "desc": "Calls llm.get_embedding(text)."},
            {"name": "cosine_distance", "sig": "LIST<FLOAT>, LIST<FLOAT> -> FLOAT", "desc": "scipy.spatial.distance.cosine"},
        ]

        # sample queries (adapt to your taste)
        samples = [
            # vector search on your table that keeps 'embedding'
            "WITH q AS (SELECT query_embedding('your query text') AS qvec)\n"
            "SELECT id, content, cosine_distance(embedding, (SELECT qvec FROM q)) AS dist\n"
            "FROM features\n"
            "WHERE dist < 0.5\n"
            "ORDER BY dist ASC\n"
            "LIMIT 10;",

            # datetime hierarchy view (assuming you created it)
            "SELECT year, month, day, COUNT(*) AS n\n"
            "FROM features_hier\n"
            "GROUP BY 1,2,3\n"
            "ORDER BY 1,2,3;",

            # enum slice example
            "SELECT * FROM features WHERE type = 'Dialogue' LIMIT 20;",

            # generic range filter
            "SELECT id, importance, content FROM facts WHERE importance >= 0.8 ORDER BY importance DESC LIMIT 20;",
        ]

        return {
            "engine": "duckdb",
            "tables": table_meta,
            "views": views,
            "udfs": udfs,
            "usage": {
                "add_list": "mem.add_list(ghost.all_features, name='features', whitelist=['id','content','embedding','timestamp_world_begin'])",
                "add_view_datetime": "mem.add_view_datetime('features','features_hier','timestamp_world_begin')",
                "add_view_range": "mem.add_view_range('facts','facts_high','importance',min_value=0.8)",
                "add_view_enum": "mem.add_view_enum('features','features_dialogue','type',only_values=['Dialogue'])",
            },
            "examples": samples,
        }

    # ---------- Help renders ----------
    def render_help_human(self, peek_rows: int = 5) -> str:
        info = self.help_info()
        lines = []
        lines.append("engine: duckdb")
        lines.append("redactions: " + ", ".join(sorted(self.redact_columns)))
        lines.append("udfs:")
        for f in info["udfs"]:
            lines.append(f"  - name: {f['name']}")
            lines.append(f"    sig: {f['sig']}")
            lines.append(f"    desc: {f['desc']}")
        lines.append("")

        for t in info["tables"]:
            name = t["name"];
            rows = t["rows"]
            df = self.tables[name]
            lines.append(f"table: {name}  (rows: {rows})")
            schema_lines = ["  schema:"]
            for col in t["columns"]:
                schema_lines.append(f"    - {col['name']}: {col['dtype']}")
            lines.extend(schema_lines)
            if rows > 0:
                lines.append("  sample:")
                tbl = self.to_pretty_table(
                    df.head(peek_rows),
                    max_rows=peek_rows,
                    max_col_width=60,
                    redact=list(self.redact_columns),
                )
                lines.extend("    " + ln for ln in tbl.splitlines())
            lines.append("")

        lines.append("views:")
        for v in info["views"]:
            lines.append(f"  - {v}")
        lines.append("")

        lines.append("example_queries:")
        for q in info["examples"]:
            lines.append("  - |")
            for ln in q.splitlines():
                lines.append(f"      {ln}")

        return "\n".join(lines)

    def render_help_llm(self, peek_rows: int = 5, sep: str = ";") -> str:
        info = self.help_info()
        parts = []
        header = {
            "engine": info["engine"],
            "redactions": sorted(self.redact_columns),
            "udfs": [{k: f[k] for k in ("name", "sig")} for f in info["udfs"]],
            "views": info["views"],
            "tips": [
                "Keep embedding columns present for SQL; renderers mask them.",
                "Use query_embedding('<text>') and cosine_distance(embedding, query_embedding('<text>')).",
                "Create views dynamically with add_view_datetime/add_view_range/add_view_enum.",
            ]
        }
        import yaml
        parts.append(yaml.safe_dump(header, sort_keys=False).rstrip())

        for t in info["tables"]:
            name = t["name"]
            schema_df = pd.DataFrame(t["columns"])
            parts.append(f"# schema:{name}")
            parts.append(self.to_csv(schema_df, max_rows=len(schema_df), sep=sep, redact=[]).rstrip())
            if t["rows"] > 0:
                parts.append(f"# sample:{name}")
                parts.append(self.to_csv(self.tables[name].head(peek_rows), max_rows=peek_rows, sep=sep,
                                         redact=list(self.redact_columns)).rstrip())

        examples_yaml = {"examples": info["examples"]}
        parts.append(yaml.safe_dump(examples_yaml, sort_keys=False).rstrip())

        return "\n\n".join(parts)

    def init_default(self):
        g = self.ghost
        self.set_redact_columns(["embedding", "mental_state_delta", "metadata", "vec", "vector"])

        self.add_list(g.all_features, name="features", whitelist=None, blacklist=[], custom_props=None)
        self.add_list(g.all_stimuli, name="stimuli", whitelist=None, blacklist=[], custom_props=None)
        self.add_list(g.all_actions, name="actions", whitelist=None, blacklist=[], custom_props=None)
        self.add_list(g.all_intentions, name="intentions", whitelist=None, blacklist=[], custom_props=None)
        self.add_list(g.all_narratives, name="narratives", whitelist=None, blacklist=[], custom_props=None)
        self.add_list(g.all_episodic_memories, name="clusters", whitelist=None, blacklist=[], custom_props=None)
        self.add_list(g.all_declarative_facts, name="facts", whitelist=None, blacklist=[], custom_props=None)

        # --------------------------------
        # 2) Canonical hierarchical views
        # --------------------------------
        # Time projections for fast grouping/drilldown (YYYY/MM/DD/HH)
        self.add_view_datetime("features",   "features_time",   "timestamp_world_begin")
        self.add_view_datetime("stimuli",    "stimuli_time",    "timestamp_world_begin")
        self.add_view_datetime("actions",    "actions_time",    "timestamp_world_begin")
        self.add_view_datetime("narratives", "narratives_time", "timestamp_world_begin")
        self.add_view_datetime("clusters",   "clusters_time",   "timestamp_world_begin")
        self.add_view_datetime("facts",      "facts_time",      "timestamp_world_begin")

        # -----------------------------
        # 3) Behavior-centric superviews
        # -----------------------------

        # 3.1 Unified events stream (for “what happened when?”)
        # NOTE: Enum values are stored as strings by reflection.
        self.conn.execute("""
            CREATE OR REPLACE VIEW events AS
            SELECT id::VARCHAR AS id, tick_id, 'feature'  AS kind, feature_type   AS subtype,
                   content, timestamp_world_begin AS ts, source, affective_valence, incentive_salience, interlocus,
                   embedding, mental_state_delta
            FROM features
            UNION ALL
            SELECT id::VARCHAR, tick_id, 'stimulus' AS kind, stimulus_type AS subtype,
                   content, timestamp_world_begin, source, NULL, NULL, NULL,
                   embedding, mental_state_delta
            FROM stimuli
            UNION ALL
            SELECT id::VARCHAR, tick_id, 'action'   AS kind, action_type   AS subtype,
                   content, timestamp_world_begin, NULL, NULL, NULL, NULL,
                   embedding, mental_state_delta
            FROM actions
            UNION ALL
            SELECT id::VARCHAR, tick_id, 'narrative' AS kind, narrative_type AS subtype,
                   content, timestamp_world_begin, target_name AS source, NULL, NULL, NULL,
                   embedding, mental_state_delta
            FROM narratives
        """)

        # 3.2 Dialogue stream (user stimuli ↔ dialogue features ↔ replies)
        self.conn.execute("""
            CREATE OR REPLACE VIEW dialogue_stream AS
            SELECT 'stimulus' AS kind, stimulus_type AS subtype, id::VARCHAR AS id, tick_id, timestamp_world_begin AS ts,
                   source, content, embedding, mental_state_delta
            FROM stimuli
            WHERE stimulus_type IN ('UserMessage','SystemMessage','WakeUp','EngagementOpportunity','UserInactivity','TimeOfDayChange','LowNeedTrigger')
            UNION ALL
            SELECT 'feature' AS kind, feature_type AS subtype, id::VARCHAR, tick_id, timestamp_world_begin,
                   source, content, embedding, mental_state_delta
            FROM features
            WHERE feature_type IN ('Dialogue','ExternalThought','SystemMessage')
            UNION ALL
            SELECT 'action' AS kind, action_type AS subtype, id::VARCHAR, tick_id, timestamp_world_begin,
                   NULL AS source, content, embedding, mental_state_delta
            FROM actions
            WHERE action_type IN ('Reply','ToolCallAndReply','ToolCall','InitiateUserConversation','InitiateInternalContemplation','Ignore','Sleep','WebSearch','CallMemory','LoopBack','ToolResult','Think','Plan','ManageIntent','ManageAwarenessFocus','ReflectThoughts')
            ORDER BY ts ASC, tick_id ASC, id ASC
        """)

        # 3.3 Affect dynamics & subjective experience (how the AI felt/experienced)
        self.conn.execute("""
            CREATE OR REPLACE VIEW affect_dynamics AS
            SELECT id::VARCHAR AS id, tick_id, timestamp_world_begin AS ts,
                   feature_type, source, content,
                   affective_valence, incentive_salience, interlocus, embedding, mental_state_delta
            FROM features
            WHERE feature_type IN ('Feeling','SubjectiveExperience','AttentionFocus','ExpectationOutcome','NarrativeUpdate','SituationalModel')
            ORDER BY ts ASC
        """)

        # 3.4 Actions joined to intentions (expectations) generated by those actions
        #    - actions.generated_expectation_ids is LIST<INT> → UNNEST
        self.conn.execute("""
            CREATE OR REPLACE VIEW actions_expectations AS
            WITH exploded AS (
                SELECT
                    a.id::VARCHAR AS action_id,
                    a.tick_id     AS action_tick,
                    a.timestamp_world_begin AS action_ts,
                    a.action_type AS action_type, 
                    UNNEST(a.generated_expectation_ids) AS intention_id
                FROM actions a
            )
            SELECT
                e.action_id, e.action_tick, e.action_ts, e.action_type,
                i.id::VARCHAR AS intention_id, i.internal, i.urgency, i.incentive_salience,
                i.affective_valence, i.fulfilment, i.originating_action_id,
                i.timestamp_world_begin AS intention_ts,
                i.content AS intention_content,
            FROM exploded e
            JOIN intentions i ON i.id = e.intention_id
            ORDER BY e.action_ts ASC
        """)

        # 3.5 Temporal clusters (behavioral summaries) and topical clusters (themes)
        self.conn.execute("""
            CREATE OR REPLACE VIEW clusters_temporal AS
            SELECT
                id::VARCHAR AS id, level, cluster_type, temporal_key,
                token, facts_extracted, content,
                timestamp_world_begin AS ts,
                min_tick_id, max_tick_id, min_event_id, max_event_id, embedding, mental_state_delta
            FROM clusters
            WHERE cluster_type = 'temporal'
            ORDER BY ts ASC, level ASC
        """)
        self.conn.execute("""
            CREATE OR REPLACE VIEW clusters_topical AS
            SELECT
                id::VARCHAR AS id, level, cluster_type,
                included_event_ids, included_cluster_ids,
                token, facts_extracted, content,
                timestamp_world_begin AS ts, embedding, mental_state_delta
            FROM clusters
            WHERE cluster_type = 'topical'
            ORDER BY ts ASC, level ASC
        """)

        # 3.6 Declarative facts — importance/time-dependence slices
        self.conn.execute("""
            CREATE OR REPLACE VIEW facts_ranked AS
            SELECT
                id::VARCHAR AS id,
                content,
                importance,
                time_dependent,
                embedding,
                timestamp_world_begin AS ts,
                CASE
                    WHEN importance >= 0.9 THEN 'critical'
                    WHEN importance >= 0.75 THEN 'high'
                    WHEN importance >= 0.5 THEN 'medium'
                    WHEN importance >= 0.25 THEN 'low'
                    ELSE 'very_low'
                END AS importance_bucket
            FROM facts
            ORDER BY importance DESC, ts DESC
        """)

        # 3.7 Event vectors (pre-embedded column for vector queries)
        #     Keep vec as the original `embedding` column for direct cosine queries.
        self.conn.execute("""
            CREATE OR REPLACE VIEW events_with_vec AS
            SELECT id, kind, subtype, ts, content, embedding AS vec
            FROM events
            ORDER BY ts ASC
        """)

        self.conn.execute("""
          CREATE OR REPLACE VIEW mental_state_panel AS
          SELECT
            f.id::VARCHAR AS feature_id,
            f.timestamp_world_begin AS ts,
            explain_mental_state(f.mental_state_delta) AS ms_explain
          FROM features f
          WHERE f.mental_state_delta IS NOT NULL
          ORDER BY ts DESC
        """)

        # ---------- Curated query catalog (welcome + tests) ----------

        def query_catalog(self) -> List[Dict[str, Any]]:
            """
            A curated set of commented SQL queries for LLM + regression tests.
            - Each item: {id, title, tags, description, sql}
            - Queries prefer portable constructs and only rely on UDFs registered in this class.
            - Placeholders like {{TEXT}} can be string-replaced by your caller before execution.
            """
            Q: List[Dict[str, Any]] = []

            # 0) Sanity checks
            Q.append(dict(
                id="sanity.counts",
                title="Row counts per main table",
                tags=["sanity", "counts"],
                description="Basic counts to ensure tables are visible and non-empty.",
                sql="""
    SELECT 'features'   AS table, COUNT(*) AS n FROM features   UNION ALL
    SELECT 'stimuli'           , COUNT(*)     FROM stimuli      UNION ALL
    SELECT 'actions'           , COUNT(*)     FROM actions      UNION ALL
    SELECT 'intentions'        , COUNT(*)     FROM intentions   UNION ALL
    SELECT 'narratives'        , COUNT(*)     FROM narratives   UNION ALL
    SELECT 'clusters'          , COUNT(*)     FROM clusters     UNION ALL
    SELECT 'facts'             , COUNT(*)     FROM facts
    ;""".strip()
            ))

            # 1) Recent events (24h)
            Q.append(dict(
                id="events.recent_24h",
                title="Events in the last 24 hours",
                tags=["events", "time", "recent"],
                description="Unified stream to check recent activity windows.",
                sql="""
    SELECT id, kind, subtype, ts, content
    FROM events
    WHERE ts >= localtimestamp - INTERVAL 1 DAY
    ORDER BY ts DESC
    LIMIT 200;
    """.strip()
            ))

            # 2) Dialogue around last user message (±10 minutes)
            Q.append(dict(
                id="dialogue.around_last_user_message",
                title="Context window around the last UserMessage",
                tags=["dialogue", "stimuli", "actions", "context"],
                description="Find the last user message, then show nearby dialogue/actions.",
                sql="""
    WITH last_user AS (
      SELECT timestamp_world_begin AS ts
      FROM stimuli
      WHERE stimulus_type = 'UserMessage'
      ORDER BY ts DESC
      LIMIT 1
    )
    SELECT kind, subtype, id, ts, source, content
    FROM dialogue_stream
    WHERE ts BETWEEN (SELECT ts FROM last_user) - INTERVAL 10 MINUTE
                 AND (SELECT ts FROM last_user) + INTERVAL 10 MINUTE
    ORDER BY ts ASC, kind ASC;
    """.strip()
            ))

            # 3) Vector search over events using a free-text query
            Q.append(dict(
                id="events.sim.text",
                title="Semantic similarity to text over events",
                tags=["vectors", "cosine", "query_embedding", "ranking"],
                description="Embed query text and rank events by cosine distance.",
                sql="""
    WITH q AS (
      SELECT query_embedding({{TEXT}}) AS qvec
    )
    SELECT id, kind, subtype, ts,
           cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
           content
    FROM events
    ORDER BY dist ASC
    LIMIT 25;
    """.strip()
            ))

            # 4) Vector search over facts (important facts first)
            Q.append(dict(
                id="facts.sim.text",
                title="Semantic similarity to text over facts",
                tags=["vectors", "facts", "ranking"],
                description="Rank declarative facts by similarity to the query text; importance as a tiebreaker.",
                sql="""
    WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec)
    SELECT id, importance,
           cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
           content
    FROM facts
    ORDER BY dist ASC, importance DESC
    LIMIT 25;
    """.strip()
            ))

            # 5) Emotional state: compare clusters to a synthetic 'now' state (EMA=1d)
            Q.append(dict(
                id="clusters.sim.emotional_now",
                title="Clusters similar to current emotional state (EMA=1d)",
                tags=["mental_state", "clusters", "cosine"],
                description="Use get_emotional_state('now','1d') and compare to clusters_temporal mental_state_delta.",
                sql="""
    WITH q AS (SELECT get_emotional_state('now','1d') AS qvec)
    SELECT id, ts, content,
           cosine_distance(mental_state_delta, (SELECT qvec FROM q)) AS dist
    FROM clusters_temporal
    WHERE mental_state_delta IS NOT NULL
    ORDER BY dist ASC
    LIMIT 15;
    """.strip()
            ))

            # 6) Emotional state: average of the last 10 feature deltas
            Q.append(dict(
                id="clusters.sim.last10_features_mean",
                title="Clusters similar to mean mental state of last 10 features",
                tags=["mental_state", "vec_mean", "list_aggregate", "clusters"],
                description="Aggregate last 10 feature mental_state_delta and compare to clusters.",
                sql="""
    WITH last10 AS (
      SELECT mental_state_delta
      FROM features
      WHERE mental_state_delta IS NOT NULL
      ORDER BY timestamp_world_begin DESC
      LIMIT 10
    ),
    q AS (
      SELECT vec_mean(list(mental_state_delta)) AS qvec
      FROM last10
    )
    SELECT id, ts, content,
           cosine_distance(mental_state_delta, (SELECT qvec FROM q)) AS dist
    FROM clusters_temporal
    WHERE mental_state_delta IS NOT NULL
    ORDER BY dist ASC
    LIMIT 15;
    """.strip()
            ))

            # 7) Explain averaged mental state (readable JSON-ish)
            Q.append(dict(
                id="mental_state.explain.mean_24h",
                title="Explain mean mental state over last 24h of features",
                tags=["mental_state", "explain", "vec_mean"],
                description="Use vec_mean(list_aggregate(...)) then explain_mental_state(...) to get a text summary.",
                sql="""
    SELECT explain_mental_state(
      (SELECT vec_mean(list(mental_state_delta))
       FROM features
       WHERE mental_state_delta IS NOT NULL
         AND timestamp_world_begin >= localtimestamp - INTERVAL 1 DAY)
    ) AS explanation;
    """.strip()
            ))

            # 8) Actions → Intentions (expectations) linkage
            Q.append(dict(
                id="actions.intentions.link",
                title="Actions joined with generated intentions",
                tags=["actions", "intentions", "join"],
                description="Explode generated_expectation_ids and join to intentions; order by action time.",
                sql="""
    SELECT action_id, action_type, action_ts,
           intention_id, internal, urgency, incentive_salience,
           affective_valence, fulfilment, intention_ts, intention_content
    FROM actions_expectations
    ORDER BY action_ts ASC, intention_ts ASC
    LIMIT 200;
    """.strip()
            ))

            # 9) Decision making: most recent 100 events ranked against the last feature's embedding
            Q.append(dict(
                id="events.sim.last_feature_emb",
                title="Rank recent events by similarity to the last feature embedding",
                tags=["vectors", "reference_id", "similarity"],
                description="Use get_embedding_by_id on the most recent feature id, then rank recent events.",
                sql="""
    WITH ref AS (
      SELECT id, timestamp_world_begin AS ts
      FROM features
      ORDER BY ts DESC
      LIMIT 1
    ),
    q AS (
      SELECT get_embedding_by_id((SELECT id FROM ref)) AS qvec
    ),
    recent AS (
      SELECT id, kind, subtype, ts, content, embedding
      FROM events
      WHERE ts >= (SELECT ts FROM ref) - INTERVAL 7 DAY
    )
    SELECT id, kind, subtype, ts,
           cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
           content
    FROM recent
    ORDER BY dist ASC
    LIMIT 25;
    """.strip()
            ))

            # 10) Affect rollup: daily averages of affective_valence over time
            Q.append(dict(
                id="affect.daily_rollup",
                title="Daily affect rollup from features",
                tags=["affect", "rollup", "time"],
                description="Compute daily averages of affective metrics to spot trends.",
                sql="""
    SELECT strftime(timestamp_world_begin, '%Y-%m-%d') AS day,
           AVG(affective_valence)     AS valence_avg,
           AVG(incentive_salience)    AS salience_avg,
           AVG(interlocus)            AS interlocus_avg,
           COUNT(*)                   AS n
    FROM features
    WHERE affective_valence IS NOT NULL
    GROUP BY 1
    ORDER BY 1 ASC;
    """.strip()
            ))

            # 11) Conversation outputs: replies in last 48h
            Q.append(dict(
                id="dialogue.replies_48h",
                title="All reply-like actions in the last 48 hours",
                tags=["dialogue", "actions", "recent"],
                description="Filter dialogue_stream for action-like replies.",
                sql="""
    SELECT kind, subtype, id, ts, content
    FROM dialogue_stream
    WHERE kind = 'action'
      AND subtype IN ('Reply','ToolCallAndReply','ToolCall')
      AND ts >= localtimestamp - INTERVAL 2 DAY
    ORDER BY ts ASC;
    """.strip()
            ))

            # 12) Facts: top-k important facts in last 90 days
            Q.append(dict(
                id="facts.top90d",
                title="Top important facts in the last 90 days",
                tags=["facts", "importance"],
                description="Simple triage: filter by recency then highest importance.",
                sql="""
    SELECT id, ts, importance, content
    FROM facts_ranked
    WHERE ts >= localtimestamp - INTERVAL 90 DAY
    ORDER BY importance DESC, ts DESC
    LIMIT 50;
    """.strip()
            ))

            # 13) Event similarity panel vs free-text (events_with_vec)
            Q.append(dict(
                id="events_with_vec.sim.text",
                title="events_with_vec: similarity to free-text",
                tags=["vectors", "events_with_vec", "ranking"],
                description="Use the vec alias to emphasize similarity-only workloads.",
                sql="""
    WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec)
    SELECT id, kind, subtype, ts,
           cosine_distance(vec, (SELECT qvec FROM q)) AS dist,
           content
    FROM events_with_vec
    ORDER BY dist ASC
    LIMIT 25;
    """.strip()
            ))

            # 14) Cluster summaries: temporal vs topical counts
            Q.append(dict(
                id="clusters.type_counts",
                title="Counts of temporal vs topical clusters",
                tags=["clusters", "counts"],
                description="Quick health check of clustering coverage.",
                sql="""
    SELECT cluster_type, COUNT(*) AS n
    FROM clusters
    GROUP BY 1
    ORDER BY n DESC;
    """.strip()
            ))

            # 15) Mentally closest clusters to the last day’s mean state, with explanation text too
            Q.append(dict(
                id="clusters.closest_and_explain",
                title="Closest clusters to last-day mean mental state (with explanation)",
                tags=["mental_state", "clusters", "explain"],
                description="Compute mean state over last day of features, rank clusters, and include an explanation row.",
                sql="""
    WITH dayv AS (
      SELECT vec_mean(list(mental_state_delta)) AS qvec
      FROM features
      WHERE mental_state_delta IS NOT NULL
        AND timestamp_world_begin >= localtimestamp - INTERVAL 1 DAY
    ),
    ranked AS (
      SELECT id, ts, content,
             cosine_distance(mental_state_delta, (SELECT qvec FROM dayv)) AS dist
      FROM clusters_temporal
      WHERE mental_state_delta IS NOT NULL
      ORDER BY dist ASC
      LIMIT 10
    )
    SELECT * FROM ranked;

    -- optional explanation (separate statement)
    -- SELECT explain_mental_state((SELECT qvec FROM dayv)) AS explanation;
    """.strip()
            ))

            return Q

        def render_welcome_llm(self) -> str:
            """
            A compact, LLM-first welcome message:
            - Engine + UDFs + views (from render_help_llm)
            - Then a YAML list of ready-to-use queries with ids/titles/tags/sql
            Replace {{TEXT}} in your caller before execution.
            """
            import yaml
            base = self.render_help_llm()
            qlist = self.query_catalog()
            payload = {"queries": [
                {k: q[k] for k in ("id", "title", "tags", "description", "sql")} for q in qlist
            ]}
            return base + "\n\n" + yaml.safe_dump(payload, sort_keys=False)

        def render_welcome_human(self) -> str:
            """
            Human-oriented welcome:
            - Pretty help header
            - Followed by numbered queries with titles and SQL blocks
            """
            parts = [self.render_help_human()]
            parts.append("queries:")
            for i, q in enumerate(self.query_catalog(), 1):
                parts.append(f"  - {i}. {q['title']} [{q['id']}]")
                parts.append(f"    tags: {', '.join(q['tags'])}")
                parts.append(f"    desc: {q['description']}")
                parts.append("    sql: |")
                for ln in q["sql"].splitlines():
                    parts.append("      " + ln)
            return "\n".join(parts)

        def queries_for_tests(self) -> List[tuple[str, str]]:
            """
            Minimal (id, sql) list for unit tests. No error handling; broken queries will raise on execution.
            Replace {{TEXT}} placeholders in tests before running.
            """
            return [(q["id"], q["sql"]) for q in self.query_catalog()]


    # ---------- Curated query catalog (welcome + tests) ----------
    def query_catalog(self) -> List[Dict[str, Any]]:
        """
        A curated set of commented SQL queries for LLM + regression tests.
        - Each item: {id, title, tags, description, sql}
        - Queries prefer portable constructs and only rely on UDFs registered in this class.
        - Placeholders like {{TEXT}} can be string-replaced by your caller before execution.
        """
        Q: List[Dict[str, Any]] = []

        # 0) Sanity checks
        Q.append(dict(
            id="sanity.counts",
            title="Row counts per main table",
            tags=["sanity","counts"],
            description="Basic counts to ensure tables are visible and non-empty.",
            sql="""
SELECT 'features'   AS table, COUNT(*) AS n FROM features   UNION ALL
SELECT 'stimuli'           , COUNT(*)     FROM stimuli      UNION ALL
SELECT 'actions'           , COUNT(*)     FROM actions      UNION ALL
SELECT 'intentions'        , COUNT(*)     FROM intentions   UNION ALL
SELECT 'narratives'        , COUNT(*)     FROM narratives   UNION ALL
SELECT 'clusters'          , COUNT(*)     FROM clusters     UNION ALL
SELECT 'facts'             , COUNT(*)     FROM facts
;""".strip()
        ))

        # 1) Recent events (24h)
        Q.append(dict(
            id="events.recent_24h",
            title="Events in the last 24 hours",
            tags=["events","time","recent"],
            description="Unified stream to check recent activity windows.",
            sql="""
SELECT id, kind, subtype, ts, content
FROM events
WHERE ts >= localtimestamp - INTERVAL 1 DAY
ORDER BY ts DESC
LIMIT 200;
""".strip()
        ))

        # 2) Dialogue around last user message (±10 minutes)
        Q.append(dict(
            id="dialogue.around_last_user_message",
            title="Context window around the last UserMessage",
            tags=["dialogue","stimuli","actions","context"],
            description="Find the last user message, then show nearby dialogue/actions.",
            sql="""
WITH last_user AS (
  SELECT timestamp_world_begin AS ts
  FROM stimuli
  WHERE stimulus_type = 'UserMessage'
  ORDER BY ts DESC
  LIMIT 1
)
SELECT kind, subtype, id, ts, source, content
FROM dialogue_stream
WHERE ts BETWEEN (SELECT ts FROM last_user) - INTERVAL 10 MINUTE
             AND (SELECT ts FROM last_user) + INTERVAL 10 MINUTE
ORDER BY ts ASC, kind ASC;
""".strip()
        ))

        # 3) Vector search over events using a free-text query
        Q.append(dict(
            id="events.sim.text",
            title="Semantic similarity to text over events",
            tags=["vectors","cosine","query_embedding","ranking"],
            description="Embed query text and rank events by cosine distance.",
            sql="""
WITH q AS (
  SELECT query_embedding({{TEXT}}) AS qvec
)
SELECT id, kind, subtype, ts,
       cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
       content
FROM events
ORDER BY dist ASC
LIMIT 25;
""".strip()
        ))

        # 4) Vector search over facts (important facts first)
        Q.append(dict(
            id="facts.sim.text",
            title="Semantic similarity to text over facts",
            tags=["vectors","facts","ranking"],
            description="Rank declarative facts by similarity to the query text; importance as a tiebreaker.",
            sql="""
WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec)
SELECT id, importance,
       cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
       content
FROM facts
ORDER BY dist ASC, importance DESC
LIMIT 25;
""".strip()
        ))

        # 5) Emotional state: compare clusters to a synthetic 'now' state (EMA=1d)
        Q.append(dict(
            id="clusters.sim.emotional_now",
            title="Clusters similar to current emotional state (EMA=1d)",
            tags=["mental_state","clusters","cosine"],
            description="Use get_emotional_state('now','1d') and compare to clusters_temporal mental_state_delta.",
            sql="""
WITH q AS (SELECT get_emotional_state('now','1d') AS qvec)
SELECT id, ts, content,
       cosine_distance(mental_state_delta, (SELECT qvec FROM q)) AS dist
FROM clusters_temporal
WHERE mental_state_delta IS NOT NULL
ORDER BY dist ASC
LIMIT 15;
""".strip()
        ))

        # 6) Emotional state: average of the last 10 feature deltas
        Q.append(dict(
            id="clusters.sim.last10_features_mean",
            title="Clusters similar to mean mental state of last 10 features",
            tags=["mental_state","vec_mean","list_aggregate","clusters"],
            description="Aggregate last 10 feature mental_state_delta and compare to clusters.",
            sql="""
WITH last10 AS (
  SELECT mental_state_delta
  FROM features
  WHERE mental_state_delta IS NOT NULL
  ORDER BY timestamp_world_begin DESC
  LIMIT 10
),
q AS (
  SELECT vec_mean(list(mental_state_delta)) AS qvec
  FROM last10
)
SELECT id, ts, content,
       cosine_distance(mental_state_delta, (SELECT qvec FROM q)) AS dist
FROM clusters_temporal
WHERE mental_state_delta IS NOT NULL
ORDER BY dist ASC
LIMIT 15;
""".strip()
        ))

        # 7) Explain averaged mental state (readable JSON-ish)
        Q.append(dict(
            id="mental_state.explain.mean_24h",
            title="Explain mean mental state over last 24h of features",
            tags=["mental_state","explain","vec_mean"],
            description="Use vec_mean(list_aggregate(...)) then explain_mental_state(...) to get a text summary.",
            sql="""
SELECT explain_mental_state(
  (SELECT vec_mean(list(mental_state_delta))
   FROM features
   WHERE mental_state_delta IS NOT NULL
     AND timestamp_world_begin >= localtimestamp - INTERVAL 1 DAY)
) AS explanation;
""".strip()
        ))

        # 8) Actions → Intentions (expectations) linkage
        Q.append(dict(
            id="actions.intentions.link",
            title="Actions joined with generated intentions",
            tags=["actions","intentions","join"],
            description="Explode generated_expectation_ids and join to intentions; order by action time.",
            sql="""
SELECT action_id, action_type, action_ts,
       intention_id, internal, urgency, incentive_salience,
       affective_valence, fulfilment, intention_ts, intention_content
FROM actions_expectations
ORDER BY action_ts ASC, intention_ts ASC
LIMIT 200;
""".strip()
        ))

        # 9) Decision making: most recent 100 events ranked against the last feature's embedding
        Q.append(dict(
            id="events.sim.last_feature_emb",
            title="Rank recent events by similarity to the last feature embedding",
            tags=["vectors","reference_id","similarity"],
            description="Use get_embedding_by_id on the most recent feature id, then rank recent events.",
            sql="""
WITH ref AS (
  SELECT id, timestamp_world_begin AS ts
  FROM features
  ORDER BY ts DESC
  LIMIT 1
),
q AS (
  SELECT get_embedding_by_id((SELECT id FROM ref)) AS qvec
),
recent AS (
  SELECT id, kind, subtype, ts, content, embedding
  FROM events
  WHERE ts >= (SELECT ts FROM ref) - INTERVAL 7 DAY
)
SELECT id, kind, subtype, ts,
       cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
       content
FROM recent
ORDER BY dist ASC
LIMIT 25;
""".strip()
        ))

        # 10) Affect rollup: daily averages of affective_valence over time
        Q.append(dict(
            id="affect.daily_rollup",
            title="Daily affect rollup from features",
            tags=["affect","rollup","time"],
            description="Compute daily averages of affective metrics to spot trends.",
            sql="""
SELECT strftime(timestamp_world_begin, '%Y-%m-%d') AS day,
       AVG(affective_valence)     AS valence_avg,
       AVG(incentive_salience)    AS salience_avg,
       AVG(interlocus)            AS interlocus_avg,
       COUNT(*)                   AS n
FROM features
WHERE affective_valence IS NOT NULL
GROUP BY 1
ORDER BY 1 ASC;
""".strip()
        ))

        # 11) Conversation outputs: replies in last 48h
        Q.append(dict(
            id="dialogue.replies_48h",
            title="All reply-like actions in the last 48 hours",
            tags=["dialogue","actions","recent"],
            description="Filter dialogue_stream for action-like replies.",
            sql="""
SELECT kind, subtype, id, ts, content
FROM dialogue_stream
WHERE kind = 'action'
  AND subtype IN ('Reply','ToolCallAndReply','ToolCall')
  AND ts >= localtimestamp - INTERVAL 2 DAY
ORDER BY ts ASC;
""".strip()
        ))

        # 12) Facts: top-k important facts in last 90 days
        Q.append(dict(
            id="facts.top90d",
            title="Top important facts in the last 90 days",
            tags=["facts","importance"],
            description="Simple triage: filter by recency then highest importance.",
            sql="""
SELECT id, ts, importance, content
FROM facts_ranked
WHERE ts >= localtimestamp - INTERVAL 90 DAY
ORDER BY importance DESC, ts DESC
LIMIT 50;
""".strip()
        ))

        # 13) Event similarity panel vs free-text (events_with_vec)
        Q.append(dict(
            id="events_with_vec.sim.text",
            title="events_with_vec: similarity to free-text",
            tags=["vectors","events_with_vec","ranking"],
            description="Use the vec alias to emphasize similarity-only workloads.",
            sql="""
WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec)
SELECT id, kind, subtype, ts,
       cosine_distance(vec, (SELECT qvec FROM q)) AS dist,
       content
FROM events_with_vec
ORDER BY dist ASC
LIMIT 25;
""".strip()
        ))

        # 14) Cluster summaries: temporal vs topical counts
        Q.append(dict(
            id="clusters.type_counts",
            title="Counts of temporal vs topical clusters",
            tags=["clusters","counts"],
            description="Quick health check of clustering coverage.",
            sql="""
SELECT cluster_type, COUNT(*) AS n
FROM clusters
GROUP BY 1
ORDER BY n DESC;
""".strip()
        ))

        # 15) Mentally closest clusters to the last day’s mean state, with explanation text too
        Q.append(dict(
            id="clusters.closest_and_explain",
            title="Closest clusters to last-day mean mental state (with explanation)",
            tags=["mental_state","clusters","explain"],
            description="Compute mean state over last day of features, rank clusters, and include an explanation row.",
            sql="""
WITH dayv AS (
  SELECT vec_mean(list(mental_state_delta)) AS qvec
  FROM features
  WHERE mental_state_delta IS NOT NULL
    AND timestamp_world_begin >= localtimestamp - INTERVAL 1 DAY
),
ranked AS (
  SELECT id, ts, content,
         cosine_distance(mental_state_delta, (SELECT qvec FROM dayv)) AS dist
  FROM clusters_temporal
  WHERE mental_state_delta IS NOT NULL
  ORDER BY dist ASC
  LIMIT 10
)
SELECT * FROM ranked;

-- optional explanation (separate statement)
-- SELECT explain_mental_state((SELECT qvec FROM dayv)) AS explanation;
""".strip()
        ))

        # 1) Preferences (music/food/etc.) synthesized from dialogue + facts
        Q.append(dict(
            id="synth.preferences.topic",
            title="Synthesize user/agent preference for a topic (e.g., music)",
            tags=["synthesis", "preferences", "semantic", "dialogue", "facts"],
            description=(
                "There is no dedicated 'preferences' table. Retrieve likely evidence from dialogue and facts, "
                "rank semantically to {{TEXT}} (e.g., 'Emmy music preference'), then the LLM must read the rows and "
                "compose a concise answer citing the strongest evidence."
            ),
            sql="""
        WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec),
        pool AS (
          SELECT id, 'event' AS kind, ts, content, embedding
          FROM events
          WHERE content ILIKE '%like%' OR content ILIKE '%prefer%' OR content ILIKE '%favorite%' 
             OR content ILIKE '%music%' OR content ILIKE '%song%' OR content ILIKE '%playlist%' OR content ILIKE '%genre%'
          UNION ALL
          SELECT id, 'fact' AS kind, ts, content, embedding
          FROM facts
          WHERE content ILIKE '%like%' OR content ILIKE '%prefer%' OR content ILIKE '%favorite%'
             OR content ILIKE '%music%' OR content ILIKE '%song%' OR content ILIKE '%playlist%' OR content ILIKE '%genre%'
        )
        SELECT kind, id, ts,
               cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
               content
        FROM pool
        ORDER BY dist ASC, ts DESC
        LIMIT 50;
        """.strip()
        ))

        # 2) Routine inference: extract morning/evening habits from last {{DAYS}} days
        Q.append(dict(
            id="synth.routine.windowed",
            title="Synthesize daily routine from recent dialogue/features",
            tags=["synthesis", "routine", "temporal", "semantic"],
            description=(
                "No 'routine' table. Pull semantically similar events to the topic ({{TEXT}}, e.g., 'morning routine'), "
                "limited to last {{DAYS}} days; then summarize likely routine steps and times from the evidence."
            ),
            sql="""
        WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec),
        recent AS (
          SELECT id, kind, subtype, ts, content, embedding
          FROM events
          WHERE ts >= localtimestamp - INTERVAL {{DAYS}} DAY
        ),
        ranked AS (
          SELECT id, kind, subtype, ts,
                 cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
                 content
          FROM recent
          ORDER BY dist ASC
          LIMIT 120
        )
        SELECT * FROM ranked ORDER BY ts ASC;
        """.strip()
        ))

        # 3) Open commitments: intentions that look like promises/tasks not fulfilled
        Q.append(dict(
            id="synth.commitments.open",
            title="Synthesize open commitments/promises",
            tags=["synthesis", "intentions", "commitments"],
            description=(
                "No 'commitments' table. Use intentions with low fulfilment, rank by urgency/incentive_salience, "
                "and synthesize a prioritized TODO/commitments list."
            ),
            sql="""
        SELECT id, internal, urgency, incentive_salience, affective_valence, fulfilment,
               timestamp_world_begin AS ts, content
        FROM intentions
        WHERE fulfilment IS NOT NULL AND fulfilment < 0.5
        ORDER BY urgency DESC, incentive_salience DESC, ts DESC
        LIMIT 100;
        """.strip()
        ))

        # 4) Safety concerns: synthesize risk assessment from evidence
        Q.append(dict(
            id="synth.safety.concerns",
            title="Synthesize safety concerns from recent context",
            tags=["synthesis", "safety", "semantic"],
            description=(
                "There is no 'safety' table. Pull recent events whose text includes risk terms or are semantically "
                "close to {{TEXT}} (e.g., 'safety risk'), then summarize risks and recommended mitigations."
            ),
            sql="""
        WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec),
        recent AS (
          SELECT id, kind, subtype, ts, content, embedding
          FROM events
          WHERE ts >= localtimestamp - INTERVAL 14 DAY
        ),
        pool AS (
          SELECT * FROM recent
          WHERE content ILIKE '%risk%' OR content ILIKE '%unsafe%' OR content ILIKE '%harm%' OR content ILIKE '%danger%'
             OR content ILIKE '%warning%' OR content ILIKE '%incident%'
          UNION ALL
          SELECT * FROM recent
        ),
        ranked AS (
          SELECT id, kind, subtype, ts,
                 cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
                 content
          FROM pool
        )
        SELECT * FROM ranked ORDER BY dist ASC, ts DESC LIMIT 120;
        """.strip()
        ))

        # 5) Relationship dynamics: synthesize stance/affect toward the user
        Q.append(dict(
            id="synth.relationship.user",
            title="Synthesize relationship stance toward the user",
            tags=["synthesis", "relationship", "affect", "semantic"],
            description=(
                "No 'relationship' table. Gather dialogue + features tied to feelings/subjective experience "
                "that are semantically close to {{TEXT}} (e.g., 'relationship with user'), then summarize stance."
            ),
            sql="""
        WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec),
        pool AS (
          SELECT id, 'affect' AS src, timestamp_world_begin AS ts, content, embedding
          FROM features
          WHERE feature_type IN ('Feeling','SubjectiveExperience','AttentionFocus')
          UNION ALL
          SELECT id, 'dialogue' AS src, ts, content, embedding
          FROM dialogue_stream
        ),
        ranked AS (
          SELECT src, id, ts,
                 cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
                 content
          FROM pool
        )
        SELECT * FROM ranked ORDER BY dist ASC, ts DESC LIMIT 150;
        """.strip()
        ))

        # 6) Topic timeline: synthesize a narrative over time
        Q.append(dict(
            id="synth.topic.timeline",
            title="Synthesize a topic-oriented timeline",
            tags=["synthesis", "timeline", "semantic"],
            description=(
                "No 'topic timeline' table. Rank events by similarity to {{TEXT}} and return a time-ordered slice; "
                "the agent must narrate the evolution over time."
            ),
            sql="""
        WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec),
        ranked AS (
          SELECT id, kind, subtype, ts,
                 cosine_distance(embedding, (SELECT qvec FROM q)) AS dist,
                 content
          FROM events
        )
        SELECT * FROM ranked ORDER BY dist ASC, ts ASC LIMIT 200;
        """.strip()
        ))

        # 7) Mood trend for topic: synthesize trend summary
        Q.append(dict(
            id="synth.topic.mood_trend",
            title="Synthesize mood trend around a topic",
            tags=["synthesis", "affect", "trend", "semantic"],
            description=(
                "No direct 'mood per topic'. Select Feeling-like features semantically close to {{TEXT}} "
                "and roll up daily averages; then synthesize the trend interpretation."
            ),
            sql="""
        WITH q AS (SELECT query_embedding({{TEXT}}) AS qvec),
        cand AS (
          SELECT timestamp_world_begin AS ts,
                 affective_valence, incentive_salience, interlocus,
                 content, embedding
          FROM features
          WHERE feature_type IN ('Feeling','SubjectiveExperience')
        ),
        ranked AS (
          SELECT *,
                 cosine_distance(embedding, (SELECT qvec FROM q)) AS dist
          FROM cand
        ),
        daily AS (
          SELECT strftime(ts, '%Y-%m-%d') AS day,
                 AVG(affective_valence)   AS valence_avg,
                 AVG(incentive_salience)  AS salience_avg,
                 AVG(interlocus)          AS interlocus_avg,
                 COUNT(*)                  AS n
          FROM ranked
          WHERE dist <= (SELECT PERCENTILE_CONT(0.4) FROM (SELECT dist FROM ranked)) -- keep closest 40%
          GROUP BY 1
        )
        SELECT * FROM daily ORDER BY day ASC;
        """.strip()
        ))

        # 8) “What changed the AI’s mind?”: intentions around actions
        Q.append(dict(
            id="synth.decision.rationale",
            title="Synthesize rationale: which observations preceded actions?",
            tags=["synthesis", "intentions", "actions", "explanations"],
            description=(
                "There is no 'rationale' table. Use actions_expectations and nearby dialogue to extract evidence "
                "of why a decision was made; agent synthesizes the narrative."
            ),
            sql="""
        WITH near AS (
          SELECT a.action_id, a.action_type, a.action_ts,
                 i.intention_id, i.intention_ts, i.intention_content, i.urgency, i.incentive_salience, i.fulfilment
          FROM actions_expectations a
          JOIN intentions i ON i.id = a.intention_id
          WHERE a.action_ts >= localtimestamp - INTERVAL 7 DAY
        ),
        ctx AS (
          SELECT d.id, d.kind, d.subtype, d.ts, d.content, d.embedding
          FROM dialogue_stream d
          WHERE d.ts >= localtimestamp - INTERVAL 7 DAY
        )
        SELECT *
        FROM near
        JOIN ctx
          ON ctx.ts BETWEEN near.action_ts - INTERVAL 10 MINUTE AND near.action_ts + INTERVAL 10 MINUTE
        ORDER BY near.action_ts ASC, ctx.ts ASC
        LIMIT 400;
        """.strip()
        ))

        # 9) Closest clusters to a synthetic emotional state (EMA)
        Q.append(dict(
            id="synth.emotion.closest_clusters",
            title="Synthesize emotional context via nearest clusters",
            tags=["synthesis", "mental_state", "clusters", "cosine"],
            description=(
                "No 'current emotion' table. Use get_emotional_state({{WHEN}}, {{EMA}}) and retrieve closest temporal clusters; "
                "agent summarizes what those clusters imply about current context."
            ),
            sql="""
        WITH q AS (SELECT get_emotional_state({{WHEN}}, {{EMA}}) AS qvec)
        SELECT id, ts, content,
               cosine_distance(mental_state_delta, (SELECT qvec FROM q)) AS dist
        FROM clusters_temporal
        WHERE mental_state_delta IS NOT NULL
        ORDER BY dist ASC
        LIMIT 20;
        """.strip()
        ))

        # 10) “What are we working on lately?”: synthesize from high-salience features
        Q.append(dict(
            id="synth.focus.recent",
            title="Synthesize current focus from high-salience features",
            tags=["synthesis", "focus", "salience"],
            description=(
                "No 'focus' table. Pull recent features with high incentive_salience and summarize main threads."
            ),
            sql="""
        SELECT id, feature_type, timestamp_world_begin AS ts,
               incentive_salience, affective_valence, interlocus,
               content
        FROM features
        WHERE timestamp_world_begin >= localtimestamp - INTERVAL 14 DAY
          AND incentive_salience IS NOT NULL
        ORDER BY incentive_salience DESC, ts DESC
        LIMIT 150;
        """.strip()
        ))

        return Q

    def render_welcome_llm(self) -> str:
        """
        A compact, LLM-first welcome message:
        - Engine + UDFs + views (from render_help_llm)
        - Then a YAML list of ready-to-use queries with ids/titles/tags/sql
        Replace {{TEXT}} in your caller before execution.
        """
        import yaml
        base = self.render_help_llm()
        qlist = self.query_catalog()
        payload = {"queries": [
            {k: q[k] for k in ("id","title","tags","description","sql")} for q in qlist
        ]}
        return base + "\n\n" + yaml.safe_dump(payload, sort_keys=False)

    def render_welcome_human(self) -> str:
        """
        Human-oriented welcome:
        - Pretty help header
        - Followed by numbered queries with titles and SQL blocks
        """
        parts = [self.render_help_human()]
        parts.append("queries:")
        for i, q in enumerate(self.query_catalog(), 1):
            parts.append(f"  - {i}. {q['title']} [{q['id']}]")
            parts.append(f"    tags: {', '.join(q['tags'])}")
            parts.append(f"    desc: {q['description']}")
            parts.append("    sql: |")
            for ln in q["sql"].splitlines():
                parts.append("      " + ln)
        return "\n".join(parts)

    def queries_for_tests(self) -> List[tuple[str,str]]:
        """
        Minimal (id, sql) list for unit tests. No error handling; broken queries will raise on execution.
        Replace {{TEXT}} placeholders in tests before running.
        """
        return [(q["id"], q["sql"]) for q in self.query_catalog()]
