#!/usr/bin/env python3
"""
memview.py — semi-visual, LLM-friendly explorer with MCP-ready command surface.

Key features:
- Commands parsed with argparse-per-command (good errors, --flags, help).
- Dates parsed via dateparser (if present) else python-dateutil else ISO.
- Every command returns a STRING (render buffer). REPL only prints it.
- Mutating commands show only CHANGED state keys + compact CSV view.
- Full state can be shown periodically or on demand.

Commands (see `help` or `man`):
  cd / | cd .. | cd PATH
  filter set [--from DATE] [--to DATE] [--ext EXT ...] [--size-min N] [--size-max N] [--ascii] [--no-hidden]
  filter clear
  sort --key KEY[:a|:d] [--key ...]         # name, ext, size, mtime
  set-preview-tokens N
  set-view-length N
  set-state-full-every N
  show_state
  scroll up | scroll down
  cat PATH
  head PATH [--n 200]
  tail PATH [--n 200]
  search --query "text" [-R]
  reset_search
  add-file-to-buffer PATH [--range "lines:A-B" | --head N | --tail N]
  add-note-to-buffer "note text"
  show_buffer
  clear_buffer
  compose_answer [--max-chars 8000]
  ls -l
  help
  man
  quit | exit
"""

import os, sys, shlex, shutil, subprocess, datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml  # PyYAML

# -------- optional date libs --------
_DATEPARSER = None
_DATEUTIL = None
try:
    import dateparser as _DATEPARSER  # type: ignore
except Exception:
    pass
try:
    from dateutil import parser as _DATEUTIL  # type: ignore
except Exception:
    pass


# -------- small utils --------

def now_str() -> str:
    """Local datetime (no tz) for display."""
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def which(cmd: str) -> bool:
    """True if command exists on PATH."""
    return shutil.which(cmd) is not None

def parse_size(s: Optional[str]) -> Optional[int]:
    """Parse sizes like 10kb, 5mb, 2gb to bytes; None passes through."""
    if not s: return None
    t = s.strip().lower().replace(" ", "")
    units = {"b":1,"kb":1024,"mb":1024**2,"gb":1024**3}
    for u in ("gb","mb","kb","b"):
        if t.endswith(u):
            return int(float(t[:-len(u)]) * units[u])
    return int(t)

def parse_date_any(s: Optional[str]) -> Optional[dt.datetime]:
    """Best-effort parse of many date forms: dateparser → dateutil → ISO."""
    if not s: return None
    if _DATEPARSER:
        d = _DATEPARSER.parse(s)
        if d: return d
    if _DATEUTIL:
        try: return _DATEUTIL.parse(s)
        except Exception: pass
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None

def ascii_probably(path: Path) -> bool:
    """Use `file --mime` if available; else attempt utf-8 decode on first bytes."""
    if which("file"):
        try:
            out = subprocess.check_output(["file", "--mime", str(path)], text=True)
            return ("charset=us-ascii" in out) or ("charset=utf-8" in out)
        except Exception:
            pass
    try:
        with open(path, "rb") as f:
            data = f.read(4096)
        data.decode("utf-8")
        return True
    except Exception:
        return False

def tokenize_whitespace(s: str, n: int) -> str:
    """Naive whitespace token cap."""
    if n <= 0: return s
    parts = s.split()
    return " ".join(parts[:n])

def run_search(cwd: Path, query: str, recursive: bool=True) -> list[tuple[str,int,str]]:
    """ripgrep if available, else grep. Returns (rel_path, line, text)."""
    hits = []
    if which("rg"):
        args = ["rg", "-n", "--no-heading", query]
        if recursive: args.append("--hidden")
        try:
            out = subprocess.check_output(args, cwd=str(cwd), text=True, errors="ignore")
            for line in out.splitlines():
                parts = line.split(":", 2)
                if len(parts)==3:
                    hits.append((parts[0], int(parts[1]), parts[2]))
        except subprocess.CalledProcessError:
            pass
        return hits
    args = ["grep", "-RIn", query, "." if recursive else ""]
    try:
        out = subprocess.check_output(args, cwd=str(cwd), text=True, errors="ignore")
        for line in out.splitlines():
            parts = line.split(":", 2)
            if len(parts)==3:
                hits.append((parts[0], int(parts[1]), parts[2]))
    except subprocess.CalledProcessError:
        pass
    return hits

def format_csv(headers: list[str], rows: list[list[str]], align: bool) -> str:
    """
    Render CSV either compact (semicolon-separated) or aligned for humans.
    If align=True, left-align text columns and right-align purely numeric columns.
    """
    if not align:
        lines = [";".join(headers)]
        for r in rows:
            lines.append(";".join(r))
        return "\n".join(lines)

    # compute widths
    cols = len(headers)
    widths = [len(h) for h in headers]
    is_numeric = [True] * cols

    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
            if is_numeric[i] and not cell.replace("-", "", 1).isdigit():
                # treat as numeric only if it looks like an integer; keep simple
                is_numeric[i] = False

    def fmt_row(row):
        parts = []
        for i, cell in enumerate(row):
            if is_numeric[i]:
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return "  ".join(parts)  # double-space for readability

    lines = [fmt_row(headers)]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)

def dump_yaml(data: dict, indent: int) -> str:
    """
    Dump YAML deterministically with a configurable indent.
    Avoid aliases & keep keys in insertion order.
    """
    return yaml.safe_dump(
        data,
        sort_keys=False,
        indent=indent,
        default_flow_style=False
    ).rstrip()
# -------- state --------

class State:
    """Mutable UI state for REPL & MCP mapping."""
    def __init__(self):
        self.cwd: Path = Path.cwd()
        self.filters: Dict[str, Any] = {
            "from": None, "to": None,
            "ext": [],                # list[str] like [".txt",".md"]
            "size_min": None, "size_max": None,
            "flags": {"ascii": False, "no_hidden": False},
        }
        self.sort_keys: List[str] = []   # ["name:a","size:d",...]
        self.preview_tokens: int = 64
        self.view_length: int = 8
        self.visible: Dict[str,int] = {"offset": 0, "count_total": 0}
        self.search: Dict[str, Any] = {"query": None, "recursive": True, "hits": []}
        self.buffer: Dict[str, Any] = {"items": []}
        self.capabilities = {
            "rg": which("rg"), "grep": which("grep"),
            "file": which("file"), "head": which("head"),
            "tail": which("tail"), "cat": which("cat")
        }
        self.last_preview: Optional[Dict[str, Any]] = None
        self.answer_preview: Optional[str] = None
        self.turn_count: int = 0
        self.state_full_every: int = 6  # full dump cadence
        self.yaml_indent: int = 2
        self.align_columns: bool = True

    def reset_paging(self):
        self.visible["offset"] = 0

# -------- directory & view --------

def scan_dir(state: State) -> list[Dict[str, Any]]:
    """Collect basic props for current dir entries (+ . and ..)."""
    rows: list[Dict[str,Any]] = []
    for e in os.scandir(state.cwd):
        typ = "folder" if e.is_dir(follow_symlinks=False) else ("symlink" if e.is_symlink() else "file")
        name = e.name
        ext = (Path(name).suffix.lower() if typ=="file" else "-") or "-"
        try:
            st = e.stat(follow_symlinks=False)
            size = st.st_size
            mtime_iso = dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            size, mtime_iso = -1, "-"
        hidden = name.startswith(".")
        is_ascii = ascii_probably(Path(state.cwd)/name) if typ=="file" else True
        rows.append({
            "name": name, "type": typ, "ext": ext, "size": size,
            "mtime_iso": mtime_iso, "path": str(Path(state.cwd)/name),
            "hidden": hidden, "ascii": is_ascii
        })
    rows.insert(0, {"name": ".", "type":"self","ext":"-","size":-1,"mtime_iso":"-","path":str(state.cwd),"hidden":False,"ascii":True})
    rows.insert(1, {"name": "..","type":"up","ext":"-","size":-1,"mtime_iso":"-","path":str(state.cwd.parent),"hidden":False,"ascii":True})
    return rows

def apply_filters(state: State, rows: list[Dict[str,Any]]) -> list[Dict[str,Any]]:
    """Filter by hidden/ascii/ext/size/date."""
    f = state.filters
    out = []
    for r in rows:
        if f["flags"]["no_hidden"] and r["hidden"]:
            continue
        if f["flags"]["ascii"] and r["type"]=="file" and not r["ascii"]:
            continue
        if f["ext"]:
            if r["type"]=="file" and r["ext"] not in f["ext"]:
                continue
        if f["size_min"] is not None and r["size"]>=0 and r["size"] < f["size_min"]:
            continue
        if f["size_max"] is not None and r["size"]>=0 and r["size"] > f["size_max"]:
            continue
        if (f["from"] or f["to"]) and r["mtime_iso"]!="-":
            m = dt.datetime.fromisoformat(r["mtime_iso"])
            if f["from"] and m < f["from"]: continue
            if f["to"] and m > f["to"]: continue
        out.append(r)
    return out

def _invert_key(v):
    if isinstance(v,(int,float)): return -v
    if isinstance(v,str): return "".join(chr(255-ord(c)) for c in v)
    return v

def apply_sort(state: State, rows: list[Dict[str,Any]]) -> list[Dict[str,Any]]:
    """Multi-key sort. keys: name, ext, size, mtime."""
    if not state.sort_keys:
        return sorted(rows, key=lambda r: (r["type"]!="folder", r["name"].lower()))
    def build(r):
        out=[]
        for spec in state.sort_keys:
            field, direction = (spec.split(":",1)+["a"])[:2] if ":" in spec else (spec,"a")
            v = r.get({"mtime":"mtime_iso"}.get(field, field), None)
            if field=="name" and isinstance(v,str): v=v.lower()
            out.append(v if direction=="a" else _invert_key(v))
        return tuple(out)
    return sorted(rows, key=build)

# -------- rendering (string buffer) --------
def render_full_state(state: State) -> str:
    """Full YAML state dump built from a dict, using configurable indent."""
    f = state.filters
    data = {
        "state": {
            "cwd": str(state.cwd),
            "filters": {
                "from": f["from"].strftime("%Y-%m-%d %H:%M:%S") if f["from"] else None,
                "to": f["to"].strftime("%Y-%m-%d %H:%M:%S") if f["to"] else None,
                "ext": f["ext"],
                "size_min": f["size_min"],
                "size_max": f["size_max"],
                "flags": {
                    "ascii": bool(f["flags"]["ascii"]),
                    "no_hidden": bool(f["flags"]["no_hidden"]),
                },
            },
            "sort": state.sort_keys,
            "preview_tokens": state.preview_tokens,
            "view_length": state.view_length,
            "visible": {
                "offset": state.visible["offset"],
                "count_total": state.visible["count_total"],
            },
            "buffer": {"items": len(state.buffer["items"])},
            "capabilities": state.capabilities,
            "state_full_every": state.state_full_every,
            "yaml_indent": state.yaml_indent,
            "align_columns": state.align_columns,
        }
    }
    return dump_yaml(data, state.yaml_indent)

def render_compact(state: State, rows: list[dict]) -> str:
    """
    Build a dict for the header (prompt + now + small metadata),
    YAML-dump it, then append CSV view text (aligned optionally).
    """
    start = state.visible["offset"]
    end = min(start + state.view_length, len(rows))

    head = {
        "prompt": f"agent@memory:{state.cwd}",
        "now": now_str(),
        "view": {
            "window": {"start": start, "end": end, "total": len(rows)},
        },
    }

    # Prepare CSV data
    headers = ["NAME", "TYPE", "EXT", "SIZE_B", "MTIME_ISO", "PATH"]
    body = []
    for r in rows[start:end]:
        body.append([
            str(r["name"]),
            str(r["type"]),
            str(r["ext"]),
            str(r["size"]),
            str(r["mtime_iso"]),
            str(r["path"]),
        ])

    # Optional sections (search/preview/answer) into YAML header
    if state.search["query"]:
        hits = state.search["hits"]
        sample = hits[: min(5, len(hits))]
        head["search_hits"] = {
            "query": state.search["query"],
            "recursive": bool(state.search["recursive"]),
            "total_matches": len(hits),
            "sample": [{"file": p, "line": ln, "text": txt} for (p, ln, txt) in sample],
        }

    if state.last_preview:
        meta = {k: v for k, v in state.last_preview.items() if k != "content"}
        content = state.last_preview.get("content", "")
        head["preview"] = {
            **meta,
            "content_first_tokens": state.preview_tokens,
        }
        # Note: actual content stays in CSV area only if desired; for now keep it out
        # to avoid mixing YAML with possibly huge text blocks. Agents can request show_buffer/cat.

    if state.answer_preview:
        head["answer"] = {"chars": len(state.answer_preview)}

    # Dump YAML header
    yaml_head = dump_yaml(head, state.yaml_indent)

    # CSV (aligned if enabled)
    csv_text = format_csv(headers, body, align=state.align_columns)

    # If we want to also display preview content (respecting preview_tokens) after CSV:
    preview_block = ""
    if state.last_preview:
        content = state.last_preview.get("content", "")
        tok = tokenize_whitespace(content, state.preview_tokens)
        preview_lines = ["preview_content: |"]
        preview_lines += [f"  {L}" for L in tok.splitlines()]
        preview_block = "\n" + "\n".join(preview_lines)

    # If there is a composed answer, append as YAML block
    answer_block = ""
    if state.answer_preview:
        answer_lines = ["answer_content: |"]
        answer_lines += [f"  {L}" for L in state.answer_preview.splitlines()]
        answer_block = "\n" + "\n".join(answer_lines)

    return f"{yaml_head}\n{csv_text}{preview_block}{answer_block}"

def render_changed(changes: Dict[str, Any]) -> str:
    """Render only changed keys for state mutations."""
    if not changes: return ""
    lines = ["changed:"]
    for k,v in changes.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

def recompute_rows(state: State) -> list[Dict[str,Any]]:
    rows = apply_sort(state, apply_filters(state, scan_dir(state)))
    state.visible["count_total"] = len(rows)
    return rows

def finalize_output(state: State, rows: list[Dict[str,Any]], changes: Dict[str,Any] | None = None) -> str:
    """Compose the final string buffer for a command."""
    parts = []
    if changes:
        parts.append(render_changed(changes))
    # periodic full state
    state.turn_count += 1
    if state.turn_count % max(1, state.state_full_every) == 0:
        parts.append(render_full_state(state))
    parts.append(render_compact(state, rows))
    return "\n".join(p for p in parts if p)

# -------- per-command argparsers --------

import argparse

def parse_line_with(parser: argparse.ArgumentParser, args: List[str]):
    """Run argparse against args list and return namespace or raise SystemExit (caught by caller)."""
    return parser.parse_args(args)

# ---- command handlers (each returns STRING) ----

def cmd_cd(state: State, argv: List[str]) -> str:
    """cd / | cd .. | cd PATH"""
    p = argparse.ArgumentParser(prog="cd", add_help=True)
    p.add_argument("path", nargs="?", default=str(Path.home()))
    ns = parse_line_with(p, argv)
    old = str(state.cwd)
    if ns.path == "/": state.cwd = Path("/")
    elif ns.path == "..": state.cwd = state.cwd.parent
    else:
        pth = Path(ns.path)
        state.cwd = pth if pth.is_absolute() else (state.cwd / pth)
    state.reset_paging(); state.last_preview=None
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"cwd": f"{old} -> {state.cwd}"})

def cmd_filter_set(state: State, argv: List[str]) -> str:
    """filter set --from DATE --to DATE --ext .txt --ext .md --size-min 10kb --size-max 5mb --ascii --no-hidden"""
    p = argparse.ArgumentParser(prog="filter set")
    p.add_argument("--from", dest="date_from")
    p.add_argument("--to", dest="date_to")
    p.add_argument("--ext", action="append", default=[])
    p.add_argument("--size-min")
    p.add_argument("--size-max")
    p.add_argument("--ascii", action="store_true")
    p.add_argument("--no-hidden", action="store_true")
    ns = parse_line_with(p, argv)

    exts = [(e if e.startswith(".") else "."+e).lower() for e in ns.ext]
    d_from = parse_date_any(ns.date_from)
    d_to = parse_date_any(ns.date_to)
    if d_to and ns.date_to and len(ns.date_to.strip())<=10:
        # end-of-day if only date
        d_to = d_to.replace(hour=23, minute=59, second=59, microsecond=999999)

    changes = {}
    state.filters["from"] = d_from; changes["filters.from"] = d_from.isoformat() if d_from else "null"
    state.filters["to"] = d_to;     changes["filters.to"] = d_to.isoformat() if d_to else "null"
    state.filters["ext"] = exts;    changes["filters.ext"] = exts
    state.filters["size_min"] = parse_size(ns.size_min); changes["filters.size_min"] = state.filters["size_min"]
    state.filters["size_max"] = parse_size(ns.size_max); changes["filters.size_max"] = state.filters["size_max"]
    state.filters["flags"]["ascii"] = bool(ns.ascii); changes["filters.flags.ascii"] = ns.ascii
    state.filters["flags"]["no_hidden"] = bool(ns.no_hidden); changes["filters.flags.no_hidden"] = ns.no_hidden

    state.reset_paging(); state.last_preview=None
    rows = recompute_rows(state)
    return finalize_output(state, rows, changes)

def cmd_filter_clear(state: State, argv: List[str]) -> str:
    """filter clear"""
    _ = argparse.ArgumentParser(prog="filter clear").parse_args(argv)
    state.filters = {"from": None, "to": None, "ext": [], "size_min": None, "size_max": None, "flags": {"ascii": False, "no_hidden": False}}
    state.reset_paging(); state.last_preview=None
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"filters": "cleared"})

def cmd_sort(state: State, argv: List[str]) -> str:
    """sort --key name:a --key size:d ..."""
    p = argparse.ArgumentParser(prog="sort")
    p.add_argument("--key", action="append", required=True, help="field[:a|:d], fields: name, ext, size, mtime")
    ns = parse_line_with(p, argv)
    state.sort_keys = ns.key
    state.reset_paging()
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"sort": ns.key})

def cmd_set_preview_tokens(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="set-preview-tokens")
    p.add_argument("n", type=int)
    ns = parse_line_with(p, argv)
    state.preview_tokens = ns.n
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"preview_tokens": ns.n})

def cmd_set_view_length(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="set-view-length")
    p.add_argument("n", type=int)
    ns = parse_line_with(p, argv)
    state.view_length = ns.n
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"view_length": ns.n})

def cmd_set_state_full_every(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="set-state-full-every")
    p.add_argument("n", type=int)
    ns = parse_line_with(p, argv)
    state.state_full_every = max(1, ns.n)
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"state_full_every": state.state_full_every})

def cmd_show_state(state: State, argv: List[str]) -> str:
    _ = argparse.ArgumentParser(prog="show_state").parse_args(argv)
    rows = recompute_rows(state)
    return "\n".join([render_full_state(state), render_compact(state, rows)])

def _read_slice(path: Path, mode: str, lines: Optional[str], head_n: Optional[int], tail_n: Optional[int]) -> str:
    if mode=="cat" and which("cat"):
        try: return subprocess.check_output(["cat", str(path)], text=True, errors="ignore")
        except Exception: pass
    if mode=="head" and which("head"):
        try: return subprocess.check_output(["head", f"-n{head_n or 200}", str(path)], text=True, errors="ignore")
        except Exception: pass
    if mode=="tail" and which("tail"):
        try: return subprocess.check_output(["tail", f"-n{tail_n or 200}", str(path)], text=True, errors="ignore")
        except Exception: pass
    # Python fallback
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        if lines and ":" in lines:
            a,b = lines.split(":")[1].split("-",1)
            a,b = int(a), int(b)
            out=[];
            for i, line in enumerate(f, start=1):
                if a <= i <= b: out.append(line)
                if i > b: break
            return "".join(out)
        if mode=="head":
            n=head_n or 200; return "".join([next(f) for _ in range(n) if True])
        if mode=="tail":
            n=tail_n or 200; return "".join(f.readlines()[-n:])
        return f.read()

def _preview(state: State, path: Path, mode: str, lines: Optional[str], head_n: Optional[int], tail_n: Optional[int]):
    content = _read_slice(path, mode, lines, head_n, tail_n)
    state.last_preview = {"path": str(path), "mode": mode, "range": lines or "", "bytes": len(content.encode("utf-8","ignore")), "content": content}

def cmd_cat(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="cat")
    p.add_argument("path")
    ns = parse_line_with(p, argv)
    path = Path(ns.path);
    if not path.is_absolute(): path = state.cwd / path
    _preview(state, path, "cat", None, None, None)
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"preview": f"cat {path}"})

def cmd_head(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="head")
    p.add_argument("path"); p.add_argument("--n", type=int, default=200)
    ns = parse_line_with(p, argv)
    path = Path(ns.path);
    if not path.is_absolute(): path = state.cwd / path
    _preview(state, path, "head", None, ns.n, None)
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"preview": f"head --n {ns.n} {path}"})

def cmd_tail(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="tail")
    p.add_argument("path"); p.add_argument("--n", type=int, default=200)
    ns = parse_line_with(p, argv)
    path = Path(ns.path);
    if not path.is_absolute(): path = state.cwd / path
    _preview(state, path, "tail", None, None, ns.n)
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"preview": f"tail --n {ns.n} {path}"})

def cmd_search(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="search")
    p.add_argument("--query", required=True)
    p.add_argument("-R", action="store_true", dest="recursive")
    ns = parse_line_with(p, argv)
    state.search["query"] = ns.query
    state.search["recursive"] = bool(ns.recursive)
    state.search["hits"] = run_search(state.cwd, ns.query, ns.recursive)
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"search.query": ns.query, "search.recursive": ns.recursive, "search.matches": len(state.search["hits"])})

def cmd_reset_search(state: State, argv: List[str]) -> str:
    _ = argparse.ArgumentParser(prog="reset_search").parse_args(argv)
    state.search = {"query": None, "recursive": True, "hits": []}
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"search": "cleared"})

def cmd_add_file_to_buffer(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="add-file-to-buffer")
    p.add_argument("path")
    p.add_argument("--range", dest="range_spec")
    p.add_argument("--head", type=int)
    p.add_argument("--tail", type=int)
    ns = parse_line_with(p, argv)
    path = Path(ns.path);
    if not path.is_absolute(): path = state.cwd / path
    mode = "cat"
    if ns.range_spec: mode="cat"
    elif ns.head: mode="head"
    elif ns.tail: mode="tail"
    content = _read_slice(path, mode, ns.range_spec, ns.head, ns.tail)
    state.buffer["items"].append({"type":"file","path":str(path),"slice": ns.range_spec or (f"head:{ns.head}" if ns.head else (f"tail:{ns.tail}" if ns.tail else "")),"text":content})
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"buffer.added": str(path)})

def cmd_add_note_to_buffer(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="add-note-to-buffer")
    p.add_argument("text")
    ns = parse_line_with(p, argv)
    state.buffer["items"].append({"type":"note","text": ns.text})
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"buffer.added_note": len(ns.text)})

def cmd_show_buffer(state: State, argv: List[str]) -> str:
    _ = argparse.ArgumentParser(prog="show_buffer").parse_args(argv)
    text=[]
    for i,it in enumerate(state.buffer["items"], start=1):
        if it["type"]=="file":
            text.append(f"[{i}] FILE {it['path']} {it.get('slice','')}")
            text.append(it["text"])
        else:
            text.append(f"[{i}] NOTE"); text.append(it["text"])
    state.last_preview = {"path":"(buffer)","mode":"show_buffer","range":"","bytes": sum(len(t) for t in text),"content":"\n".join(text)}
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"preview": "buffer"})

def cmd_clear_buffer(state: State, argv: List[str]) -> str:
    _ = argparse.ArgumentParser(prog="clear_buffer").parse_args(argv)
    state.buffer["items"].clear()
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"buffer": "cleared"})

def cmd_compose_answer(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="compose_answer")
    p.add_argument("--max-chars", type=int, default=8000)
    ns = parse_line_with(p, argv)
    parts=[]
    for i,it in enumerate(state.buffer["items"], start=1):
        if it["type"]=="file":
            parts.append(f"### Source {i}: FILE {it['path']} {it.get('slice','')}")
            parts.append(it["text"])
        else:
            parts.append(f"### Note {i}"); parts.append(it["text"])
    ans = "\n".join(parts)
    if len(ans)>ns.max_chars: ans = ans[:ns.max_chars]
    state.answer_preview = ans
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"answer.chars": len(ans)})

def cmd_ls_long(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="ls")
    p.add_argument("-l", action="store_true")
    ns = parse_line_with(p, argv)
    rows = recompute_rows(state)
    # properties table for current window
    start = state.visible["offset"]; end = min(start + state.view_length, len(rows))
    lines = []
    #lines.append("props: CSV")
    headers = "NAME;TYPE;EXT;SIZE_B;MTIME_ISO;MODE;OWNER;GROUP;IS_ASCII;IS_HIDDEN;PATH"
    for r in rows[start:end]:
        pth = Path(r["path"])
        try:
            st = pth.stat(); mode = oct(st.st_mode & 0o777); owner = st.st_uid; group = st.st_gid
        except Exception:
            mode, owner, group = "-", "-", "-"
        tmp = f"{r['name']};{r['type']};{r['ext']};{r['size']};{r['mtime_iso']};{mode};{owner};{group};{'true' if r['ascii'] else 'false'};{'true' if r['hidden'] else 'false'};{r['path']}"
        lines.append(tmp.split(";"))
    #lines.append(render_compact(state, rows))

    csv_text = format_csv(headers.split(";"), lines, align=state.align_columns)

    return csv_text

def cmd_scroll(state: State, argv: List[str]) -> str:
    p = argparse.ArgumentParser(prog="scroll")
    p.add_argument("direction", choices=["up","down"])
    ns = parse_line_with(p, argv)
    if ns.direction=="up":
        state.visible["offset"] = max(0, state.visible["offset"] - state.view_length)
    else:
        state.visible["offset"] = state.visible["offset"] + state.view_length
    rows = recompute_rows(state)
    return finalize_output(state, rows, {"visible.offset": state.visible["offset"]})

def cmd_help(_: State, __: List[str]) -> str:
    return """Commands:
  cd / | cd .. | cd PATH
  filter set [--from DATE] [--to DATE] [--ext EXT ...] [--size-min N] [--size-max N] [--ascii] [--no-hidden]
  filter clear
  sort --key KEY[:a|:d] [--key ...]
  set-preview-tokens N
  set-view-length N
  set-state-full-every N
  show_state
  scroll up | scroll down
  cat PATH
  head PATH [--n 200]
  tail PATH [--n 200]
  search --query "text" [-R]
  reset_search
  add-file-to-buffer PATH [--range "lines:A-B" | --head N | --tail N]
  add-note-to-buffer "text"
  show_buffer
  clear_buffer
  compose_answer [--max-chars 8000]
  ls -l
  help
  man
  quit | exit
"""

def cmd_man(_: State, __: List[str]) -> str:
    return """MAN (memview)
Design:
  - Deterministic, LLM-parsable output.
  - Argparse-driven subcommands & flags.
  - Dates via dateparser/dateutil/ISO; Sizes like 10kb/5mb.

Output (per command):
  - 'changed:' block when state mutated (only changed keys).
  - periodic 'state:' full dump (controlled by set-state-full-every).
  - compact CSV 'view' block always.
  - optional 'search_hits:', 'preview:', 'answer:'.

Filter fields:
  --from/--to → mtime window
  --ext       → repeatable; e.g., --ext .txt --ext .md
  --size-min/--size-max
  --ascii     → only ascii/utf-8 files
  --no-hidden → exclude dotfiles/dirs

Sort keys: name, ext, size, mtime with :a or :d (asc/desc).
"""

# -------- dispatcher / REPL --------

COMMANDS = {
    "cd": cmd_cd,
    "filter": None,     # group: filter set|clear
    "sort": cmd_sort,
    "set-preview-tokens": cmd_set_preview_tokens,
    "set-view-length": cmd_set_view_length,
    "set-state-full-every": cmd_set_state_full_every,
    "show_state": cmd_show_state,
    "scroll": cmd_scroll,
    "cat": cmd_cat,
    "head": cmd_head,
    "tail": cmd_tail,
    "search": cmd_search,
    "reset_search": cmd_reset_search,
    "add-file-to-buffer": cmd_add_file_to_buffer,
    "add-note-to-buffer": cmd_add_note_to_buffer,
    "show_buffer": cmd_show_buffer,
    "clear_buffer": cmd_clear_buffer,
    "compose_answer": cmd_compose_answer,
    "ls": cmd_ls_long,
    "help": cmd_help,
    "man": cmd_man,
}

def dispatch_line(state: State, line: str) -> str:
    """
    Parse one input line, run the command, and return the output string.
    """
    if not line.strip():
        return ""
    parts = shlex.split(line)
    cmd = parts[0]
    args = parts[1:]
    if cmd in ("quit","exit"):
        return "__EXIT__"
    if cmd == "filter":
        # subcommands: set / clear
        if not args:
            return "usage: filter set|clear"
        sub, sub_args = args[0], args[1:]
        if sub == "set":
            return cmd_filter_set(state, sub_args)
        elif sub == "clear":
            return cmd_filter_clear(state, sub_args)
        else:
            return "usage: filter set|clear"
    fn = COMMANDS.get(cmd)
    if not fn:
        return f"unknown command: {cmd}\n" + cmd_help(state, [])
    try:
        return fn(state, args)
    except SystemExit as e:
        # argparse error path prints to stderr; capture minimal notice
        return f"(arg error) {cmd}: {e}\nUse 'help' for usage."
    except Exception as e:
        # keep hackable; no deep error handling
        return f"(error) {cmd}: {e}"

def main():
    """Run as REPL; prints returned strings."""
    st = State()
    # initial view
    rows = recompute_rows(st)
    print(render_full_state(st))
    print(render_compact(st, rows))
    while True:
        try:
            s = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        out = dispatch_line(st, s)
        if out == "__EXIT__":
            break
        if out:
            print(out)
    print("bye.")

if __name__ == "__main__":
    main()
