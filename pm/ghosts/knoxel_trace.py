from __future__ import annotations

import json
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from pm.data_structures import KnoxelBase, KnoxelList


def _now_utc() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_excerpt(text: str, max_len: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _looks_like_ghost(obj: Any) -> bool:
    return hasattr(obj, "all_knoxels") and hasattr(obj, "current_tick_id")


def _find_ghost(args: tuple, kwargs: dict) -> Any:
    for obj in args:
        if _looks_like_ghost(obj):
            return obj
    for obj in kwargs.values():
        if _looks_like_ghost(obj):
            return obj
    return None


def _iter_knoxels(value: Any, seen: Optional[Set[int]] = None) -> Iterable[KnoxelBase]:
    seen = seen or set()
    if value is None:
        return
    if isinstance(value, KnoxelBase):
        if id(value) not in seen:
            seen.add(id(value))
            yield value
        return
    if isinstance(value, KnoxelList):
        for item in value.to_list():
            if isinstance(item, KnoxelBase) and id(item) not in seen:
                seen.add(id(item))
                yield item
        return
    if isinstance(value, (list, tuple, set)):
        for sub in value:
            yield from _iter_knoxels(sub, seen)
        return
    if isinstance(value, dict):
        for sub in value.values():
            yield from _iter_knoxels(sub, seen)


def _extract_knoxel_ids(value: Any, max_ids: int = 256) -> List[int]:
    ids: Set[int] = set()
    for k in _iter_knoxels(value):
        try:
            kid = int(getattr(k, "id", -1) or -1)
            if kid >= 0:
                ids.add(kid)
        except Exception:
            continue
        if len(ids) >= max_ids:
            break
    return sorted(ids)


def _collect_existing_ids(ghost: Any) -> Set[int]:
    out: Set[int] = set()
    for kid in (getattr(ghost, "all_knoxels", {}) or {}).keys():
        try:
            out.add(int(kid))
        except Exception:
            continue
    return out


def _fingerprint_knoxel(k: KnoxelBase) -> tuple:
    return (
        bool(getattr(k, "causal", False)),
        str(getattr(k, "content", "") or ""),
        float(getattr(k, "affective_valence", 0.0) or 0.0),
        float(getattr(k, "incentive_salience", 0.0) or 0.0),
        float(getattr(k, "interlocus", 0.0) or 0.0),
        int(getattr(k, "tick_id", -1) or -1),
        int(getattr(k, "sub_tick_id", 0) or 0),
    )


def _snapshot_fingerprints(ghost: Any, ids: Optional[Set[int]] = None) -> Dict[int, tuple]:
    out: Dict[int, tuple] = {}
    src = getattr(ghost, "all_knoxels", {}) or {}
    for kid in (ids or set(src.keys())):
        k = src.get(kid)
        if k is None:
            continue
        try:
            out[int(kid)] = _fingerprint_knoxel(k)
        except Exception:
            continue
    return out


def _snapshot_knoxel(ghost: Any, knoxel_id: int) -> Optional[Dict[str, Any]]:
    try:
        k = ghost.get_knoxel_by_id(knoxel_id)
    except Exception:
        k = None
    if k is None:
        return None

    return {
        "id": int(getattr(k, "id", -1) or -1),
        "type": k.__class__.__name__,
        "feature_type": str(getattr(k, "feature_type", "") or ""),
        "source": str(getattr(k, "source", "") or ""),
        "causal": bool(getattr(k, "causal", False)),
        "tick_id": int(getattr(k, "tick_id", -1) or -1),
        "sub_tick_id": int(getattr(k, "sub_tick_id", 0) or 0),
        "content_excerpt": _safe_excerpt(str(getattr(k, "content", "") or ""), 220),
        "valence": float(getattr(k, "affective_valence", 0.0) or 0.0),
        "salience": float(getattr(k, "incentive_salience", 0.0) or 0.0),
        "interlocus": float(getattr(k, "interlocus", 0.0) or 0.0),
        "metadata": _jsonable(getattr(k, "metadata", {}) or {}),
        "trace_tags": list(getattr(k, "trace_tags", []) or []),
    }


def _ensure_session(ghost: Any) -> Optional[Dict[str, Any]]:
    return getattr(ghost, "_knoxel_trace_session", None)


def set_knoxel_trace_phase(ghost: Any, phase_name: str) -> None:
    s = _ensure_session(ghost)
    if s is None:
        return
    s["current_phase"] = str(phase_name or "")


def start_knoxel_trace_session(ghost: Any, output_dir: Optional[str] = None) -> None:
    enabled = bool(getattr(ghost, "enable_knoxel_trace", True))
    if not enabled:
        return

    setattr(
        ghost,
        "_knoxel_trace_session",
        {
            "version": 2,
            "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
            "created_at": _now_utc(),
            "current_phase": "",
            "counter": 0,
            "calls": [],
            "knoxels": {},
            "events": [],
            "output_dir": output_dir or str(getattr(ghost, "knoxel_trace_output_dir", "data/knoxel_traces")),
            "active_call_id": None,
        },
    )


def _record_call(
    ghost: Any,
    call_name: str,
    module_name: str,
    phase_name: str,
    duration_ms: float,
    input_ids: List[int],
    output_ids: List[int],
    created_ids: List[int],
    error: str = "",
) -> None:
    s = _ensure_session(ghost)
    if s is None:
        return

    s["counter"] += 1
    call_id = int(s["counter"])

    for kid in sorted(set(input_ids + output_ids + created_ids)):
        snap = _snapshot_knoxel(ghost, kid)
        if snap is not None:
            s["knoxels"][kid] = snap

    s["calls"].append(
        {
            "call_id": call_id,
            "phase": str(phase_name or s.get("current_phase", "")),
            "name": call_name,
            "module": module_name,
            "duration_ms": round(float(duration_ms), 3),
            "input_knoxels": sorted(set(int(x) for x in input_ids)),
            "output_knoxels": sorted(set(int(x) for x in output_ids)),
            "created_knoxels": sorted(set(int(x) for x in created_ids)),
            "ok": not bool(error),
            "error": str(error or ""),
        }
    )


def _semantic_tags_for_call(call_name: str, phase_name: str) -> List[str]:
    tags = {
        f"phase:{str(phase_name or '').strip()}",
        f"fn:{str(call_name or '').split('.')[-1]}",
    }
    n = str(call_name or "").lower()
    p = str(phase_name or "").lower()
    if "workspace" in n or p == "workspace":
        tags.add("workspace_basis")
    if "state" in n or "mental_state" in n or p == "state":
        tags.add("mental_state_basis")
    if "csm" in n or p == "csm":
        tags.add("csm_basis")
    if "pam" in n or p == "pam":
        tags.add("pam_basis")
    if "attention" in n or p == "attention":
        tags.add("attention_basis")
    if "action" in n or p == "action":
        tags.add("action_basis")
    if "qualia" in n:
        tags.add("qualia_basis")
    if "simulation" in n or p == "simulation":
        tags.add("simulation_basis")
    if "arbitration" in n or p == "arbitration":
        tags.add("arbitration_basis")
    if "learn" in n or p.startswith("learn"):
        tags.add("learning_basis")
    if "ccq" in n or "finalize" in n:
        tags.add("ccq_basis")
    return sorted(x for x in tags if x)


def _tag_knoxels(ghost: Any, ids: List[int], tags: List[str]) -> None:
    if not tags:
        return
    for kid in ids:
        try:
            k = ghost.get_knoxel_by_id(int(kid))
        except Exception:
            k = None
        if k is None:
            continue
        try:
            for t in tags:
                if t:
                    k.add_trace_tag(t)
        except Exception:
            continue


def log_knoxel_addition(ghost: Any, knoxel: Any, generated_embedding: bool = False) -> None:
    """
    Global hook for BaseGhost.add_knoxel to record new entities/features in trace,
    including those produced outside wrapped phase functions.
    """
    s = _ensure_session(ghost)
    if s is None:
        return

    try:
        kid = int(getattr(knoxel, "id", -1) or -1)
    except Exception:
        kid = -1
    if kid < 0:
        return

    if kid not in s["knoxels"]:
        snap = _snapshot_knoxel(ghost, kid)
        if snap is not None:
            s["knoxels"][kid] = snap

    s["events"].append(
        {
            "t": _now_utc(),
            "type": "knoxel_add",
            "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
            "phase": str(s.get("current_phase", "") or ""),
            "active_call_id": s.get("active_call_id"),
            "knoxel_id": kid,
            "knoxel_type": knoxel.__class__.__name__,
            "feature_type": str(getattr(knoxel, "feature_type", "") or ""),
            "source": str(getattr(knoxel, "source", "") or ""),
            "causal": bool(getattr(knoxel, "causal", False)),
            "generated_embedding": bool(generated_embedding),
        }
    )


def trace_knoxel_flow(name: Optional[str] = None, phase: Optional[str] = None):
    """Decorator for tracing knoxel dataflow through a function."""

    def decorator(fn):
        if getattr(fn, "__knoxel_trace_wrapped__", False):
            return fn

        @wraps(fn)
        def wrapper(*args, **kwargs):
            ghost = _find_ghost(args, kwargs)
            if ghost is None:
                return fn(*args, **kwargs)

            s = _ensure_session(ghost)
            if s is None:
                return fn(*args, **kwargs)

            call_name = name or fn.__qualname__
            module_name = getattr(fn, "__module__", "")
            phase_name = phase or s.get("current_phase", "")

            in_args = args[1:] if args and _looks_like_ghost(args[0]) else args
            input_ids = sorted(set(_extract_knoxel_ids(in_args) + _extract_knoxel_ids(kwargs)))

            before_ids = _collect_existing_ids(ghost)
            before_fp = _snapshot_fingerprints(ghost, before_ids)
            tracked_read_ids: Set[int] = set()
            tracked_write_ids: Set[int] = set()

            original_get = getattr(ghost, "get_knoxel_by_id", None)
            original_add = getattr(ghost, "add_knoxel", None)
            csm_manager = getattr(ghost, "csm_manager", None)
            original_csm_items = getattr(csm_manager, "items", None) if csm_manager is not None else None
            original_csm_active = getattr(csm_manager, "get_active_items", None) if csm_manager is not None else None

            if callable(original_get):
                @wraps(original_get)
                def _wrapped_get_knoxel_by_id(kid, *a, **kw):
                    try:
                        tracked_read_ids.add(int(kid))
                    except Exception:
                        pass
                    return original_get(kid, *a, **kw)

                setattr(ghost, "get_knoxel_by_id", _wrapped_get_knoxel_by_id)

            if callable(original_add):
                @wraps(original_add)
                def _wrapped_add_knoxel(knoxel, *a, **kw):
                    res = original_add(knoxel, *a, **kw)
                    try:
                        kid = int(getattr(knoxel, "id", -1) or -1)
                    except Exception:
                        kid = -1
                    if kid < 0:
                        try:
                            kid = int(res)
                        except Exception:
                            kid = -1
                    if kid >= 0:
                        tracked_write_ids.add(kid)
                    return res

                setattr(ghost, "add_knoxel", _wrapped_add_knoxel)

            if callable(original_csm_items):
                @wraps(original_csm_items)
                def _wrapped_csm_items(*a, **kw):
                    res = original_csm_items(*a, **kw)
                    try:
                        for item in res:
                            kid = int(getattr(item, "knoxel_id", -1) or -1)
                            if kid >= 0:
                                tracked_read_ids.add(kid)
                    except Exception:
                        pass
                    return res

                setattr(csm_manager, "items", _wrapped_csm_items)

            if callable(original_csm_active):
                @wraps(original_csm_active)
                def _wrapped_csm_active(*a, **kw):
                    res = original_csm_active(*a, **kw)
                    try:
                        for item in res:
                            kid = int(getattr(item, "knoxel_id", -1) or -1)
                            if kid >= 0:
                                tracked_read_ids.add(kid)
                    except Exception:
                        pass
                    return res

                setattr(csm_manager, "get_active_items", _wrapped_csm_active)

            # reserve call id early so add events can bind to active call
            predicted_call_id = int(s.get("counter", 0)) + 1
            previous_active = s.get("active_call_id")
            s["active_call_id"] = predicted_call_id

            t0 = time.perf_counter()
            result = None
            err = ""
            try:
                result = fn(*args, **kwargs)
            except Exception as ex:
                err = f"{ex.__class__.__name__}: {ex}"
                raise
            finally:
                s["active_call_id"] = previous_active
                if callable(original_get):
                    setattr(ghost, "get_knoxel_by_id", original_get)
                if callable(original_add):
                    setattr(ghost, "add_knoxel", original_add)
                if callable(original_csm_items) and csm_manager is not None:
                    setattr(csm_manager, "items", original_csm_items)
                if callable(original_csm_active) and csm_manager is not None:
                    setattr(csm_manager, "get_active_items", original_csm_active)

                after_ids = _collect_existing_ids(ghost)
                after_fp = _snapshot_fingerprints(ghost, after_ids)
                created_ids = sorted(after_ids - before_ids)

                mutated_ids: List[int] = []
                for kid in sorted(after_ids & before_ids):
                    if before_fp.get(kid) != after_fp.get(kid):
                        mutated_ids.append(kid)

                out_ids = _extract_knoxel_ids(result)
                input_ids = sorted(set(input_ids + list(tracked_read_ids)))
                output_ids = sorted(set(out_ids + created_ids + mutated_ids + list(tracked_write_ids)))
                dt = (time.perf_counter() - t0) * 1000.0

                tags = _semantic_tags_for_call(call_name=call_name, phase_name=phase_name)
                _tag_knoxels(ghost, input_ids, tags)
                _tag_knoxels(ghost, output_ids, tags)

                _record_call(
                    ghost=ghost,
                    call_name=call_name,
                    module_name=module_name,
                    phase_name=phase_name,
                    duration_ms=dt,
                    input_ids=input_ids,
                    output_ids=output_ids,
                    created_ids=created_ids,
                    error=err,
                )

            return result

        setattr(wrapper, "__knoxel_trace_wrapped__", True)
        return wrapper

    return decorator


def trace_substep(ghost: Any, name: str, phase: str, fn, *args, **kwargs):
    """
    Convenience wrapper for sub-procedure instrumentation.
    Expects `fn` to accept `ghost` as first arg.
    """
    wrapped = trace_knoxel_flow(name=name, phase=phase)(fn)
    return wrapped(ghost, *args, **kwargs)


def _build_trace_payload(ghost: Any, ccq: Any = None) -> Dict[str, Any]:
    s = _ensure_session(ghost)
    if s is None:
        return {}

    ccq_ids: List[int] = []
    if ccq is not None and getattr(ccq, "knoxels", None) is not None:
        try:
            ccq_ids = [int(k.id) for k in ccq.knoxels.to_list()]
        except Exception:
            ccq_ids = []

    return {
        "version": 2,
        "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
        "created_at": _now_utc(),
        "workspace_gain": float(getattr(ghost, "workspace_gain", 0.0) or 0.0),
        "ccq_knoxel_ids": ccq_ids,
        "calls": list(s.get("calls", [])),
        "events": list(s.get("events", [])),
        "knoxels": dict(s.get("knoxels", {})),
    }


def _to_cytoscape(trace: Dict[str, Any]) -> Dict[str, Any]:
    calls = list(trace.get("calls", []))
    knoxel_map = dict(trace.get("knoxels", {}))
    events = list(trace.get("events", []))

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_node_ids: Set[str] = set()
    edge_counter = 0

    def add_node(node_id: str, data: Dict[str, Any], classes: str) -> None:
        if node_id in seen_node_ids:
            return
        seen_node_ids.add(node_id)
        nodes.append({"data": {"id": node_id, **data}, "classes": classes})

    def add_edge(source: str, target: str, relation: str, extra: Optional[Dict[str, Any]] = None) -> None:
        nonlocal edge_counter
        edge_counter += 1
        payload = {
            "id": f"e:{edge_counter}",
            "source": source,
            "target": target,
            "relation": relation,
        }
        if extra:
            payload.update(extra)
        edges.append({"data": payload, "classes": relation})

    for c in calls:
        call_id = int(c.get("call_id", 0) or 0)
        node_id = f"call:{call_id}"
        add_node(
            node_id,
            {
                "kind": "call",
                "call_id": call_id,
                "name": str(c.get("name", "") or ""),
                "phase": str(c.get("phase", "") or ""),
                "module": str(c.get("module", "") or ""),
                "duration_ms": float(c.get("duration_ms", 0.0) or 0.0),
                "ok": bool(c.get("ok", False)),
                "error": str(c.get("error", "") or ""),
            },
            "call",
        )
    # Explicit execution order chain.
    sorted_calls = sorted(calls, key=lambda x: int(x.get("call_id", 0) or 0))
    for i in range(1, len(sorted_calls)):
        prev_id = int(sorted_calls[i - 1].get("call_id", 0) or 0)
        cur_id = int(sorted_calls[i].get("call_id", 0) or 0)
        add_edge(f"call:{prev_id}", f"call:{cur_id}", "exec_next")

    for k, info in knoxel_map.items():
        kid = int(k)
        node_id = f"knoxel:{kid}"
        add_node(
            node_id,
            {
                "kind": "knoxel",
                "knoxel_id": kid,
                "type": str(info.get("type", "") or ""),
                "feature_type": str(info.get("feature_type", "") or ""),
                "source_name": str(info.get("source", "") or ""),
                "causal": bool(info.get("causal", False)),
                "tick_id": int(info.get("tick_id", -1) or -1),
                "content_excerpt": str(info.get("content_excerpt", "") or ""),
                "valence": float(info.get("valence", 0.0) or 0.0),
                "salience": float(info.get("salience", 0.0) or 0.0),
                "interlocus": float(info.get("interlocus", 0.0) or 0.0),
                "metadata": _jsonable(info.get("metadata", {})),
                "trace_tags": list(info.get("trace_tags", []) or []),
            },
            "knoxel causal" if bool(info.get("causal", False)) else "knoxel noncausal",
        )

    for c in calls:
        call_node = f"call:{int(c.get('call_id', 0) or 0)}"
        input_ids = list(c.get("input_knoxels", []) or [])
        if len(input_ids) > 1:
            proxy_node = f"proxy:input:{int(c.get('call_id', 0) or 0)}"
            add_node(
                proxy_node,
                {
                    "kind": "proxy_input",
                    "call_id": int(c.get("call_id", 0) or 0),
                    "phase": str(c.get("phase", "") or ""),
                    "name": str(c.get("name", "") or ""),
                    "input_count": len(input_ids),
                },
                "proxy input",
            )
            add_edge(proxy_node, call_node, "input_proxy")
            for kid in input_ids:
                knx_node = f"knoxel:{int(kid)}"
                if knx_node in seen_node_ids:
                    add_edge(knx_node, proxy_node, "input")
        else:
            for kid in input_ids:
                knx_node = f"knoxel:{int(kid)}"
                if knx_node in seen_node_ids:
                    add_edge(knx_node, call_node, "input")

        for kid in (c.get("output_knoxels", []) or []):
            knx_node = f"knoxel:{int(kid)}"
            if knx_node in seen_node_ids:
                add_edge(call_node, knx_node, "output")
        for kid in (c.get("created_knoxels", []) or []):
            knx_node = f"knoxel:{int(kid)}"
            if knx_node in seen_node_ids:
                add_edge(call_node, knx_node, "created")

    # Derivation links from knoxel metadata (e.g., memory -> recall feature).
    for k, info in knoxel_map.items():
        kid = int(k)
        dst = f"knoxel:{kid}"
        if dst not in seen_node_ids:
            continue
        meta = info.get("metadata", {}) or {}
        if not isinstance(meta, dict):
            continue

        src_mem = meta.get("source_memory_id")
        try:
            src_id = int(src_mem)
            src = f"knoxel:{src_id}"
            if src in seen_node_ids:
                add_edge(src, dst, "derived_from_memory")
        except Exception:
            pass

        src_feats = meta.get("source_feature_ids")
        if isinstance(src_feats, list):
            for sid in src_feats:
                try:
                    src_id = int(sid)
                except Exception:
                    continue
                src = f"knoxel:{src_id}"
                if src in seen_node_ids:
                    add_edge(src, dst, "derived_from_feature")

    # Explicit add events make it easy to show/hide creation waves by phase.
    for idx, ev in enumerate(events, start=1):
        if ev.get("type") != "knoxel_add":
            continue
        kid = int(ev.get("knoxel_id", -1) or -1)
        if kid < 0:
            continue
        target = f"knoxel:{kid}"
        if target not in seen_node_ids:
            continue

        source_call = ev.get("active_call_id")
        if source_call is not None:
            src = f"call:{int(source_call)}"
            if src in seen_node_ids:
                add_edge(src, target, "add_event", extra={"event_idx": idx, "phase": str(ev.get("phase", "") or "")})
                continue

        phase = str(ev.get("phase", "") or "unknown")
        phase_node = f"phase:{phase}"
        add_node(phase_node, {"kind": "phase", "phase": phase}, "phase")
        add_edge(phase_node, target, "add_event", extra={"event_idx": idx, "phase": phase})

    # Tag grouping nodes for quick basis analysis (workspace/mental_state/ccq/...).
    tag_to_knoxels: Dict[str, Set[int]] = {}
    for k, info in knoxel_map.items():
        tags = list(info.get("trace_tags", []) or [])
        for t in tags:
            if not isinstance(t, str):
                continue
            if t.endswith("_basis"):
                tag_to_knoxels.setdefault(t, set()).add(int(k))
    for tag, kids in sorted(tag_to_knoxels.items()):
        tag_node = f"tag:{tag}"
        add_node(tag_node, {"kind": "tag_group", "tag": tag, "count": len(kids)}, "tag")
        for kid in sorted(kids):
            knx_node = f"knoxel:{kid}"
            if knx_node in seen_node_ids:
                add_edge(tag_node, knx_node, "tag_group", extra={"tag": tag})

    return {
        "format": "cytoscape",
        "meta": {
            "tick": int(trace.get("tick", 0) or 0),
            "created_at": str(trace.get("created_at", "") or ""),
            "workspace_gain": float(trace.get("workspace_gain", 0.0) or 0.0),
            "ccq_knoxel_ids": list(trace.get("ccq_knoxel_ids", []) or []),
            "call_count": len(calls),
            "knoxel_count": len(knoxel_map),
            "event_count": len(events),
        },
        "elements": {
            "nodes": nodes,
            "edges": edges,
        },
        "trace": trace,
    }


def finalize_knoxel_trace_session(ghost: Any, ccq: Any = None) -> Optional[Dict[str, str]]:
    s = _ensure_session(ghost)
    if s is None:
        return None

    trace = _build_trace_payload(ghost, ccq=ccq)
    if not trace:
        return None

    out_dir = Path(s.get("output_dir", "data/knoxel_traces"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tick = int(trace.get("tick", 0) or 0)

    raw_path = out_dir / f"tick_{tick}_trace.json"
    cytoscape_path = out_dir / f"tick_{tick}_cytoscape.json"

    raw_path.write_text(json.dumps(trace, indent=2, ensure_ascii=True), encoding="utf-8")

    cy = _to_cytoscape(trace)
    cytoscape_path.write_text(json.dumps(cy, indent=2, ensure_ascii=True), encoding="utf-8")

    ghost.last_knoxel_trace = trace
    ghost.last_knoxel_trace_json = str(raw_path)
    ghost.last_knoxel_trace_cytoscape_json = str(cytoscape_path)
    # compatibility key kept for callers still printing html field
    ghost.last_knoxel_trace_html = ""

    if not hasattr(ghost, "knoxel_trace_history"):
        ghost.knoxel_trace_history = []
    ghost.knoxel_trace_history.append(
        {
            "tick": tick,
            "json_path": str(raw_path),
            "cytoscape_json_path": str(cytoscape_path),
            "calls": len(trace.get("calls", [])),
            "knoxels": len(trace.get("knoxels", {})),
            "events": len(trace.get("events", [])),
        }
    )
    if len(ghost.knoxel_trace_history) > 100:
        ghost.knoxel_trace_history = ghost.knoxel_trace_history[-100:]

    setattr(ghost, "_knoxel_trace_session", None)

    return {
        "json": str(raw_path),
        "cytoscape_json": str(cytoscape_path),
        "html": "",
    }
