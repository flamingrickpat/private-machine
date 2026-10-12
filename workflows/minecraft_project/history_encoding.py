"""Gameplay history for model inputs, in the compact v4 columnar encoding.

The specialist prompt in ``workflows/minecraft/specialist_session.py`` describes
the same format. Change both together.
"""
from __future__ import annotations

import json


from workflows.minecraft_body.journal import read_actor_journal
from workflows.minecraft_body.call_outcomes import GameplayEvidence, signature
from workflows.minecraft_project.tree_store import TreeStore, digest


def gameplay_history(store: TreeStore) -> dict:
    """Rebuild cumulative outcomes from every retained actor journal.

    Failed attempts remain useful evidence. A specification revision cannot
    erase inventory deltas, searched cells, paused methods, or active recovery.
    """
    evidence = GameplayEvidence()
    for attempt in store.state["attempts"]:
        relative = attempt.get("journal", attempt["path"] + "/scratch/actor-journal.jsonl")
        for row in read_actor_journal(store.path_for(relative)):
            evidence.consume(row, relative)
    if store.state.get("planning_observation"):
        path = store.path_for(store.state["planning_observation"])
        observation = json.loads(path.read_text(encoding="utf-8"))
        # Planning can discover danger. It cannot supply an actor's geometry
        # checks or grant an escape completion certificate.
        evidence.consume({"event": "result", "seq": 0, "tool": "minecraft_observe", "arguments": {},
                          "time": path.stat().st_mtime, "ok": True, "result": observation}, str(path))
    return scoped_gameplay(evidence.snapshot())


def scoped_gameplay(history: dict) -> dict:
    """Share repeated method preconditions without discarding cumulative facts.

    The actor projection and original journals retain their full structure.
    Planning keeps every method, search, quantity, and original reference.
    Named columns describe method and outcome rows. Shared source paths replace
    repeated journal strings. Changed result operands remain explicit, so this
    representation preserves quantities, identities, and paused counters.
    A recent result can reference its method's last result plus `changes`.
    Apply those exact field replacements to recover the original sequence,
    time, quantities, and source. Missing fields keep the full older record.
    """
    states, indexes, methods = [], {}, []
    for method in history["methods"]:
        state = {**method["state"],
                 "inventory": [{key: value for key, value in item.items() if key != "display_name" and value is not None}
                               for item in method["state"]["inventory"]]}
        if state["held_item"] is not None:
            state["held_item"] = {key: value for key, value in state["held_item"].items()
                                  if key != "display_name" and value is not None}
        key = digest(state)
        if key not in indexes:
            indexes[key] = len(states)
            states.append(state)
        methods.append({"tool": method["tool"], "arguments": method["arguments"],
                        "ineffective": method["ineffective"], "state_index": indexes[key],
                        "last": {key: value for key, value in method["last"].items()
                                 if key not in {"tool", "arguments"} or value != method[key]}})
    sources, source_indexes = [], {}

    def share_sources(value):
        if isinstance(value, list):
            return [share_sources(row) for row in value]
        if not isinstance(value, dict):
            return value
        result = {key: share_sources(row) for key, row in value.items()}
        # Only journal provenance has both source and seq. A Minecraft tool's
        # source operand remains an ordinary callable argument.
        if isinstance(value.get("source"), str) and "seq" in value:
            source = result.pop("source")
            if source not in source_indexes:
                source_indexes[source] = len(sources)
                sources.append(source)
            result["source_index"] = source_indexes[source]
        return result

    kinds, kind_indexes, tool_names, tool_indexes = [], {}, [], {}
    outcome_fields = ["kind_index", "unsuccessful", "reason", "quantities", "inventory_delta", "made_progress",
                      "completion_verified", "source_index", "seq", "time"]
    rows = []
    for method in methods:
        tool = method["tool"]
        if tool not in tool_indexes:
            tool_indexes[tool] = len(tool_names)
            tool_names.append(tool)
        last = share_sources(method["last"])
        kind = last.pop("kind")
        if kind not in kind_indexes:
            kind_indexes[kind] = len(kinds)
            kinds.append(kind)
        last["kind_index"] = kind_indexes[kind]
        outcome = [last[key] for key in outcome_fields]
        outcome.append({key: value for key, value in last.items() if key not in outcome_fields})
        rows.append([tool_indexes[tool], method["arguments"], method["ineffective"], method["state_index"], outcome])
    # Repeated no-ops differ mainly in sequence and time. Reference the
    # method's last result and state every changed field explicitly. This
    # preserves each call's provenance without replaying identical failures.
    method_indexes = {signature(method["tool"], method["arguments"]): index
                      for index, method in enumerate(history["methods"])}
    recent = []
    for outcome in history["recent_outcomes"]:
        index = method_indexes.get(signature(outcome["tool"], outcome["arguments"]))
        if index is None:
            recent.append(outcome)
            continue
        last = history["methods"][index]["last"]
        changes = {key: value for key, value in outcome.items() if key not in last or value != last[key]}
        # A missing old field cannot be represented by an added null value.
        # Keep that uncommon older shape explicit instead of changing it.
        reference = {"method_index": index, **({"changes": changes} if changes else {})}
        recent.append(reference if last.keys() <= outcome.keys()
                      and len(json.dumps(reference)) < len(json.dumps(outcome)) else outcome)
    remaining = share_sources({**{key: value for key, value in history.items() if key != "methods"},
                               "recent_outcomes": recent})
    search_defaults = ({key: remaining["searches"][0][key] for key in ("world", "method", "uncertainty")}
                       if remaining["searches"] else {})
    remaining["searches"] = [{key: value for key, value in search.items()
                             if key not in search_defaults or value != search_defaults[key]}
                            for search in remaining["searches"]]
    # Tools and stock recur across different body positions. Share complete
    # item facts once; a changed count, slot, or durability gets its own row.
    items, item_indexes = [], {}
    def item_index(item):
        key = digest(item)
        if key not in item_indexes:
            item_indexes[key] = len(items)
            items.append(item)
        return item_indexes[key]
    shared_states = [{**{key: value for key, value in state.items() if key not in {"inventory", "held_item"}},
                      "inventory_indexes": [item_index(item) for item in state["inventory"]],
                      "held_item_index": item_index(state["held_item"]) if state["held_item"] is not None else None}
                     for state in states]
    return {**remaining, "schema": "pm.gameplay-planning.v4", "methods": rows, "method_states": shared_states,
            "inventory_items": items,
            "method_fields": ["tool_index", "arguments", "ineffective", "state_index", "last"], "tool_names": tool_names,
            "outcome_fields": [*outcome_fields, "additional_fields"], "source_paths": sources, "outcome_kinds": kinds,
            "search_defaults": search_defaults}

