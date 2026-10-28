"""Format serialized runtime records without runtime imports or inference."""
from __future__ import annotations

from pm.records.copying import pickle_copy

from collections import defaultdict

def record_links(record: dict) -> list[dict]:
    """Return typed links from explicit record fields and consumed versions.

    These links describe stored dependencies. They do not prove that a model's
    interpretation was correct or measure the effect of an individual input.
    """
    links = []

    def add(relation: str, ids) -> None:
        for key in ids:
            if key is not None:
                link = {"relation": relation, "id": int(key)}
                if link not in links and int(key) != record["id"]:
                    links.append(link)

    fields = {
        "supersedes_id": "previous immutable record",
        "focus_knoxel_ids": "focus", "winning_attention_ids": "selected codelet output",
        "supporting_knoxel_ids": "support", "opposing_knoxel_ids": "counterevidence",
        "source_knoxel_ids": "source", "internal_knoxel_ids": "structure member",
    }
    for key, label in fields.items():
        value = record.get(key, [])
        add(label, value if isinstance(value, list) else [value])
    lineage = record.get("lineage") or {}
    add("parent", lineage.get("parent_record_ids", []))
    add("source event", lineage.get("source_event_ids", []))
    metadata = record.get("metadata", {})
    add("evidence", metadata.get("evidence_ids", []))
    add("source", metadata.get("source_ids", []))
    add("responding to", metadata.get("responding_to_ids", []))
    add("speech context source", metadata.get("generation_source_revisions", {}).keys())
    add("source revision", metadata.get("source_revisions", {}).keys())
    add("effective experiment version", metadata.get("reflection_versions", {}).values())
    add("experiment", metadata.get("reflection_versions", {}).keys())
    for key, label in (("revised_from", "previous belief"), ("reflection_experiment_id", "experiment"),
                       ("effective_record_id", "effective belief"), ("appraisal_id", "appraisal"),
                       ("action_schema_id", "selected public action"), ("public_reply_id", "public reply")):
        if metadata.get(key) is not None:
            add(label, [metadata[key]])
    for source in metadata.get("memory_evidence", []):
        add("public context memory", [source["knoxel_id"]])
    # A19F records: percepts name the records they are about and their
    # recommendation, expectations name their action and evidence, and an
    # action names the shard content that its reply read.
    add("about", metadata.get("about_ids", []))
    if metadata.get("recommendation_id") is not None:
        add("recommendation", [metadata["recommendation_id"]])
    add("evidence", record.get("evidence_ids", []))
    add("observation", record.get("observation_ids", []))
    if record.get("created_from_action_id") is not None:
        add("expected after action", [record["created_from_action_id"]])
    payload = metadata.get("dispatch_payload") or {}
    for shard in payload.get("shards", []):
        add(f"shard item read by the action ({shard['name']})",
            [item["record_id"] for item in shard["shard"].get("items", [])])
        add(f"shard tail read by the action ({shard['name']})", shard["shard"].get("tail_ids", []))
    add("stimulus of the action", [item["knoxel_id"] for item in payload.get("stimuli", [])])
    return links


def record_card(wrapped: dict) -> dict:
    """Copy exact record text and selected explanatory fields without vectors."""
    data = wrapped["data"]
    metadata = data.get("metadata", {})
    provenance = data.get("provenance") or {}
    kind = wrapped["__type__"].removesuffix("Knoxel")
    role = ("Private thought" if data.get("feature_type") in {"InnerMonologue", "Thought"} else
            "Subjective experience" if data.get("feature_type") == "SubjectiveExperience" else
            "Private appraisal" if metadata.get("origin") == "appraisal" or metadata.get("operation") == "appraisal" else
            "Codelet output" if kind == "Attention" else
            "Codelet recommendation" if kind == "CodeletPercept" and metadata.get("percept_type") == "recommendation" else
            "Codelet percept" if kind == "CodeletPercept" else
            "Structure" if kind == "Structure" else
            "Expectation" if kind == "ExpectationRecord" else
            "Reflection experiment" if metadata.get("schema") == "pm.reflection-experiment.v1" else
            "Decision and routing record" if metadata.get("operation") == "action_selection" else kind)
    details = {key: metadata[key] for key in (
        "trigger_reason", "evidence_summary", "interpretation", "alternatives", "continuation_reason",
        "disclosure", "reflection", "plan", "history", "selection", "why", "reason",
    ) if key in metadata}
    if data.get("frame_reason"):
        details["frame_reason"] = data["frame_reason"]
    return {"id": data["id"], "tick": data["tick_id"], "kind": kind, "role": role,
            "text": str(data.get("content", "")), "source": provenance.get("source_ref", ""),
            "operation": metadata.get("operation", ""), "codelet": data.get("from_codelet") or data.get("codelet_name", ""),
            "selected": data.get("selected"), "status": metadata.get("status", metadata.get("state", "")),
            "fixture": bool(metadata.get("gate_supplied")), "details": details,
            "route": metadata.get("workflow_name", ""), "links": record_links(data),
            "record": visible_data(data)}


def visible_data(value):
    """Preserve record fields while hiding embeddings from imported packages."""
    if isinstance(value, dict):
        return {key: visible_data(item) for key, item in value.items() if key not in {"embedding", "embeddings"}}
    if isinstance(value, list):
        return [visible_data(item) for item in value]
    return value


def _shard_name(key: dict, names: dict) -> str:
    """Return the reader's name of a shard key: self, the entity name, or the workflow name."""
    if key["kind"] == "self":
        return "self"
    if key["kind"] == "entity":
        return names.get(str(key["ref"]), f"entity {key['ref']}")
    return f"workflow {key['ref']}"


def _flat(value: dict, prefix: str = "") -> dict:
    """Flatten nested numeric fields to dotted names. Other values stay out."""
    flat = {}
    for key, item in value.items():
        if isinstance(item, dict):
            flat.update(_flat(item, f"{prefix}{key}."))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            flat[f"{prefix}{key}"] = item
    return flat


def state_change(before: dict | None, after: dict) -> list[dict]:
    """Return the mental-state axes that changed in one tick, the largest change first.

    ``before`` is the state of the previous tick, or None for the first tick
    of a capture. Both are serialized ``FullMentalState`` objects. The
    result is arithmetic on captured values, not a judgment.
    """
    old, new = _flat(before or {}), _flat(after)
    rows = [{"axis": axis, "before": old.get(axis), "after": value,
             "change": round(value - old[axis], 4) if axis in old else None}
            for axis, value in new.items() if axis in old and abs(value - old[axis]) > 1e-4]
    return sorted(rows, key=lambda row: -abs(row["change"]))


def _record_row(key: int, label: str = "", data: dict | None = None) -> dict:
    return {"kind": "record", "id": int(key), "label": label, "data": data or {}}


def _text_row(text: str, data=None) -> dict:
    return {"kind": "text", "text": text, "data": data if data is not None else {}}


def present_cycle(package: dict, cards: dict, after: dict | None = None) -> dict:
    """Arrange one ``cycle`` package as the path from the stimuli to the reply.

    Each section lists captured records (shown as full record cards) or
    captured values (shown as text with all fields on hover and expansion).
    The order follows the cycle: stimuli, feeling before and after,
    appraisal, codelets, CSM, structures and attention, shards, experience,
    decisions and routes, expectations, and the reply. Records that the
    package does not hold stay as disabled references in the desk.
    """
    cycle, turn = package["cycle"], package.get("gwt") or {}
    names = package.get("entity_names") or {}
    trace = turn.get("trace") or {}
    tick = cycle["tick_id"]
    created = [key for key in turn.get("knoxel_ids", []) if key in cards and cards[key]["tick"] == tick]
    sections = []

    def section(title: str, rows: list[dict], note: str = "") -> None:
        sections.append({"title": title, "note": note, "rows": rows})

    groups = cycle.get("stimulus_groups", [])
    section("Stimuli by group", [
        row for group in groups for row in [
            _text_row(f"Group {_shard_name(group['key'], names)} · triage {group.get('category') or '-'}"
                      f"{' · insignificant' if group.get('insignificant') else ''} · correlation {group['correlation_id']}",
                      group),
            *(_record_row(key, "stimulus") for key in group["stimulus_ids"])]])
    changes = state_change(package.get("mental_state_before"), cycle["mental_state"])
    section("Mental state before and after this tick", [
        _text_row(f"{row['axis']}: {row['before']} → {row['after']} ({row['change']:+})", row) for row in changes[:16]
    ] + [_text_row("All fields after this tick", cycle["mental_state"]),
         _text_row("All fields before this tick", package.get("mental_state_before") or {})],
        "The 16 largest changes against the previous tick. Hover or expand for every field.")
    section("Appraisal", [_record_row(key) for key in created if cards[key]["role"] == "Private appraisal"
                          or cards[key]["operation"] in {"appraisal", "main_appraisal"}])
    decoded = package.get("decoded_vectors") or {}
    codelet_rows = [_text_row(f"Selector patterns: {trace.get('selector_patterns') or '(none)'}", trace.get("codelet_ratings", {}))]
    codelet_rows += [_record_row(key, "earlier line that echoes the newest line") for key in trace.get("echo_ids", [])]
    codelet_rows += [_text_row(f"Selector rating {name}: {value}", {"codelet": name, "rating": value})
                     for name, value in sorted(trace.get("codelet_ratings", {}).items(), key=lambda pair: -pair[1])]
    for run in trace.get("codelet_runs", []):
        codelet_rows.append(_text_row(f"Run {run['codelet']} · wave {run.get('wave')} · {run.get('mode')} · "
                                      f"{run.get('status')} · score {run.get('score')}", run))
        for key in trace.get("percept_ids", []):
            card = cards.get(key)
            if card is not None and card["codelet"] == run["codelet"]:
                codelet_rows.append(_record_row(key, "causal" if card["record"].get("causal") else "not causal",
                                                decoded.get(str(key), {})))
    for boost in trace.get("pathway_boosts", []):
        codelet_rows.append(_text_row(f"Pathway {boost['pathway']}: {boost['source']} → {boost['target']} "
                                      f"(strength {boost['strength']})", boost))
    section("Codelets: selector, runs, variants, percepts, mini-appraisals", codelet_rows,
            "A percept card shows its mini-appraisal (decoded vectors) on hover. Causal means it entered a shard.")
    csm = sorted(cycle.get("csm", {}).get("items", {}).values(), key=lambda item: -item["activation"])
    section("Current situational model (CSM)", [
        _record_row(item["record_id"], f"activation {item['activation']:.2f} · base {item['base_activation']:.2f} · "
                                      f"endorsers {', '.join(item['endorsers']) or '-'}", item) for item in csm])
    structure_rows = []
    for key in trace.get("structure_ids", []):
        card = cards.get(key)
        structure_rows.append(_record_row(key, "structure"))
        if card is None:
            continue
        for rating in card["record"].get("metadata", {}).get("attention", []):
            structure_rows.append(_text_row(
                f"Attention for {_shard_name(rating['shard'], names)}: "
                f"rating {rating.get('rating')} · score {rating.get('score')}"
                f"{' · won' if rating.get('won') else ''} · {rating.get('reasoning', '')}", rating))
    section("Structures and attention ratings", structure_rows,
            "Members of each structure are its links. Only the winners of a shard bring new items.")
    shard_rows = []
    for change in turn.get("shard_changes", []):
        name = _shard_name(change["key"], names)
        shard_rows.append(_text_row(f"Shard {name}: {len(change['entered'])} entered, {len(change['left'])} left, "
                                    f"{len(change['kept_ids'])} kept, {len(change['tail_ids'])} in the tail", change))
        shard_rows += [_record_row(item["record_id"], f"{name} · entered · {item.get('required_by') or 'rated'} · "
                                                      f"{item['reason']}", item) for item in change["entered"]]
        shard_rows += [_record_row(item["record_id"], f"{name} · left · {item['reason']}", item) for item in change["left"]]
        shard_rows += [_record_row(key, f"{name} · kept") for key in change["kept_ids"]]
    section("Workspace shards: entered, left, kept", shard_rows)
    experiences = [key for key in created if cards[key]["role"] == "Subjective experience"]
    section("Subjective experience", [_record_row(key) for key in experiences])
    decision_rows = [_text_row(f"Decision for {_shard_name(item['group'], names)}: {item['text']}", item)
                     for item in cycle.get("decisions", [])]
    if turn.get("decision_id") is not None:
        decision_rows.append(_record_row(turn["decision_id"], "decision and routing record"))
    actions = [key for key in created if cards[key]["kind"] == "ActionSchema"]
    decision_rows += [_record_row(key, "routed action") for key in actions]
    decision_rows += [_text_row(f"Dispatch {ref['action_type']} → {ref['workflow']} for "
                                f"{_shard_name(ref['group'], names)}", ref) for ref in cycle.get("dispatches", [])]
    section("Decisions and routing", decision_rows)
    expectations = [key for key in created if cards[key]["role"] == "Expectation"]
    checked = [key for key, card in cards.items() if card["role"] == "Expectation"
               and card["record"].get("metadata", {}).get("last_check_tick") == tick]
    section("Expectations: recorded and checked", [_record_row(key, "recorded in this tick") for key in expectations]
            + [_record_row(key, "checked against this tick's stimuli") for key in checked])
    memory_rows = []
    for key in actions:
        payload = cards[key]["record"].get("metadata", {}).get("dispatch_payload") or {}
        for shard in payload.get("shards", []):
            memory_rows += [_record_row(item["record_id"], f"{shard['name']} · item read by the reply", item)
                            for item in shard["shard"].get("items", [])]
    section("Memory contents supplied to the decision and the reply", memory_rows,
            "The shard items that the action request carried to the dialogue workflow.")
    replies = [card["id"] for card in cards.values()
               if card["record"].get("metadata", {}).get("action_schema_id") in actions]
    section("Public reply", [_record_row(key, "delivered reply") for key in replies])
    return {"tick": tick, "title": f"Cycle {tick}: " + "; ".join(
        f"{_shard_name(group['key'], names)}" for group in groups), "summary": turn.get("summary", ""),
        "sections": sections}


def present_packages(item: dict) -> dict:
    """Format named runtime collections for the desk without changing evidence.

    Accept arbitrary collection members. Uncaptured references stay explicit.
    Conflicting copies of one record raise ValueError. Scores remain separate.
    """
    if "packages" not in item:
        return item
    records = {}
    for package in item["packages"]:
        if package["schema_id"] != "pm.record-package.v1":
            raise ValueError("Unsupported runtime record package.")
        if package["view"] not in {"records", "chat", "gwt", "cycle"}:
            raise ValueError("Unsupported record package display hint.")
        for wrapped in package["records"]:
            key = wrapped["data"]["id"]
            if key in records and records[key] != wrapped:
                raise ValueError(f"Conflicting captured versions of knoxel {key}.")
            records[key] = wrapped
        if any(key not in records for key in package["record_ids"]):
            raise ValueError("A package member has no serialized record.")
        if package["view"] == "chat" and any(
                records[key]["data"].get("feature_type") != "Dialogue" for key in package["record_ids"]):
            raise ValueError("A chat package contains a private record.")
    primary = next((package["record_ids"] for package in item["packages"] if package["view"] == "records"),
                   list(dict.fromkeys(key for package in item["packages"] for key in package["record_ids"])))
    cards = {key: record_card(value) for key, value in records.items()}
    for package in item["packages"]:
        for key, revision in package["revisions"].items():
            cards[int(key)]["revision"] = revision
    groups = defaultdict(list)
    for key in primary:
        groups[cards[key]["tick"]].append(cards[key])
    episodes = []
    for tick, rows in groups.items():
        thoughts = any(row["role"] == "Private thought" for row in rows)
        if not thoughts and not any(row["role"] == "Routing decision" for row in rows):
            continue
        # A19F turn records keep the codelet runs of their cycle. Older
        # captures kept them in the attention frame record.
        frames = [records[row["id"]]["data"] for row in rows if row["kind"] == "AttentionFrame"]
        schedule = [entry for frame in frames for entry in frame.get("metadata", {}).get("codelet_runs", [])]
        schedule += [entry for package in item["packages"] if (package.get("gwt") or {}).get("tick_id") == tick
                     for entry in (package["gwt"].get("trace") or {}).get("codelet_runs", [])]
        episodes.append({"tick": tick, "title": "Private reflection" if thoughts else "Attention and routing",
                         "records": [row["id"] for row in rows],
                         "codelets": [{key: entry[key] for key in ("codelet", "status", "reason")}
                                      for entry in schedule]})
    conversation = []
    for package in item["packages"]:
        if package["view"] != "chat":
            continue
        for key in package["record_ids"]:
            data = records[key]["data"]
            source = data.get("override_source") or (data.get("provenance") or {}).get("source_ref", "Unknown")
            conversation.append({"id": key, "speaker": source, "text": str(data["content"]),
                                 "fixture": cards[key]["fixture"]})
    cycles = [present_cycle(package, cards) for package in item["packages"] if package["view"] == "cycle"]
    for package in item["packages"]:
        for key, value in (package.get("decoded_vectors") or {}).items():
            cards[int(key)]["decoded"] = value
    result = {key: pickle_copy(value) for key, value in item.items() if key != "packages"}
    result.update(conversation=conversation, cycles=cycles,
                  cognition={"episodes": episodes, "primary_ids": primary,
                             "records": {str(key): value for key, value in cards.items()}},
                  package_views=[{key: value for key, value in package.items()
                                  if key not in {"records", "cycle", "decoded_vectors"}}
                                 for package in item["packages"]])
    return result
