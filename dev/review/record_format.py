"""Format serialized runtime records without runtime imports or inference."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

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
    return links


def record_card(wrapped: dict) -> dict:
    """Copy exact record text and selected explanatory fields without vectors."""
    data = wrapped["data"]
    metadata = data.get("metadata", {})
    provenance = data.get("provenance") or {}
    kind = wrapped["__type__"].removesuffix("Knoxel")
    role = ("Private thought" if data.get("feature_type") == "InnerMonologue" else
            "Private appraisal" if metadata.get("origin") == "appraisal" else
            "Codelet output" if kind == "Attention" else
            "Reflection experiment" if metadata.get("schema") == "pm.reflection-experiment.v1" else
            "Routing decision" if metadata.get("operation") == "action_selection" else kind)
    details = {key: metadata[key] for key in (
        "trigger_reason", "evidence_summary", "interpretation", "alternatives", "continuation_reason",
        "disclosure", "reflection", "plan", "history", "selection", "why", "reason",
    ) if key in metadata}
    if data.get("frame_reason"):
        details["frame_reason"] = data["frame_reason"]
    return {"id": data["id"], "tick": data["tick_id"], "kind": kind, "role": role,
            "text": str(data.get("content", "")), "source": provenance.get("source_ref", ""),
            "operation": metadata.get("operation", ""), "codelet": data.get("from_codelet", ""),
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
    result = {key: deepcopy(value) for key, value in item.items() if key != "packages"}
    result.update(conversation=conversation,
                  cognition={"episodes": episodes, "primary_ids": primary,
                             "records": {str(key): value for key, value in cards.items()}},
                  package_views=[{key: value for key, value in package.items() if key != "records"}
                                 for package in item["packages"]])
    return result
