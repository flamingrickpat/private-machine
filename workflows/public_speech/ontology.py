"""Durable private ontology: stable words for implemented inner state.

The ontology is state, not a gate. A term records one word the companion may
use for its inner life. A binding links that word to the accepted state that
supports its current use. The public review reads the terms and bindings and
reports an unsupported label. An unsupported label never blocks speech and
never creates a feeling.

Terms are seeded only from mechanisms that exist in the runtime:

- One ``mood_word`` per implemented emotion in ``StateEmotions``.
- One ``relationship_label`` per implemented axis in ``StateRelationship``.
- A small fixed set of ``mental_move`` words for actions the architecture
  performs, such as deciding, withholding, deceiving, reflecting, and planning.

A binding is created for an emotion that is notably present in the latest
accepted mental state. The binding names the internal-state record and the
causal evidence ids that support it. A previous binding for the same term is
made inactive in the same commit, so the active binding set stays small.

The record classes already exist in ``pm.records.cognitive_records``. This
module only writes them. It adds no second ontology store.
"""
from __future__ import annotations

from pm.records.cognitive_records import OntologyBindingRecord, OntologyTermRecord
from pm.records.mental_state import FullMentalState, StateEmotions, StateRelationship
from pm.records.base import Provenance
from pm.records.enums import SourceKind
from pm.bus.envelopes import new_id

#: An emotion at or above this intensity gets an active binding.
BIND_THRESHOLD = 0.35

#: The most bindings one observation may create. Keeps the active set small.
MAX_BINDINGS = 12

#: Mental moves the architecture implements. Each one is a real runtime action.
MENTAL_MOVES: list[tuple[str, str]] = [
    ("decide", "Choose one action among the current candidates."),
    ("withhold", "Keep a private thought out of public speech on purpose."),
    ("deceive", "State something false on purpose with retained facts."),
    ("reflect", "Turn attention inward on past experience."),
    ("plan", "Choose future steps toward one goal."),
    ("speculate", "Imagine a possibility that did not happen."),
]


def _emotion_terms() -> list[dict]:
    """Return one ``mood_word`` seed per implemented emotion."""
    terms = []
    for name, field in StateEmotions.model_fields.items():
        terms.append({
            "term": name,
            "term_kind": "mood_word",
            "definition": str(field.description or name),
            "usage_rule": "Use only when the accepted state shows this feeling. "
                          "A word without state support is an unsupported label.",
            "stability": 1.0,
            "content": f"mood word {name}",
        })
    return terms


def _relationship_terms() -> list[dict]:
    """Return one ``relationship_label`` seed per implemented relationship axis."""
    terms = []
    for name, field in StateRelationship.model_fields.items():
        if name == "entity_id":
            continue
        terms.append({
            "term": name,
            "term_kind": "relationship_label",
            "definition": str(field.description or name),
            "usage_rule": "Use only when the accepted relationship state supports this label.",
            "stability": 1.0,
            "content": f"relationship label {name}",
        })
    return terms


def _mental_move_terms() -> list[dict]:
    """Return the fixed ``mental_move`` seeds."""
    return [{
        "term": name,
        "term_kind": "mental_move",
        "definition": definition,
        "usage_rule": "Use only when the companion actually performs this move in this turn.",
        "stability": 1.0,
        "content": f"mental move {name}",
    } for name, definition in MENTAL_MOVES]


def seed_terms() -> list[dict]:
    """Return every seed term. Order is stable so term ids are predictable."""
    return [*_emotion_terms(), *_relationship_terms(), *_mental_move_terms()]


def term_lines(context) -> list[str]:
    """Render the ontology terms of one public context, without the bindings.

    A term line carries the word, its kind, and its definition and usage rule.
    The generation prompt uses this render, so the companion can draw on the
    same words the review checks.
    """
    lines = []
    for item in context.ontology:
        if item.get("term") is None:
            continue
        term = str(item.get("term") or "").strip()
        if not term:
            continue
        definition = str(item.get("definition") or "").strip()
        rule = str(item.get("usage_rule") or "").strip()
        detail = "; ".join(part for part in (definition, rule) if part)
        lines.append(f"- {term} ({item.get('term_kind', 'term')}): {detail}" if detail else f"- {term}")
    return lines


def ontology_lines(context) -> list[str]:
    """Render the terms and the active bindings of one public context.

    A binding line says that the term fits the accepted state right now. The
    review prompt uses this render.
    """
    # A term record carries ``term``; a binding record carries ``term_id``.
    terms = {int(item["id"]): item for item in context.ontology if item.get("term") is not None}
    lines = term_lines(context)
    for item in context.ontology:
        if item.get("term_id") is None or item.get("active") is False:
            continue
        term = terms.get(int(item.get("term_id", -1)), {})
        name = str(term.get("term") or item.get("term_id"))
        lines.append(f"- {name} fits right now (fit {item.get('fit_score')}): {item.get('explanation', '')}")
    return lines


def _existing_terms(records: dict) -> dict[str, int]:
    """Map a term name to the committed term id."""
    found = {}
    for key, raw in records.items():
        if raw["__type__"] != "OntologyTermKnoxel":
            continue
        data = raw["data"]
        name = str(data.get("term") or "").strip()
        if name:
            found[name] = int(data.get("id", key))
    return found


def _latest_state_feature(ghost, records: dict) -> dict | None:
    """Return the newest accepted internal-state record, if one exists."""
    states = ghost.read_cycles()
    tick = states[-1].tick_id if states else None
    for raw in reversed(list(records.values())):
        data = raw["data"]
        if data.get("metadata", {}).get("memory_kind") != "internal_state":
            continue
        if tick is None or data.get("tick_id") == tick:
            return data
    return None


def observe_ontology(ghost, *, threshold: float = BIND_THRESHOLD, max_bindings: int = MAX_BINDINGS,
                     authority_id: int | None = None) -> dict:
    """Commit seed terms and bind notable accepted feelings.

    Returns a receipt with the created term ids, created binding ids, and the
    terms that were bound. Repeated calls with the same accepted state make no
    new binding. The caller may ignore the receipt.
    A selected speech action can authorize these derived records. The batch
    retains their input dependencies; the authority does not admit new stimuli.
    """
    with ghost.lock:
        records = ghost.snapshot().serialize_knoxels()
        existing = _existing_terms(records)
        missing = [item for item in seed_terms() if item["term"] not in existing]
        state = _latest_state_feature(ghost, records)
        active = {int(raw["data"].get("term_id", -1)): int(raw["data"].get("id", key))
                  for key, raw in records.items()
                  if raw["__type__"] == "OntologyBindingKnoxel" and raw["data"].get("active", True)}
        if not missing and (state is None or not active):
            return {"created_terms": [], "created_bindings": [], "bound_terms": []}

        batch = ghost.begin_batch(f"ontology:{new_id()}")
        if authority_id is not None:
            batch.depend_on([], authority=authority_id, manifest={"kind": "private_ontology"})
        created_terms = {}
        for item in missing:
            created = batch.create(OntologyTermRecord(
                content=item["content"], term=item["term"], term_kind=item["term_kind"],
                definition=item["definition"], usage_rule=item["usage_rule"],
                stability=item["stability"],
                provenance=Provenance(source_kind=SourceKind.Self, source_ref="private_ontology"),
            ))
            created_terms[item["term"]] = created.id
        known = {**existing, **created_terms}

        created_bindings: list[int] = []
        bound_terms: list[str] = []
        if state is not None:
            vector = state.get("mental_state_appraisal_vec") or state.get("mental_state_latent_vec") or []
            accepted = FullMentalState.init_from_list(vector)
            evidence_ids = [int(item) for item in state.get("metadata", {}).get("evidence_ids", [])]
            support_ids = list(dict.fromkeys([int(state.get("id", 0)), *evidence_ids])) if state.get("id") else evidence_ids
            notable = [(name, float(getattr(accepted.state_emotions, name)))
                       for name in StateEmotions.model_fields
                       if float(getattr(accepted.state_emotions, name)) >= threshold]
            notable.sort(key=lambda pair: pair[1], reverse=True)
            for name, intensity in notable[:max_bindings]:
                term_id = known.get(name)
                if term_id is None:
                    continue
                old_id = active.get(term_id)
                if old_id is not None:
                    old = batch.view().get(old_id).copy_for_edit()
                    if (old is not None and list(old.bound_knoxel_ids) == support_ids
                            and abs(float(old.fit_score) - min(1.0, intensity)) < 0.02):
                        continue
                    if old is not None:
                        old.active = False
                        batch.replace(old)
                binding = batch.create(OntologyBindingRecord(
                    content=f"{name} fits the current accepted state",
                    term_id=term_id, bound_knoxel_ids=support_ids, binding_kind="quale_match",
                    fit_score=min(1.0, intensity), active=True,
                    explanation=f"accepted {name} intensity {intensity:.2f}",
                    provenance=Provenance(source_kind=SourceKind.Self, source_ref="private_ontology"),
                ))
                created_bindings.append(binding.id)
                bound_terms.append(name)
        if not created_terms and not created_bindings:
            return {"created_terms": [], "created_bindings": [], "bound_terms": []}
        ghost.commit(batch)
        return {"created_terms": list(created_terms.values()), "created_bindings": created_bindings,
                "bound_terms": bound_terms}


__all__ = ["BIND_THRESHOLD", "MAX_BINDINGS", "MENTAL_MOVES", "observe_ontology",
           "ontology_lines", "seed_terms", "term_lines"]
