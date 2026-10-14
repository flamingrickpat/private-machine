"""Versioned emotional pathways over accepted codelet proposals.

A pathway connects mechanisms through evidence. An accepted (selected) codelet
proposal from a recent committed cycle is a *precursor*: it raises the
pathway's activation for the next cycles, and that activation can boost later
codelets that belong to the same pathway. The pathway is a state of mind that
persists between cycles, not a fixed sequence of dialogue lines.

The main module derives precursors fresh from committed ``AttentionRecord``
evidence on every cycle. Nothing is cached, so a restart keeps the pathway
state (it lives in the MemoryStore) and no concurrent or interrupted cycle can
reuse stale pathway state. Before a precursor counts, the module re-reads the
record, so a precursor whose revision changed cannot drive activation.

Activation decays per committed cycle. Several precursors add up, capped at
1.0, so a repeated concern can stay strong while a single old one fades. The
scheduler sees the result as the signal ``pathway.<name>``.

This module is pure: no model calls, no MemoryStore access, no file writes.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

PATHWAY_SIGNAL_PREFIX = "pathway."


class PathwayDefinition(BaseModel):
    """One named pathway as declared in the codelet catalogue.

    ``members`` are the codelets whose accepted proposals feed the pathway.
    ``emotions`` names the implemented state fields that the pathway maps to;
    a named human emotion without an implemented field must be documented as a
    mapping, never as an invented state mechanism.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = ""
    emotions: list[str] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)
    window_ticks: int = Field(default=6, ge=1)
    decay_per_tick: float = Field(default=0.6, ge=0.0, le=1.0)

    def member_set(self) -> set[str]:
        return set(self.members)


def pathway_map(pathways: list[PathwayDefinition]) -> dict[str, PathwayDefinition]:
    """Return the pathways by name. Duplicate names stop the catalogue load."""
    result: dict[str, PathwayDefinition] = {}
    for pathway in pathways:
        if pathway.name in result:
            raise ValueError(f"Duplicate pathway name: {pathway.name}.")
        result[pathway.name] = pathway
    return result


def precursors_by_pathway(
    precursors: list[dict],
    pathways: dict[str, PathwayDefinition],
    *,
    latest_tick: int,
) -> dict[str, list[dict]]:
    """Group fresh precursor records under their pathway names.

    A precursor is one committed ``AttentionRecord`` summary: it must be
    selected, name a member codelet, and fall inside the pathway's window.
    Older or unselected records are dropped here, so the caller cannot boost
    from stale or losing evidence.
    """
    grouped: dict[str, list[dict]] = {}
    for pathway in pathways.values():
        members = pathway.member_set()
        for item in precursors or []:
            if not item.get("selected", False):
                continue
            if str(item.get("codelet", "")) not in members:
                continue
            tick_age = int(latest_tick) - int(item.get("tick_id", latest_tick))
            if tick_age < 0 or tick_age >= pathway.window_ticks:
                continue
            entry = dict(item)
            entry["tick_age"] = tick_age
            entry["pathway"] = pathway.name
            entry["pathway_version"] = pathway.version
            grouped.setdefault(pathway.name, []).append(entry)
    return grouped


def pathway_signals(
    precursors: list[dict],
    pathways: dict[str, PathwayDefinition],
    *,
    latest_tick: int,
) -> dict[str, float]:
    """Return ``pathway.<name> -> activation in 0..1`` for this cycle.

    Each precursor contributes ``decay_per_tick ** tick_age``. The sum is
    capped at 1.0. A pathway with no fresh precursor stays at 0, which closes
    every boost gate on it.
    """
    signals: dict[str, float] = {}
    for name, entries in precursors_by_pathway(precursors, pathways, latest_tick=latest_tick).items():
        pathway = pathways[name]
        total = sum(pathway.decay_per_tick ** float(entry["tick_age"]) for entry in entries)
        signals[PATHWAY_SIGNAL_PREFIX + name] = round(min(1.0, total), 4)
    return signals


__all__ = [
    "PATHWAY_SIGNAL_PREFIX",
    "PathwayDefinition",
    "pathway_map",
    "pathway_signals",
    "precursors_by_pathway",
]
