"""Capability overlay: design, wiring, and observed execution stay separate.

A18 must answer "what can this running instance do?" without collapsing a plan,
source code, an active configuration, and a real receipt into one flag. This
module builds a per-capability overlay with one evidence list per state:

``intended``
    A plan or documented requirement.
``implemented``
    Source with an identified entry point.
``enabled``
    An active assembly, configuration, model connection, or tool grant.
``observed``
    A receipt or trace showing actual execution and its result.
``unavailable``
    A missing endpoint, missing grant, or explicit limit.
``uncertain``
    No state reached, or a stale evidence source.

The builder is deterministic. It reads no model and writes no memory. The
caller supplies the active wiring and the real receipts, and every claim keeps
the source reference or revision it was read from.

Reading map: :class:`CapabilityState` names the states,
:class:`CapabilityEvidence` is one evidence item, :class:`CapabilityDeclaration`
says how to gather the evidence for one capability, :class:`CapabilityContext`
carries the live wiring and receipts, :class:`CapabilityFinding` groups the
items per state, and :class:`CapabilityOverlay` adds the date and the
staleness check that a later inspection compares against.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
from typing import TypedDict

from pm.introspection.grants import SourceDenied, SourceInspector, SourceUnavailable

__all__ = [
    "CapabilityDeclaration",
    "CapabilityEvidence",
    "CapabilityFinding",
    "CapabilityOverlay",
    "CapabilityState",
    "build_overlay",
    "default_capability_declarations",
]

#: One real execution row that a caller supplies for the observed state.
#: ``reference`` names the receipt, ``detail`` says what it showed, and
#: ``revision`` is the receipt revision when one exists.
ObservedRow = Mapping[str, object]


class CapabilityEvidenceRow(TypedDict):
    """One serialized evidence item, tagged with the state it supports."""

    state: str
    reference: str
    detail: str
    revision: str
    observed_at: str


class CapabilityFindingRow(TypedDict):
    """One serialized finding: the evidence rows per state and the summary."""

    name: str
    summary: str
    strongest: str
    evidence: dict[str, list[CapabilityEvidenceRow]]
    revision: str
    observed_at: str
    stale: bool


class CapabilityOverlayRow(TypedDict):
    """One serialized overlay with its build time and staleness summary."""

    built_at: str
    stale_names: list[str]
    findings: dict[str, CapabilityFindingRow]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CapabilityState(StrEnum):
    """The distinct evidence states a capability can occupy."""

    Intended = "intended"
    Implemented = "implemented"
    Enabled = "enabled"
    Observed = "observed"
    Unavailable = "unavailable"
    Uncertain = "uncertain"


#: The order used to choose the strongest reached state. ``unavailable`` and
#: ``uncertain`` are not strengths; they describe missing evidence.
_STRENGTH = (
    CapabilityState.Observed,
    CapabilityState.Enabled,
    CapabilityState.Implemented,
    CapabilityState.Intended,
)


@dataclass(frozen=True)
class CapabilityEvidence:
    """One evidence item for one capability state.

    ``reference`` is a grant-relative path, an ``enabled:<name>`` marker, or a
    receipt id. ``revision`` is the digest the evidence was read at, so a later
    inspection can prove the belief stale instead of silently refreshing it.
    """

    state: CapabilityState
    reference: str
    detail: str = ""
    revision: str = ""
    observed_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> CapabilityEvidenceRow:
        """Return the serialized row for a trace, a package, or a belief."""
        return {
            "state": self.state.value,
            "reference": self.reference,
            "detail": self.detail,
            "revision": self.revision,
            "observed_at": self.observed_at.isoformat(),
        }


@dataclass(frozen=True)
class CapabilityDeclaration:
    """How to gather evidence for one named capability.

    ``enabled_names`` and ``unavailable_reasons`` are keys into the build
    context. ``intended_refs`` and ``implemented_refs`` are grant-relative
    file paths. ``observed`` is a callable that returns evidence rows from the
    real receipts, so the declaration never invents execution.
    """

    name: str
    summary: str
    intended_refs: tuple[str, ...] = ()
    implemented_refs: tuple[str, ...] = ()
    observed: Callable[["CapabilityContext"], Sequence[ObservedRow]] | None = None

    def collect(self, context: "CapabilityContext") -> list[CapabilityEvidence]:
        """Return the evidence this declaration can show for this context.

        A missing or denied file becomes an ``unavailable`` item rather than
        support for the state it was declared for.
        """
        evidence: list[CapabilityEvidence] = []
        for state, refs in ((CapabilityState.Intended, self.intended_refs),
                            (CapabilityState.Implemented, self.implemented_refs)):
            for reference in refs:
                item = _file_evidence(state, reference, context)
                if item is not None:
                    evidence.append(item)
        if self.name in context.enabled_names:
            detail = context.enabled_details.get(self.name, "Active assembly or configuration.")
            evidence.append(CapabilityEvidence(
                CapabilityState.Enabled, f"enabled:{self.name}", detail=detail))
        for row in (self.observed(context) if self.observed is not None else []):
            evidence.append(CapabilityEvidence(
                CapabilityState.Observed,
                reference=str(row.get("reference", "")),
                detail=str(row.get("detail", "")),
                revision=str(row.get("revision", "")),
            ))
        return evidence


def _file_evidence(state: CapabilityState, reference: str,
                   context: "CapabilityContext") -> CapabilityEvidence | None:
    """Read one grant-relative file, or record an explicit evidence limit.

    A missing or denied file never becomes support for the requested state.
    It becomes one ``unavailable`` row, so the state list cannot overstate
    what the source showed.
    """
    inspector = context.inspector
    if inspector is None:
        return CapabilityEvidence(CapabilityState.Unavailable, reference,
                                  detail="No source grant is active.")
    try:
        metadata = inspector.digest(reference)
    except (SourceDenied, SourceUnavailable, OSError) as error:
        return CapabilityEvidence(CapabilityState.Unavailable, reference,
                                  detail=f"Not inspectable: {error}")
    return CapabilityEvidence(state, reference, detail="Read from the active source grant.",
                              revision=metadata["sha256"],
                              observed_at=datetime.fromisoformat(metadata["observed_at"]))


@dataclass
class CapabilityContext:
    """The live wiring and real receipts the overlay reads.

    ``enabled_names`` lists what the active assembly, configuration, model
    connection, or tool grant turned on. ``observed_receipts`` maps a
    capability name to real execution rows. Neither field is inferred from the
    other, and a tool name alone never becomes an observation.
    """

    inspector: SourceInspector | None = None
    enabled_names: frozenset[str] = frozenset()
    enabled_details: Mapping[str, str] = field(default_factory=dict)
    observed_receipts: Mapping[str, Sequence[ObservedRow]] = field(default_factory=dict)
    unavailable_reasons: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityFinding:
    """The per-state evidence for one capability.

    ``strongest`` is a convenience summary, not a replacement for the separate
    state lists. A caller that needs "enabled but never observed" reads
    ``evidence["enabled"]`` and ``evidence["observed"]`` directly.
    """

    name: str
    summary: str
    evidence: dict[str, list[CapabilityEvidenceRow]]
    strongest: CapabilityState
    revision: str
    observed_at: datetime = field(default_factory=_utcnow)
    stale: bool = False

    def has(self, state: CapabilityState | str) -> bool:
        """Return whether this finding holds at least one item for ``state``."""
        key = state.value if isinstance(state, CapabilityState) else str(state)
        return bool(self.evidence.get(key))

    def to_dict(self) -> CapabilityFindingRow:
        """Return the serialized finding with every state list separate."""
        return {
            "name": self.name,
            "summary": self.summary,
            "strongest": self.strongest.value,
            "evidence": self.evidence,
            "revision": self.revision,
            "observed_at": self.observed_at.isoformat(),
            "stale": self.stale,
        }


def _finding(declaration: CapabilityDeclaration, collected: list[CapabilityEvidence],
             unavailable_reason: str | None) -> CapabilityFinding:
    """Group one declaration's evidence by state and pick the strongest state."""
    grouped: dict[str, list[CapabilityEvidenceRow]] = {state.value: [] for state in CapabilityState}
    for item in collected:
        grouped[item.state.value].append(item.to_dict())
    reached = [state for state in _STRENGTH if grouped[state.value]]
    if reached:
        strongest = reached[0]
    elif grouped[CapabilityState.Unavailable.value]:
        strongest = CapabilityState.Unavailable
    elif unavailable_reason:
        strongest = CapabilityState.Unavailable
        grouped[CapabilityState.Unavailable.value].append(CapabilityEvidenceRow(
            state=CapabilityState.Unavailable.value,
            reference=f"unavailable:{declaration.name}",
            detail=unavailable_reason,
            revision="",
            observed_at=_utcnow().isoformat(),
        ))
    else:
        # No state was reached and nothing explains the gap. Say so instead of
        # reporting the capability as merely unimplemented.
        strongest = CapabilityState.Uncertain
        grouped[CapabilityState.Uncertain.value].append(CapabilityEvidenceRow(
            state=CapabilityState.Uncertain.value,
            reference=f"uncertain:{declaration.name}",
            detail="No plan, source, wiring, or receipt was found.",
            revision="",
            observed_at=_utcnow().isoformat(),
        ))
    revision = _revision_of(grouped)
    return CapabilityFinding(
        name=declaration.name, summary=declaration.summary, evidence=grouped,
        strongest=strongest, revision=revision,
    )


def _revision_of(grouped: Mapping[str, Sequence[CapabilityEvidenceRow]]) -> str:
    # Observation time is volatile, so it stays out of the revision. The
    # source or configuration digest is the meaningful change signal.
    normalized = {
        state: sorted((row.get("reference", ""), row.get("detail", ""), row.get("revision", ""))
                      for row in rows)
        for state, rows in grouped.items()
    }
    payload = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CapabilityOverlay:
    """A dated overlay of capability findings with a staleness check.

    The previous overlay supplies the revisions a caller compares against.
    A changed revision marks the finding stale until new evidence replaces it.
    """

    findings: dict[str, CapabilityFinding]
    built_at: datetime = field(default_factory=_utcnow)
    previous: "CapabilityOverlay | None" = None

    def stale_names(self) -> list[str]:
        """Return capabilities whose evidence revision changed since ``previous``."""
        if self.previous is None:
            return []
        return sorted(
            name for name, finding in self.findings.items()
            if name in self.previous.findings
            and self.previous.findings[name].revision != finding.revision
        )

    def refresh(self, context: CapabilityContext) -> "CapabilityOverlay":
        """Rebuild against a fresh context and keep this overlay as history."""
        rebuilt = build_overlay(
            [CapabilityDeclaration(name, finding.summary) for name, finding in self.findings.items()],
            context,
        )
        rebuilt.previous = self
        return rebuilt

    def to_dict(self) -> CapabilityOverlayRow:
        """Return the serialized overlay, keyed by capability name."""
        return {
            "built_at": self.built_at.isoformat(),
            "stale_names": self.stale_names(),
            "findings": {name: finding.to_dict() for name, finding in self.findings.items()},
        }


def build_overlay(declarations: Iterable[CapabilityDeclaration],
                  context: CapabilityContext) -> CapabilityOverlay:
    """Collect each declaration's evidence into one dated overlay."""
    findings: dict[str, CapabilityFinding] = {}
    for declaration in declarations:
        collected = declaration.collect(context)
        reason = context.unavailable_reasons.get(declaration.name)
        findings[declaration.name] = _finding(declaration, collected, reason)
    return CapabilityOverlay(findings=findings)


def default_capability_declarations() -> list[CapabilityDeclaration]:
    """Declare the capabilities A18 inspects, with their real evidence paths.

    The declarations name existing files. An entry point that moves must move
    here as well, so the overlay never reports a stale plan as wiring.
    """

    def observed_from_receipts(name: str) -> Callable[[CapabilityContext], Sequence[ObservedRow]]:
        """Return a reader for the real receipts that a caller files under ``name``."""

        def collect(context: CapabilityContext) -> Sequence[ObservedRow]:
            return list(context.observed_receipts.get(name, []))

        return collect

    return [
        CapabilityDeclaration(
            name="public_dialogue",
            summary="Produce an in-character public reply through the configured chat model.",
            intended_refs=("docs/subsystems/public-dialogue.md",),
            implemented_refs=("workflows/dialogue_story/host.py", "workflows/public_speech/public_dialogue.py"),
            observed=observed_from_receipts("public_dialogue"),
        ),
        CapabilityDeclaration(
            name="private_thought",
            summary="Run the inner loop on its own timer with a distinct GWT shard.",
            intended_refs=("docs/subsystems/private-loop.md",),
            implemented_refs=("workflows/inner_thought/host.py", "workflows/inner_thought/pipeline.py"),
            observed=observed_from_receipts("private_thought"),
        ),
        CapabilityDeclaration(
            name="memory_tools",
            summary="Read accepted memory through a scoped grant with revisions.",
            intended_refs=("docs/subsystems/agent-memory.md",),
            implemented_refs=("pm/memory/agent_tools.py",),
            observed=observed_from_receipts("memory_tools"),
        ),
        CapabilityDeclaration(
            name="source_inspection",
            summary="Read selected source, documentation, configuration, and runtime evidence.",
            intended_refs=("docs/subsystems/introspection.md",),
            implemented_refs=("pm/introspection/grants.py", "pm/introspection/capabilities.py"),
            observed=observed_from_receipts("source_inspection"),
        ),
        CapabilityDeclaration(
            name="minecraft_body",
            summary="Observe and act in the Minecraft world through the body transport.",
            intended_refs=("docs/subsystems/minecraft-actor.md",),
            implemented_refs=("workflows/minecraft_body/gateway.py",),
            observed=observed_from_receipts("minecraft_body"),
        ),
    ]
