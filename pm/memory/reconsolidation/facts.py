"""File source occurrences and canonical claims at one episode's publication.

Claim identity includes the observer's reality, the subject, and the source
of testimony. Entity-context statements retain the recorded interpretation
and its authority. A new mention extends the current semantic head.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re

from pm.memory.entity_history import ENTITY_CONTEXT_FAMILY, EntityContextKey, family_key
from pm.memory.reconsolidation.entity_claims import EntityMemoryClaim, EntityMemoryClaims
from pm.memory.reconsolidation.publication import MemoryUnitReceipt
from pm.memory.reconsolidation.rendering import FactSubject
from pm.memory.reconsolidation.schemas import CanonicalFactOption, FactClaimOption, FactFiling
from pm.memory.snapshot import Snapshot
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.enums import FeatureType
from pm.records.lineage import RecordLineage
from pm.records.memory_records import DeclarativeFactRecord, Entity, Feature, MemoryClusterRecord, SemanticRecord
from pm.records.taxonomy import EpistemicStatus, MemoryTaxonomy, MemoryVolatility, RecordStatus, TruthStatus


RELATIVE_TIME = re.compile(r"\b(?:today|tomorrow|yesterday|tonight|this morning|next week|last week)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FiledFact:
    text: str
    filing: FactFiling
    sources: tuple[Feature, ...]
    subject_id: int
    identity: str
    claim: EntityMemoryClaim | None = None


@dataclass(frozen=True, order=True, slots=True)
class CanonicalMatch:
    overlap: float
    last_support: datetime
    head_id: int
    about: str


class EpisodeFacts:
    def __init__(self, view: Snapshot, observer_id: int | None, now: datetime) -> None:
        self.view = view
        self.observer_id = observer_id
        self.entities = {key: row for key, row in view.all_knoxels.items() if isinstance(row, Entity)}
        self.claims = EntityMemoryClaims.from_snapshot(view, observer_id or 0, now)
        self.heads: dict[str, int] = {}  # pm-lint: allow PM004 -- semantic head ID by qualified claim identity
        self.by_authority: dict[str, set[str]] = {}  # pm-lint: allow PM004 -- eligible identities by subject/speaker/scope
        self.families: dict[str, dict[int, int]] = {}  # pm-lint: allow PM004 -- record revisions by entity family
        for record in view.all_knoxels.values():
            if isinstance(record, SemanticRecord):
                if record.status != RecordStatus.Active or record.truth_status != TruthStatus.Truth:
                    continue
                stored_key = record.metadata.get("filing_key")
                if isinstance(stored_key, str):
                    self.index_head(stored_key, record.id)
                elif record.subject_entity_id and record.epistemic_status == EpistemicStatus.Asserted:
                    # Legacy records can join an exact self-report in the same
                    # scope. Ambiguous zero-subject or hearsay rows stay separate.
                    key = self.filing_identity(record.scope, f"entity:{record.subject_entity_id}", record.essence,
                                        f"speaker:{record.subject_entity_id}")
                    if key not in self.heads:
                        self.index_head(key, record.id)
            family = record.metadata.get(ENTITY_CONTEXT_FAMILY)
            if isinstance(family, str):
                if family not in self.families:
                    self.families[family] = {}
                self.families[family][record.id] = view.revisions[record.id]

    def subjects(self, scene: Sequence[Feature], participants: Sequence[int]) -> tuple[FactSubject, ...]:
        visible = set(participants)
        for source in scene:
            for claim in self.claims.by_source.get(source.id, ()):
                if claim.scope == source.scope:
                    visible.add(claim.subject_entity_id)
        text = "\n".join(feature.content for feature in scene).casefold()
        for entity in self.entities.values():
            if self.mentions(entity, text):
                visible.add(entity.id)
        result = [FactSubject(f"entity:{key}", self.entities[key].content)
                  for key in sorted(visible) if key in self.entities]
        result.extend((FactSubject("group", "the participants together"), FactSubject("world", "the environment")))
        return tuple(result)

    @staticmethod
    def mentions(entity: Entity, text: str) -> bool:
        return any(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE) is not None
                   for name in (entity.content, *entity.aliases) if name)

    def claim_options(self, sources: Sequence[Feature]) -> tuple[FactClaimOption, ...]:
        """Offer specific statements for semantic comparison within their actual source grant."""
        options: dict[str, FactClaimOption] = {}  # pm-lint: allow PM004 -- interpretations keyed by entity claim ID
        for source in sources:
            for claim in self.claims.by_source.get(source.id, ()):
                if claim.scope == source.scope:
                    options[claim.claim_id] = FactClaimOption(claim_id=claim.claim_id,
                        about=f"entity:{claim.subject_entity_id}", statement=claim.statement)
        return tuple(options.values())

    @staticmethod
    def filing_identity(scope: RecordScope, subject: str, essence: str, authority: str) -> str:
        return f"{scope.reality_id}:{scope.frame.value}:{subject}:{authority}:{MemoryTaxonomy.normalize_text(essence)}"

    def index_head(self, identity: str, head_id: int) -> None:
        self.heads[identity] = head_id
        if not identity.startswith("entity-context:"):
            # normalize_text removes punctuation from the essence, so the
            # final separator always ends the scope/subject/authority prefix.
            family = identity.rpartition(":")[0]
            self.by_authority.setdefault(family, set()).add(identity)

    @staticmethod
    def source_speakers(sources: Sequence[Feature]) -> list[int]:
        return sorted({row.entity_id for row in sources if row.feature_type == FeatureType.Dialogue and row.entity_id > 0})

    @staticmethod
    def subject_key(about: str, participants: Sequence[int]) -> str:
        if about == "group":
            return "group:" + ",".join(str(key) for key in sorted(participants))
        return about

    @staticmethod
    def source_authority(sources: Sequence[Feature]) -> str:
        people: set[int] = set()
        unanchored: set[str] = set()
        for source in sources:
            if source.feature_type != FeatureType.Dialogue:
                continue
            if source.entity_id > 0:
                people.add(source.entity_id)
            else:
                name = MemoryTaxonomy.normalize_text(source.override_source)
                unanchored.add(f"name-{name}" if name else f"source-{source.id}")
        if not people and not unanchored:
            return "observation"
        witnesses = [str(person) for person in sorted(people)]
        witnesses.extend(sorted(unanchored))
        return "speaker:" + ",".join(witnesses)

    @staticmethod
    def compatible_qualifiers(text: str, prior: SemanticRecord, source_at: datetime) -> bool:
        """Keep explicit reversals, changed numbers and newly relative dates out of a merge."""
        old = prior.content.casefold()
        new = text.casefold()
        negation = r"\b(?:not|never|no longer)\b|n['’]t\b"
        if bool(re.search(negation, old)) != bool(re.search(negation, new)):
            return False
        old_numbers = set(re.findall(r"\b\d+(?:[.:]\d+)?\b", old))
        new_numbers = set(re.findall(r"\b\d+(?:[.:]\d+)?\b", new))
        if old_numbers and new_numbers and old_numbers != new_numbers:
            return False
        if prior.timestamp_world_end.date() != source_at.date() and RELATIVE_TIME.search(old + " " + new):
            return False
        return True

    def canonical_options(self, text: str, sources: tuple[Feature, ...], subjects: Sequence[FactSubject],
                          participants: Sequence[int]) -> tuple[CanonicalFactOption, ...]:
        """Rank only live unlinked claims with matching speaker evidence and scope."""
        authority = self.source_authority(sources)
        query = set(MemoryTaxonomy.normalize_text(text).split())
        ranked: list[CanonicalMatch] = []
        for subject in subjects:
            qualified = self.subject_key(subject.identity, participants)
            family = self.filing_identity(sources[0].scope, qualified, "", authority).removesuffix(":")
            for identity in self.by_authority.get(family, ()):
                row = self.view.get(self.heads[identity])
                if not isinstance(row, SemanticRecord):
                    raise TypeError(f"Canonical claim {identity} names a {row.type} record.")
                if not self.compatible_qualifiers(text, row, sources[0].timestamp_world_begin):
                    continue
                terms = set(MemoryTaxonomy.normalize_text(row.content + " " + row.essence).split())
                overlap = len(query & terms) / max(len(query | terms), 1)
                ranked.append(CanonicalMatch(overlap, row.timestamp_world_end, row.id, subject.identity))
        result: list[CanonicalFactOption] = []
        for match in sorted(ranked, reverse=True)[:6]:
            row = self.view.get(match.head_id)
            assert isinstance(row, SemanticRecord)
            result.append(CanonicalFactOption(head_id=row.id, about=match.about, statement=row.content,
                                              record_type=row.record_type, supported_through=row.timestamp_world_end))
        return tuple(result)

    def attribute(self, text: str, filing: FactFiling, sources: tuple[Feature, ...],
                  participants: Sequence[int]) -> list[FiledFact]:
        subject = filing.about
        subject_id = int(subject.removeprefix("entity:")) if subject.startswith("entity:") else 0
        scope = sources[0].scope
        source_ids = [item.id for item in sources]
        claims = self.claims.for_sources(source_ids, scope, subject_id)
        matched: EntityMemoryClaim | None = None
        if filing.entity_claim is not None:
            matched = next(claim for claim in claims if claim.claim_id == filing.entity_claim)
        else:
            wording = {MemoryTaxonomy.normalize_text(text), MemoryTaxonomy.normalize_text(filing.essence)}
            matches = [claim for claim in claims if MemoryTaxonomy.normalize_text(claim.statement) in wording]
            if len(matches) == 1:
                matched = matches[0]
        if matched is not None:
            if matched.status in {"retracted", "expired", "future"}:
                return []
            epistemic = {"observation": EpistemicStatus.Observed, "report": EpistemicStatus.ReportedByOther,
                         "inference": EpistemicStatus.Inferred, "hypothetical": EpistemicStatus.Hypothesis}
            qualified = filing.model_copy()
            qualified.epistemic_status = epistemic[matched.epistemic_kind]
            qualified.essence = matched.statement
            evidence = tuple(row for key in matched.source_ids if isinstance(row := self.view.get(key), Feature))
            if not evidence:
                return []
            content = matched.statement if matched.status != "contested" else f"Contested: {matched.statement}"
            return [FiledFact(content, qualified, evidence, subject_id, f"entity-context:{matched.claim_id}", matched)]
        # A private interpretation alone cannot file another person's actual
        # preference or an external-world observation.
        private_only = all(item.feature_type not in {FeatureType.Dialogue, FeatureType.WorldEvent} for item in sources)
        if private_only:
            return []
        reporters = self.source_speakers(sources)
        subject = self.subject_key(subject, participants)
        authority = self.source_authority(sources)
        qualified = filing.model_copy()
        unanchored = {row.override_source or "an unidentified speaker" for row in sources
                      if row.feature_type == FeatureType.Dialogue and row.entity_id <= 0}
        if all(item.feature_type == FeatureType.Dialogue for item in sources):
            if subject_id and not unanchored and reporters == [subject_id]:
                qualified.epistemic_status = EpistemicStatus.Asserted
            elif subject.startswith("group:") and not unanchored and set(reporters) == set(participants):
                qualified.epistemic_status = EpistemicStatus.Asserted
            else:
                qualified.epistemic_status = EpistemicStatus.ReportedByOther
                names = ", ".join([*(self.entities[reporter].content for reporter in reporters), *sorted(unanchored)])
                text = f"{names} reported: {text}"
                qualified.essence = f"{names} reported: {filing.essence}"
        dated_essence = filing.essence
        if RELATIVE_TIME.search(text) or RELATIVE_TIME.search(dated_essence):
            dated_essence += f" (stated {sources[0].timestamp_world_begin:%Y-%m-%d})"
        key = self.filing_identity(scope, subject, dated_essence, authority)
        if key in self.heads:
            previous = self.view.get(self.heads[key])
            if not isinstance(previous, SemanticRecord):
                raise TypeError(f"Canonical claim {key} names a {previous.type} record.")
            if not self.compatible_qualifiers(text, previous, sources[0].timestamp_world_begin):
                key = self.filing_identity(scope, subject, text + " " + dated_essence, authority)
        if key not in self.heads and filing.canonical_id is not None:
            family = key.rpartition(":")[0]
            for existing in self.by_authority[family]:
                if self.heads[existing] == filing.canonical_id:
                    key = existing
                    break
            else:
                raise ValueError(f"Selected canonical head {filing.canonical_id} is outside {family}.")
        return [FiledFact(text, qualified, sources, subject_id, key)]

    def guard_subjects(self, batch: WriteBatch, scope: RecordScope, subject_ids: Sequence[int]) -> None:
        """Qualify filing against the same family even when its first receipt arrives mid-preparation."""
        if self.observer_id is None:
            return
        for subject_id in set(subject_ids):
            if not subject_id:
                continue
            key = EntityContextKey(subject_entity_id=subject_id, observer_entity_id=self.observer_id, scope=scope)
            family = family_key(key)
            if family not in self.families:
                self.families[family] = {}
            revisions = self.families[family]
            batch.expect_metadata_version(ENTITY_CONTEXT_FAMILY, family, revisions)

    def prerequisites(self, facts: Sequence[FiledFact]) -> set[int]:
        result: set[int] = set()
        for fact in facts:
            result.update(item.id for item in fact.sources)
            if fact.identity in self.heads:
                result.add(self.heads[fact.identity])
            if fact.claim is not None:
                result.add(fact.claim.head_receipt_id)
        return result

    def stage(self, batch: WriteBatch, episode: MemoryClusterRecord, facts: Sequence[FiledFact],
              receipt: MemoryUnitReceipt, provenance: Provenance) -> None:
        edits: dict[str, SemanticRecord] = {}  # pm-lint: allow PM004 -- semantic successor by qualified claim identity
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for item in facts:
            ids = tuple(sorted({source.id for source in item.sources}))
            occurrence = (item.identity, ids)
            if occurrence in seen:
                continue
            seen.add(occurrence)
            filing = item.filing
            parents = [episode.id]
            if item.claim is not None:
                parents.append(item.claim.receipt_id)
            lineage = RecordLineage(episode_id=batch.commit_id, operation="memory_reconsolidation.fact",
                                    source_event_ids=list(ids), parent_record_ids=parents)
            fact = batch.create(DeclarativeFactRecord(content=item.text, scope=episode.scope,
                subject_entity_id=item.subject_id, record_type=filing.record_type,
                object_kind=filing.object_kind, object_value=filing.object_value, epistemic_status=filing.epistemic_status,
                confidence=filing.confidence, topic_card=filing.topic_card, essence=filing.essence,
                source_cluster_id=episode.id, reason=filing.reason, category=filing.category, importance=filing.importance,
                time_dependent=filing.time_dependent, timestamp_world_begin=min(row.timestamp_world_begin for row in item.sources),
                timestamp_world_end=max(row.timestamp_world_end for row in item.sources),
                min_event_id=min(ids), max_event_id=max(ids), min_tick_id=min(row.tick_id for row in item.sources),
                max_tick_id=max(row.tick_id for row in item.sources), provenance=provenance, lineage=lineage))
            fact.metadata["memory_kind"] = "experience_summary"
            current = edits.get(item.identity)
            if current is None and item.identity in self.heads:
                previous = self.view.get(self.heads[item.identity])
                if not isinstance(previous, SemanticRecord):
                    raise TypeError(f"Semantic head {previous.id} has kind {previous.type}.")
                current = batch.replace(previous.copy_for_edit())
                current.timestamp_world_begin = previous.timestamp_world_begin
                current.timestamp_world_end = previous.timestamp_world_end
                if not previous.metadata.get("filing_source_interval"):
                    # Older heads may be dated when they were filed. Recover
                    # the interval from the occurrences they actually cite.
                    support = [self.view.get(key) for key in previous.source_knoxel_ids]
                    if support:
                        current.timestamp_world_begin = min(row.timestamp_world_begin for row in support)
                        current.timestamp_world_end = max(row.timestamp_world_end for row in support)
                receipt.semantic_updated_ids.append(current.id)
            if current is None:
                label = self.entities[item.subject_id].content if item.subject_id else filing.about
                digest = hashlib.sha256(item.identity.encode()).hexdigest()[:20]
                current = batch.create(SemanticRecord(content=item.text, scope=episode.scope, record_id=f"mem_{digest}",
                    subject_label=label, subject_entity_id=item.subject_id, record_type=filing.record_type,
                    object_kind=filing.object_kind, object_value=filing.object_value, epistemic_status=filing.epistemic_status,
                    confidence=filing.confidence, topic_card=filing.topic_card, essence=filing.essence,
                    timestamp_world_begin=fact.timestamp_world_begin, timestamp_world_end=fact.timestamp_world_end,
                    importance=filing.importance, volatility=MemoryVolatility.Ephemeral if filing.time_dependent >= .7 else MemoryVolatility.Stable,
                    provenance=provenance, lineage=RecordLineage(episode_id=batch.commit_id,
                        operation=lineage.operation, source_event_ids=list(ids), parent_record_ids=[fact.id])))
                receipt.semantic_ids.append(current.id)
            current.semantic_record_id = current.id
            current.source_knoxel_ids.append(fact.id)
            if item.text != current.content and item.text not in current.source_texts:
                current.source_texts.append(item.text)
            current.importance = max(current.importance, filing.importance)
            current.timestamp_world_begin = min(current.timestamp_world_begin, fact.timestamp_world_begin)
            current.timestamp_world_end = max(current.timestamp_world_end, fact.timestamp_world_end)
            current.metadata["filing_source_interval"] = True
            current.metadata["filing_key"] = item.identity
            if item.claim is not None:
                for record in (fact, current):
                    record.metadata["entity_claim_id"] = item.claim.claim_id
                    record.metadata["entity_claim_receipt_id"] = item.claim.receipt_id
                    record.metadata["entity_claim_head_id"] = item.claim.head_receipt_id
                    record.metadata["entity_claim_status"] = item.claim.status
            fact.semantic_record_id = current.id
            receipt.fact_ids.append(fact.id)
            edits[item.identity] = current
        for key, current in edits.items():
            self.index_head(key, current.id)
