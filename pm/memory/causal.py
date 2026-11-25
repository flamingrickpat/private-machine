"""Immutable transaction history and the accepted recovery projection.

Main cognition admits input transactions. Explicit recovery roots retain durable
input and control receipts without admitting speculative dependencies. An approved
action can add results only when its inputs belong to accepted history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def public_acknowledgement_id(reply_id: int) -> str:
    """Return the stable commit ID of delivery for a public reply."""
    return f"public-ack:{reply_id}"


@dataclass(frozen=True, slots=True)
class PublishedReply:
    """Link the utterance to the receipt that it reached public delivery."""

    reply_id: int
    publication_commit_id: str
    acknowledgement_commit_id: str


class Transaction(BaseModel):
    """Keep exact input IDs and atomic effects of one durable commit.

    Dependencies name transactions, not structural record links. A main commit
    admits its dependency closure. Other commits need an accepted authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    commit_id: str
    sequence: int
    input_ids: tuple[int, ...] = ()
    dependencies: tuple[str, ...] = ()
    created_ids: tuple[int, ...] = ()
    superseded_ids: tuple[int, ...] = ()
    tombstone_ids: tuple[int, ...] = ()
    lifecycle: dict[int, dict] = Field(default_factory=dict)
    runs: dict[str, dict] = Field(default_factory=dict)
    consolidations: dict[str, dict] = Field(default_factory=dict)
    tick_id: int
    main: bool = False
    recovery_root: bool = Field(default=False, description="Retain an append-only input or control receipt independently of a main cycle.")
    authority: str | None = None
    authority_record_id: int | None = None
    baseline_authority: bool = False
    manifest: dict = Field(default_factory=dict)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore Pydantic's pickle payload and absent legacy admission fields.

        The opaque mapping is the external Pydantic serialization contract.
        Old transactions remain ordinary branches without a recovery marker.
        """
        super().__setstate__(state)
        self.__dict__.setdefault("authority_record_id", None)
        self.__dict__.setdefault("baseline_authority", False)
        self.__dict__.setdefault("recovery_root", False)


class CausalHistory(BaseModel):
    """Retain the import baseline and all later transaction branches.

    Legacy checkpoints become explicit baselines. Their missing past versions
    cannot be recovered. Sequence numbers never establish acceptance alone.
    """

    model_config = ConfigDict(extra="forbid")
    baseline_ids: tuple[int, ...] = ()
    baseline_ticks: tuple[int, ...] = ()
    baseline_runs: dict[str, dict] = Field(default_factory=dict)
    baseline_consolidations: dict[str, dict] = Field(default_factory=dict)
    baseline_commit_ids: tuple[str, ...] = ()
    transactions: dict[str, Transaction] = Field(default_factory=dict)

    def __setstate__(self, state):
        """Keep trusted earlier causal checkpoints readable after this addition."""
        super().__setstate__(state)
        self.__dict__.setdefault("baseline_commit_ids", ())

    def published_replies(self) -> list[PublishedReply]:
        """Read delivery evidence from the same receipts used by existing checkpoints."""
        replies: list[PublishedReply] = []
        for receipt in self.transactions.values():
            reply_id = receipt.authority_record_id
            if reply_id is None or receipt.authority is None:
                continue
            logical_id = receipt.manifest.get("logical_commit_id", receipt.commit_id)
            if logical_id != public_acknowledgement_id(reply_id):
                continue
            publication = self.transactions[receipt.authority]
            if publication.manifest.get("kind") != "public_reply":
                continue
            if reply_id not in publication.created_ids or receipt.lifecycle[reply_id]["publication_status"] != "published":
                raise ValueError(f"Invalid public delivery receipt {receipt.commit_id!r} for record {reply_id}.")
            replies.append(PublishedReply(reply_id, publication.commit_id, receipt.commit_id))
        return replies

    def accepted(self) -> set[str]:
        """Compute accepted transactions or raise for a broken causal graph.

        The graph uses commit order. Reject forward links and missing inputs
        instead of treating incomplete evidence as accepted history.
        """
        accepted: set[str] = set()
        sequences = [item.sequence for item in self.transactions.values()]
        if len(set(sequences)) != len(sequences) or any(value < 1 for value in sequences):
            raise ValueError("Transaction sequences must be positive and unique.")
        for transaction in self.transactions.values():
            if transaction.recovery_root and (
                transaction.main or transaction.superseded_ids or transaction.tombstone_ids
                or transaction.lifecycle or transaction.runs or transaction.consolidations
                or transaction.manifest.get("gwt")
            ):
                raise ValueError("A recovery root must only append input or control records.")
            if transaction.baseline_authority and transaction.authority_record_id not in self.baseline_ids:
                raise ValueError("A baseline action authority must identify an imported record.")
            for dependency in (*transaction.dependencies, *([transaction.authority] if transaction.authority else [])):
                parent = self.transactions.get(dependency)
                if parent is None or parent.sequence >= transaction.sequence:
                    raise ValueError(f"Invalid causal dependency {dependency!r} in {transaction.commit_id!r}.")

        publications = self.published_replies()
        published_by_ack = {item.acknowledgement_commit_id: item.publication_commit_id
                            for item in publications}
        expanded: set[str] = set()

        def admit(commit_id: str) -> None:
            pending = [commit_id]
            while pending:
                key = pending.pop()
                if key in expanded:
                    continue
                expanded.add(key)
                transaction = self.transactions[key]
                accepted.add(key)
                if key in published_by_ack:
                    accepted.add(published_by_ack[key])
                    continue
                pending.extend(transaction.dependencies)
                if transaction.authority is not None:
                    pending.append(transaction.authority)

        for transaction in self.transactions.values():
            if transaction.main:
                admit(transaction.commit_id)
        # Completed background work is its own authority. Reconsolidation and chat import
        # commit outside a workspace slot, so they can never be ``main``; the commit that
        # records their finished consolidation receipt names the commits it finished as
        # dependencies, so admitting it admits that run and no partial one. A run that
        # stopped before its receipt keeps its work unaccepted, as unconfirmed work must.
        for transaction in self.transactions.values():
            if transaction.consolidations:
                admit(transaction.commit_id)
        # Public delivery proves the utterance happened. It does not admit
        # the unfinished private work that the original draft could see.
        for publication in publications:
            accepted.add(publication.acknowledgement_commit_id)
            accepted.add(publication.publication_commit_id)
        # Main admission can release an older result. The ordered pass then
        # releases its descendants without recursion or a fixed depth limit.
        for transaction in sorted(self.transactions.values(), key=lambda item: item.sequence):
            if transaction.recovery_root:
                missing = set(transaction.dependencies) - accepted
                if transaction.authority is not None and transaction.authority not in accepted:
                    missing.add(transaction.authority)
                if missing:
                    raise ValueError(f"Recovery root {transaction.commit_id!r} depends on unaccepted transactions: {sorted(missing)}")
                accepted.add(transaction.commit_id)
            if (transaction.commit_id not in accepted and (transaction.authority in accepted or transaction.baseline_authority)
                    and set(transaction.dependencies).issubset(accepted)):
                accepted.add(transaction.commit_id)
        return accepted

    def producers(self) -> dict[int, str]:
        """Map each appended record to its single creating transaction."""
        result = {}
        for transaction in self.transactions.values():
            for record_id in transaction.created_ids:
                if record_id in result or record_id in self.baseline_ids:
                    raise ValueError(f"Duplicate historical knoxel ID: {record_id}.")
                result[record_id] = transaction.commit_id
        return result
