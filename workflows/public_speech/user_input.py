"""``UserInputProxy``: the bus module that turns terminal, console, or web input into records."""
from __future__ import annotations


from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind

from pm.bus.envelopes import new_id
from pm.host.application import BusModule
from workflows.public_speech.publication import acknowledge_reply
from pm.bus.envelopes import new_envelope
from pm.records.base import Provenance
from pm.records.memory_records import Entity, new_feature


class UserInputProxy(BusModule):
    """Wraps human input and prints public replies."""

    DATA_TOPICS = frozenset({"dialogue.reply"})

    def __init__(self, module_id, bus, on_error, clock, ghost, emit_reply):
        super().__init__(module_id, bus, on_error, clock, None)
        self.ghost = ghost
        self.emit_reply = emit_reply

    def submit(self, text: str, *, session: str = "", speaker: str | None = None) -> str:
        """Commit one public message of a human speaker and announce it on the bus.

        ``speaker`` is the name of the person who speaks. ``None`` or the
        configured user name means the user, so the terminal keeps its old
        behavior. Another name is a different person: the message gets the
        entity of that name, and a new name gets a new ``Human`` entity in the
        same commit. Main cognition then puts the message into the stimulus
        group of that entity, not into the group of the user.

        The record commits before the bus event, so a crash after the commit
        keeps the pending exchange for recovery. Returns the exchange
        correlation ID.
        """
        correlation_id = new_id()
        metadata = {"exchange_correlation": correlation_id, "exchange_status": "pending", "session": session}
        if speaker is None or speaker == self.ghost.get_user_name():
            feature = new_feature(
                self.ghost, text, FeatureType.Dialogue,
                source=self.ghost.get_user_name(), interlocus=InterlocusType.Public,
                metadata=metadata,
            )
            knoxel_id = feature.id
        else:
            # The lock keeps the entity lookup and the commit together, so two
            # first messages of one new person cannot create two entities.
            with self.ghost.lock:
                batch = self.ghost.begin_batch(new_id())
                entity_id = self.ghost.get_entity_id(speaker)
                if entity_id is None:
                    entity_id = batch.create(Entity(
                        content=speaker, aliases=[], entity_class=EntityClass.Human,
                        provenance=Provenance(source_kind=SourceKind.User, source_ref=speaker),
                    )).id
                # SourceKind.User marks a human report, so publication can
                # resolve this exchange like an exchange of the user.
                feature = new_feature(
                    self.ghost, text, FeatureType.Dialogue,
                    source=speaker, interlocus=InterlocusType.Public, source_kind=SourceKind.User,
                    metadata=metadata, entity_id=entity_id, changes=batch,
                )
                self.ghost.commit(batch)
            knoxel_id = feature.id
        self.bus.put(new_envelope(
            self.module_id, "data", "user.input",
            correlation_id, {"knoxel_id": knoxel_id},
        ))
        return correlation_id

    def handle_batch(self, batch) -> None:
        for envelope in batch:
            feature = self.ghost.get_knoxel_by_id(envelope.payload["knoxel_id"])
            if feature is not None:
                self.emit_reply(feature.content)
                acknowledge_reply(self.ghost, feature.id)
