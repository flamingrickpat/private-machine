"""``UserInputProxy``: the bus module that turns terminal, console, or web input into records."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import threading
from typing import TYPE_CHECKING

from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind

from pm.bus.envelopes import Envelope, new_id
from pm.host.application import BusModule
from workflows.public_speech.publication import acknowledge_reply
from pm.bus.envelopes import new_envelope
from pm.records.base import Provenance
from pm.records.media import RecordImage, attach_images, with_tags
from pm.records.memory_records import Entity, new_feature

if TYPE_CHECKING:
    from pm.bus.bus import BusSystem
    from pm.memory.store import MemoryStore


class UserInputProxy(BusModule):
    """Wraps human input and prints public replies."""

    DATA_TOPICS = frozenset({"dialogue.reply"})

    def __init__(self, module_id: str, bus: BusSystem, on_error: Callable[[BaseException], None],
                 clock: Callable[[], float], ghost: MemoryStore,
                 emit_reply: Callable[[str], None]) -> None:
        """Retain the store that owns the records and the public reply sink."""
        super().__init__(module_id, bus, on_error, clock, None)
        self.ghost = ghost
        self.emit_reply = emit_reply
        self._intake_lock = threading.RLock()
        self._accepting = True

    def close_intake(self) -> None:
        """Reject new stimuli after all current submissions finish."""
        with self._intake_lock:
            self._accepting = False

    def open_intake(self) -> None:
        """Reopen a controller-frozen input after its checkpoint barrier."""
        with self._intake_lock:
            self._accepting = True

    def submit(self, text: str, *, session: str = "", speaker: str | None = None,
               images: Sequence[RecordImage] = (), delivery_id: str = "", entity_ref: str = "") -> str:
        """Durably deliver a stimulus. Closed intake raises RuntimeError.

        A repeated delivery ID reuses its exchange. Different content under
        that ID raises ValueError. The intake fence also covers terminal input.
        """
        with self._intake_lock:
            if not self._accepting:
                raise RuntimeError("Stimulus intake is closed.")
            if delivery_id:
                for record in self.ghost.all_features:
                    if record.metadata.get("delivery_id") == delivery_id:
                        if record.content != text or record.override_source != (speaker or self.ghost.get_user_name()):
                            raise ValueError("Delivery ID already names a different stimulus.")
                        return record.metadata["exchange_correlation"]
            return self._submit(text, session=session, speaker=speaker, images=images,
                                delivery_id=delivery_id, entity_ref=entity_ref)

    def _submit(self, text: str, *, session: str, speaker: str | None,
                images: Sequence[RecordImage], delivery_id: str, entity_ref: str) -> str:
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

        ``images`` are attached to the dialogue record, and their tags follow
        the text, one per line. With images, ``text`` may be empty.
        """
        correlation_id = new_id()
        metadata = {"exchange_correlation": correlation_id, "exchange_status": "pending", "session": session,
                    "delivery_id": delivery_id, "entity_ref": entity_ref}
        if images:
            text = with_tags(text, attach_images(metadata, images))
        # Intake stores the stimulus before any inference. The cognition worker
        # can prepare its vector after this durable acknowledgment.
        with self.ghost.lock:
            batch = self.ghost.begin_batch(new_id(), embed=False)
            batch.depend_on([], manifest={"kind": "stimulus_delivery", "embedding_policy": "deferred"})
            if not entity_ref and (speaker is None or speaker == self.ghost.get_user_name()):
                feature = new_feature(
                    self.ghost, text, FeatureType.Dialogue,
                    source=self.ghost.get_user_name(), interlocus=InterlocusType.Public,
                    metadata=metadata, changes=batch,
                )
            else:
                if entity_ref:
                    entity_id = next((item.id for item in self.ghost.all_entities
                                      if item.metadata.get("external_entity_ref") == entity_ref), None)
                else:
                    entity_id = self.ghost.get_entity_id(speaker)
                if entity_id is None:
                    entity_id = batch.create(Entity(
                        content=speaker or self.ghost.get_user_name(), aliases=[],
                        entity_class=EntityClass.Agent if entity_ref else EntityClass.Human,
                        metadata={"external_entity_ref": entity_ref} if entity_ref else {},
                        provenance=Provenance(source_kind=SourceKind.User, source_ref=speaker),
                    )).id
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

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Print each committed public reply and mark it delivered."""
        for envelope in batch:
            feature = self.ghost.get_knoxel_by_id(envelope.payload["knoxel_id"])
            if feature is not None:
                self.emit_reply(feature.content)
                acknowledge_reply(self.ghost, feature.id)
