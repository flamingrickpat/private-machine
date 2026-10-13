"""``UserInputProxy``: the bus module that turns terminal or console input into records."""
from __future__ import annotations


from pm.records.enums import FeatureType, InterlocusType

from pm.bus.envelopes import new_id
from pm.host.application import BusModule
from workflows.public_speech.publication import acknowledge_reply
from pm.bus.envelopes import new_envelope
from pm.records.memory_records import new_feature


class UserInputProxy(BusModule):
    """Wraps terminal input and prints public replies."""

    DATA_TOPICS = frozenset({"dialogue.reply"})

    def __init__(self, module_id, bus, on_error, clock, ghost, emit_reply):
        super().__init__(module_id, bus, on_error, clock, None)
        self.ghost = ghost
        self.emit_reply = emit_reply

    def submit(self, text: str, *, session: str = "") -> str:
        correlation_id = new_id()
        feature = new_feature(
            self.ghost, text, FeatureType.Dialogue,
            source=self.ghost.get_user_name(), interlocus=InterlocusType.Public,
            metadata={"exchange_correlation": correlation_id, "exchange_status": "pending", "session": session},
        )
        self.bus.put(new_envelope(
            self.module_id, "data", "user.input",
            correlation_id, {"knoxel_id": feature.id},
        ))
        return correlation_id

    def handle_batch(self, batch) -> None:
        for envelope in batch:
            feature = self.ghost.get_knoxel_by_id(envelope.payload["knoxel_id"])
            if feature is not None:
                self.emit_reply(feature.content)
                acknowledge_reply(self.ghost, feature.id)

