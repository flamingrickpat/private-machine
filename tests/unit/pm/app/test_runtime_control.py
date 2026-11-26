"""Exercise the bound operator against real intake and persisted checkpoint formats."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.app.control import DeliveryState, OperatorApplication, OperatorStimulus, TraceReceipt
from pm.bus.bus import BusSystem
from pm.memory.checkpoint import write_checkpoint
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from workflows.companion import build_app
from workflows.public_speech.user_input import UserInputProxy


class RuntimeControlTests(unittest.TestCase):
    def test_operator_uses_the_assembled_intake_store_and_snapshot_location(self) -> None:
        with TemporaryDirectory() as home:
            root = Path(home)
            store = MemoryStore(PersonaConfig(companion_name="Spamuel", user_name="Jesse"))
            app = build_app(None, root, ghost=store, cognition="dummy", enable_minecraft=False, record_lifecycle=False)
            operator = app.runtime_control
            try:
                self.assertIs(operator.intake, app.user_input)
                self.assertIs(operator.ghost, store)
                operator("/control/freeze", {})  # pm-lint: allow PM005 -- empty HTTP control request
                stimulus = OperatorStimulus(text="Good morning.", delivery_id="first").model_dump(mode="json")  # pm-lint: allow PM007 -- HTTP control stimulus
                with self.assertRaises(RuntimeError):
                    operator("/control/send", stimulus)
                operator("/control/snapshot", {"name": "first"})  # pm-lint: allow PM005 -- HTTP snapshot request
                self.assertTrue((root / "snapshots" / "first.pickle").is_file())
                self.assertFalse((root / "ghost.pickle").exists())
                operator("/control/thaw", {})  # pm-lint: allow PM005 -- empty HTTP control request
                sent = operator("/control/send", stimulus)
                self.assertEqual(sent, operator("/control/send", stimulus))
                message = next(row for row in store.all_features if row.content == "Good morning.")
                lookup = operator("/control/delivery", {"delivery_id": "first"})  # pm-lint: allow PM005 -- HTTP delivery lookup
                self.assertEqual(lookup, DeliveryState(present=True, accepted=True, record_id=message.id))
            finally:
                app.shutdown()
                app.join()

    def test_legacy_presence_does_not_invent_a_transaction(self) -> None:
        with TemporaryDirectory() as home:
            root = Path(home)
            store = MemoryStore(PersonaConfig(companion_name="Spamuel", user_name="Jesse"))
            app = OperatorApplication(BusSystem(), [])
            intake = UserInputProxy("input", app, store, print)
            intake.submit("Before the journal.", delivery_id="legacy-input")
            legacy = store.checkpoint_data()
            legacy.causal_history = None
            path = root / "legacy.pickle"
            write_checkpoint(path, legacy)
            reopened = MemoryStore.from_checkpoint(path, raw=True)
            self.assertIsNone(reopened.causal_history)
            app.bind_operator(reopened, UserInputProxy("input", app, reopened, print), root)
            message = next(row for row in reopened.all_features if row.content == "Before the journal.")
            delivery = app.runtime_control("/control/delivery", {"delivery_id": "legacy-input"})  # pm-lint: allow PM005 -- HTTP delivery lookup
            self.assertEqual(delivery, DeliveryState(present=True, accepted=False, record_id=message.id))
            trace = app.runtime_control(f"/control/trace/{message.id}", {})  # pm-lint: allow PM005 -- empty HTTP trace request
            self.assertEqual(trace, TraceReceipt(record=message.as_payload(include_embedding=False), transaction=None,
                calls=[], call_linkage="unavailable: contributor gateway IDs are not supplied"))
