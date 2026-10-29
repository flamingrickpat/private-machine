"""Message isolation through the real bus, without module threads."""
import unittest

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from tests.unit.pm.host.test_bus_module import _EchoModule


class EnvelopeDeserializationTests(unittest.TestCase):
    def test_omitted_optional_fields_keep_constructor_defaults(self) -> None:
        encoded = ('{"id":"id","source":"sender","kind":"data","topic":"ping",'
                   '"correlation_id":"corr","payload":{}}')
        envelope = Envelope.from_json(encoded)
        self.assertIsNone(envelope.target)
        self.assertIsNone(envelope.causation_id)

    def test_unknown_fields_raise_instead_of_disappearing(self) -> None:
        encoded = ('{"id":"id","source":"sender","kind":"data","topic":"ping",'
                   '"correlation_id":"corr","payload":{},"target":null,"causation_id":null,"unexpected":1}')
        with self.assertRaisesRegex(TypeError, "unexpected"):
            Envelope.from_json(encoded)

    def test_missing_required_field_raises_constructor_error(self) -> None:
        encoded = ('{"id":"id","source":"sender","kind":"data","topic":"ping",'
                   '"correlation_id":"corr"}')
        with self.assertRaisesRegex(TypeError, "payload"):
            Envelope.from_json(encoded)


class BusIsolationTests(unittest.TestCase):
    def test_observer_receives_an_isolated_copy_outside_the_lock(self):
        bus = BusSystem()
        received, observed = [], []
        bus.subscribe("receiver", lambda envelope: True, received.append)

        def observer(envelope, recipient):
            observed.append((recipient, bus._lock._is_owned()))
            envelope.payload["items"].append(2)

        bus.observer = observer
        bus.put(Envelope("id", "sender", "data", "ping", "corr", {"items": [1]}))
        self.assertEqual(observed, [("receiver", False)])
        self.assertEqual(received[0].payload, {"items": [1]})

    def test_observation_error_prevents_unrecorded_delivery(self):
        bus = BusSystem()
        received = []
        bus.subscribe("receiver", lambda envelope: True, received.append)

        def observer(envelope, recipient):
            raise OSError("Trace storage failed")

        bus.observer = observer
        with self.assertRaisesRegex(OSError, "Trace storage failed"):
            bus.put(Envelope("id", "sender", "data", "ping", "corr", {}))
        self.assertEqual(received, [])

    def test_sender_and_subscribers_have_independent_nested_payloads(self):
        bus = BusSystem()
        received = []
        bus.subscribe("a", lambda envelope: True, received.append)
        bus.subscribe("b", lambda envelope: True, received.append)
        message = Envelope("id", "sender", "data", "ping", "corr", {"items": [{"value": 1}]})
        bus.put(message)
        received[0].payload["items"][0]["value"] = 2
        message.payload["items"].append({"value": 3})
        self.assertEqual(received[1].payload, {"items": [{"value": 1}]})
        self.assertIsNot(received[0], received[1])
        self.assertEqual(received[0].id, received[1].id)

    def test_predicate_mutation_does_not_change_another_delivery(self):
        bus = BusSystem()
        def predicate(envelope):
            envelope.payload["changed"] = True
            return False
        received = []
        bus.subscribe("first", predicate, lambda envelope: None)
        bus.subscribe("second", lambda envelope: True, received.append)
        bus.put(Envelope("id", "sender", "data", "ping", "corr", {}))
        self.assertEqual(received[0].payload, {})

    def test_callbacks_do_not_hold_the_subscription_lock(self):
        bus = BusSystem()
        ownership = []
        bus.subscribe("receiver", lambda envelope: True, lambda envelope: ownership.append(bus._lock._is_owned()))
        bus.put(Envelope("id", "sender", "data", "ping", "corr", {}))
        self.assertEqual(ownership, [False])

    def test_only_serialized_payload_values_cross_the_bus(self):
        with self.assertRaises(TypeError):
            BusSystem().put(Envelope("id", "sender", "data", "ping", "corr", {"live_object": object()}))

    def test_targeted_data_does_not_reach_another_module(self):
        module = _EchoModule(BusSystem())
        self.assertFalse(module.accepts(Envelope("id", "sender", "data", "ping", "corr", {}, target="other")))
        self.assertTrue(module.accepts(Envelope("id", "sender", "data", "ping", "corr", {}, target="echo")))

    def test_causation_survives_delivery(self):
        bus = BusSystem()
        received = []
        bus.subscribe("receiver", lambda envelope: True, received.append)
        bus.put(Envelope("id", "sender", "data", "ping", "corr", {}, causation_id="parent"))
        self.assertEqual(received[0].causation_id, "parent")
