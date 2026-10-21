"""Check rendering rules without changing durable source data."""

from pm.records.copying import pickle_copy
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic import BaseModel

from pm.llm.object_rendering import render_object_for_llm


ROOT = Path(__file__).resolve().parents[4]
SCHEMA = ROOT / "workflows/minecraft_project/resources/llm_context.json"


class ObjectRenderingTests(unittest.TestCase):
    def test_minecraft_precision_omissions_and_table_values(self):
        source = {"original_digest": "a" * 64, "position": {"x": 1.234567, "y": 105, "z": -2.876543},
                  "time": 1791977370.2088978, "radius": 0.125, "reason": "not_found",
                  "properties": {}, "collision_shapes": [], "verified": False,
                  "inventory": [{"name": "birch_planks", "count": n} for n in range(4)],
                  "required_certificates": {"gather": "b" * 64}, "unknown": "keep this field"}
        original = pickle_copy(source)
        result = json.loads(render_object_for_llm(source, SCHEMA).split("\n", 1)[1])
        self.assertNotIn("original_digest", result)
        self.assertNotIn("properties", result)
        self.assertEqual(result["position"], {"x": 1.2, "y": 105, "z": -2.9})
        self.assertEqual(result["time"], 1791977370)
        self.assertEqual(result["radius"], 0.125)
        self.assertEqual(result["reason"], "not_found")
        self.assertEqual(result["collision_shapes"], [])
        self.assertIs(result["verified"], False)
        self.assertEqual(result["required_certificates"], ["gather"])
        table = result["inventory"]
        self.assertEqual([dict(zip(table["table_fields"], row)) for row in table["table_rows"]], source["inventory"])
        self.assertEqual(result["unknown"], source["unknown"])
        self.assertEqual(source, original)

    def test_other_callers_select_their_own_rules_and_input_types(self):
        @dataclass
        class Record:
            value: float

        class Model(BaseModel):
            value: float

        with TemporaryDirectory() as folder:
            path = Path(folder) / "rules.json"
            path.write_text('{"version":1}', encoding="utf-8")
            source = [Record(1.234567), Model(value=2.345678)]
            self.assertEqual(json.loads(render_object_for_llm(source, path)),
                             [{"value": 1.234567}, {"value": 2.345678}])
            path.write_text('{"version":1,"float_digits":2}', encoding="utf-8")
            self.assertEqual(json.loads(render_object_for_llm(source, path)), [{"value": 1.23}, {"value": 2.35}])
            with self.assertRaises(TypeError):
                render_object_for_llm(object(), path)

    def test_mixed_and_empty_lists_keep_their_shape(self):
        source = {"reads": [{"ok": True}, {"error": "offline"}], "positions": [], "inventory": []}
        self.assertEqual(json.loads(render_object_for_llm(source, SCHEMA).split("\n", 1)[1]), source)

