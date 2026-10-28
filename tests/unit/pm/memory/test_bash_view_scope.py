"""Direct debugger checks for the real coder tools on a MemoryStore projection."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml
from pm_coder import build_settings, make_file_tools, prompt_text

from pm.memory.bash_view import MemoryBashEnvironment


class AgentFileScopeTests(unittest.TestCase):
    def test_virtual_record_files_render_without_source_revision_maps(self) -> None:
        from pm.records.memory_records import Feature
        from pm.records.enums import FeatureType
        record = Feature(id=1, content="Original source content.", feature_type=FeatureType.Dialogue,
                         metadata={"generation_source_revisions": {str(key): 1 for key in range(1000)}})
        environment = MemoryBashEnvironment({1: record}, {}, "read")
        text = environment.read_text("ghost/knoxels/1.yaml")
        rendered = json.loads(text)
        self.assertEqual(rendered["data"]["content"], record.content)
        self.assertNotIn("generation_source_revisions", text)
        self.assertEqual(len(environment.records[1].metadata["generation_source_revisions"]), 1000)

    def test_agent_tools_cannot_change_projected_memory(self):
        from pm.records.memory_records import Feature
        from pm.records.enums import FeatureType
        record = Feature(id=1, content="accepted memory", feature_type=FeatureType.Dialogue).as_payload(include_embedding=False)
        environment = MemoryBashEnvironment({1: record}, {}, "read")
        with TemporaryDirectory() as folder:
            settings = build_settings(cwd=folder, model="unused", context_window=96_000)
            tools = {tool.name: tool.function for tool in make_file_tools(settings, environment.machine, user="workflow")}
            self.assertIn("accepted memory", tools["read"]("ghost/knoxels/1.yaml"))
            attempts = [
                tools["write"]("ghost/knoxels/1.yaml", "changed"),
                tools["write"]("ghost/knoxels/1.yaml", "changed", 1, 1),
                tools["edit"]("ghost/knoxels/1.yaml", "accepted memory", "changed"),
            ]
            for attempt in attempts:
                self.assertFalse(json.loads(attempt)["success"])
            rendered = yaml.safe_load(environment.read_text("ghost/knoxels/1.yaml"))
            self.assertEqual(rendered["__type__"], record["__type__"])
            self.assertEqual(rendered["data"]["content"], "accepted memory")
            self.assertNotIn("embedding", rendered["data"])
            self.assertEqual(environment.visible_knoxels[1], record)

    def test_agent_tools_use_virtual_cwd_without_host_fallback(self):
        environment = MemoryBashEnvironment({}, {}, "read")
        with TemporaryDirectory() as folder:
            host_file = Path(folder) / "host-only.txt"
            host_file.write_text("host-only content", encoding="utf-8")
            settings = build_settings(cwd=folder, model="unused", context_window=96_000)
            tools = {tool.name: tool.function for tool in make_file_tools(settings, environment.machine, user="workflow")}
            environment.write_text("inbox/request.md", "virtual request")
            self.assertEqual(prompt_text("inbox/request.md", Path(folder), bash_machine=environment.machine, user="workflow"), "virtual request")
            self.assertFalse(json.loads(tools["read"](str(host_file)))["success"])
            self.assertEqual(environment.exec("cd scratch").exit_code, 0)
            self.assertIn("whole file", tools["write"]("result.txt", "agent result"))
            self.assertEqual(environment.read_text("scratch/result.txt"), "agent result\n")
            self.assertFalse((Path(folder) / "result.txt").exists())


if __name__ == "__main__":
    unittest.main()
