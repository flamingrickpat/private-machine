"""Startup check for the configured Minecraft body endpoint."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from pm.runs.errors import TaskServiceError
from workflows.minecraft_body import mcp_client


class ProbeBodyTests(unittest.TestCase):
    def probe(self, names):
        tools = {"minecraft_list_tools": lambda: names}
        with patch.object(mcp_client, "build_minecraft_tools", return_value=tools):
            return mcp_client.probe_body("http://127.0.0.1:6767/mcp")

    def test_a_body_offers_the_direct_observe_tool(self):
        # The remote server exposes real action names. "minecraft_call" is
        # only the local wrapper, so it must not be the startup requirement.
        names = ["minecraft_observe", "minecraft_mine_block"]
        self.assertEqual(self.probe(names), names)

    def test_another_mcp_server_does_not_pass(self):
        with self.assertRaisesRegex(TaskServiceError, "minecraft_observe"):
            self.probe(["unrelated_tool"])


if __name__ == "__main__":
    unittest.main()
