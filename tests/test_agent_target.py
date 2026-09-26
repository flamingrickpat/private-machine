"""Application targets supply the actual tool environment to agent roles."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from working_target import FolderTarget, GhostBashEnvironment


class AgentTargetTests(unittest.TestCase):
    def test_virtual_binding_preserves_user_and_machine(self):
        environment = GhostBashEnvironment({}, {}, "read")
        target = environment.agent_target(Path("artifacts"))
        self.assertIs(target.bash_machine, environment.machine)
        self.assertEqual(target.user, "workflow")
        environment.write_text("scratch/marker.txt", "virtual")
        self.assertEqual(target.bash_machine.read_binary_as(target.user, "scratch/marker.txt"), b"virtual")

    def test_folder_binding_uses_target_instead_of_journal_workspace(self):
        with TemporaryDirectory() as root:
            environment = FolderTarget(Path(root), "bash").open("run")
            target = environment.agent_target(Path(root) / "journal")
            self.assertEqual(target.cwd, environment.root.resolve())
            self.assertIsNone(target.bash_machine)
