"""Check local configuration repair and template rendering without starting any services."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.paths import REPO_ROOT
from simulation.baseline_minecraft import MinecraftSetup
from simulation.local_world import LocalWorld, SetupError, seed_file
from simulation.simulation_launcher import launch


class LauncherSetupTests(unittest.TestCase):
    def test_missing_installation_config_names_the_example(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "config_simulation.yaml"
            with self.assertRaisesRegex(SetupError, "Copy config_simulation.yaml.example"):
                launch(path, Path("unused"), "probe", True, False)
            self.assertFalse(path.exists())

    def test_rerun_preserves_user_template_and_rejects_a_shared_world(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            suite = REPO_ROOT / "test_data/simulation/baseline-v1"
            layout = MinecraftSetup.load(suite / "minecraft.json").worlds[0]
            world = LocalWorld(root, root / "launch", root / "fabric", root / "mcp", root / "install.yaml", layout)
            template = root / "server.properties"
            seed_file("server.properties.example", template)
            rendered = world.properties(31001, 31002, "example")
            self.assertIn("server-port=31001\n", rendered)
            self.assertIn("rcon.port=31002\n", rendered)
            self.assertIn("level-name=simulation-world\n", rendered)
            template.write_text(template.read_text().replace("simulation-world", "../existing-world"), encoding="utf-8")
            seed_file("server.properties.example", template)
            with self.assertRaisesRegex(SetupError, "Restore the level-name setting"):
                world.properties(31001, 31002, "example")
