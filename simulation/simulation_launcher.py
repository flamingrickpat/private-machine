"""Launch one versioned scenario with a private character seed and optional local Minecraft services."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import yaml

from pm.host.durable_files import ProcessLock
from pm.llm.config import load_config
from pm.memory.store import MemoryStore
from pm.paths import REPO_ROOT
from pm.utils.operating_system import OperatingSystem
from simulation.baseline import BaselineCampaign, Scenario
from simulation.baseline_minecraft import MinecraftSetup
from simulation.local_world import LocalWorld, Settings, SetupError
from simulation.personas import CANON, DEFAULT_RUNS, canon_home, write_home


class Installation(Settings):
    model_config_path: Path
    base_checkpoint: Path | None
    mcp_checkout: Path
    server_installation: Path
    require_capped_openrouter_key: bool = True


def model_settings(source: Path, destination: Path) -> None:
    """Keep credentials local and enable trace capture in the run's private model configuration."""
    if not source.is_file():
        raise SetupError(f"Create {source} from {REPO_ROOT / 'config.example.yaml'}. Set chat and embedding endpoints there.")
    try:
        settings = load_config(source)
        for connection in settings.chat.values():
            connection.resolve()
        settings.embedding.resolve()
    except (ValueError, KeyError) as error:
        raise SetupError(f"Invalid model endpoint settings in {source}. Check the chat, embedding and API key settings.") from error
    # Preserve original key references and provider-specific fields in the on-disk model schema.
    document = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    document["profile_cognition"] = True
    for name, connection in settings.chat.items():
        if connection.startup_script is not None:
            document["chat"][name]["startup_script"] = str(connection.startup_script)
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def prepare_seed(checkpoint: Path | None, destination: Path) -> Path:
    """Use a staged home so an interruption cannot leave a partially written seed."""
    if checkpoint is not None and checkpoint.name == "ghost.pickle":
        home = checkpoint.parent
        if (home / "character.yaml").is_file() and (home / "personality.json").is_file():
            return home
    with TemporaryDirectory(dir=destination.parent) as temporary:
        home = Path(temporary) / "seed"
        if checkpoint is None:
            canon_home(home)
        else:
            write_home(MemoryStore.from_checkpoint(checkpoint), home)
        # This seed belongs to the exact launch, including retries after a setup error.
        destination.mkdir(exist_ok=True)
        for name in ("ghost.pickle", "character.yaml", "personality.json"):
            (home / name).replace(destination / name)
    return destination


def launch(config: Path, sim_config: Path, run_id: str, preflight: bool, check_services: bool) -> None:
    config = config.resolve()
    if not config.is_file():
        raise SetupError(f"Missing {config}. Copy config_simulation.yaml.example to {config}. "
                         "Set model_config_path to your PM endpoint configuration and run the same command again.")
    installation = Installation.read(config)
    suite = REPO_ROOT / "test_data/simulation/baseline-v1"
    try:
        scenario = Scenario(sim_config.stem)
    except ValueError as error:
        raise SetupError(f"Unknown scenario {sim_config}. Choose a numbered launcher in {REPO_ROOT / 'simulation'}.") from error
    if sim_config.resolve() != (suite / f"{scenario}.json").resolve():
        raise SetupError(f"Select one of the four versioned scenario files in {suite}.")
    campaign = BaselineCampaign(run_id, config, scenario)
    run = campaign.runs[0]
    root = DEFAULT_RUNS / "instances" / run.root.name
    with ProcessLock(root / "launcher.lock"):
        if run.root.exists() and not preflight and not check_services:
            raise SetupError(f"Run {run.root.name} already exists. Choose a new -RunId to start another simulation.")
        source = (config.parent / installation.model_config_path.expanduser()).resolve()
        campaign.config = root / "model-config.yaml"
        model_settings(source, campaign.config)
        checkpoint = installation.base_checkpoint
        if checkpoint is not None:
            checkpoint = (config.parent / checkpoint.expanduser()).resolve()
        if not (checkpoint or CANON).is_file():
            raise SetupError(f"Missing checkpoint {checkpoint or CANON}. Set base_checkpoint in {config}, "
                             "or set it to null to use the shipped Spamuel canon.")
        layout = MinecraftSetup.load(suite / "minecraft.json")
        setup = next((world for world in layout.worlds if world.scenario == scenario.value), None)
        world = None
        if setup is not None:
            purpose = "service-checks" if check_services else "launches"
            work_root = root / purpose / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            world = LocalWorld(root, work_root, (config.parent / installation.server_installation.expanduser()).resolve(),
                               (config.parent / installation.mcp_checkout.expanduser()).resolve(), config, setup)
            world.validate()
        print(f"{run.root.name}\nScenario: {sim_config}\nLocal settings and service logs: {root}\nPM evidence: {run.root}", flush=True)
        if preflight:
            print("Local preflight passed. Start without -Preflight to launch. Paid endpoints were not contacted.")
            return
        if check_services and world is not None:
            campaign.check_capacity(0)
        if not check_services:
            campaign.check_capacity(len(run.template.participants))
            if installation.require_capped_openrouter_key:
                try:
                    campaign.check_budget()
                except ValueError as error:
                    raise SetupError(f"{error} Check the key in {source} and require_capped_openrouter_key in {config}.") from error
        if (world is not None or not check_services) and not OperatingSystem.install_child_cleanup():
            raise SetupError("Windows could not attach the child processes to a cleanup job. Run the launcher in PowerShell.")
        try:
            if world is not None:
                print(f"Starting the private Fabric world and Minecraft bodies in {world.work_root}", flush=True)
                world.start()
                if not check_services:
                    from simulation.current import append_current
                    for body in world.layout.bodies:
                        append_current(f"{run.root.name} · {body.username} · Minecraft {body.server} · MCP {body.endpoint} · services in {world.work_root}")
                for body in world.layout.bodies:
                    member = next(member for member in run.template.participants if member.id == body.participant)
                    member.minecraft_mcp_url = body.endpoint
            if check_services:
                campaign.check_bodies([run], layout)
                print("Service check passed. Closing the check world. No simulation started.")
                return
            seed = prepare_seed(checkpoint, root / "seed")
            for member in run.template.participants:
                member.home = str(seed)
            from simulation.current import append_current
            append_current(f"{run.root.name} · starting scenario · evidence in {run.root}")
            campaign.launch(resume=False, layout=layout, require_capped_key=False, services=world)
            append_current(f"{run.root.name} · completed")
        finally:
            if world is not None:
                world.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sim_config", type=Path, required=True)
    parser.add_argument("--run-id", default="local")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-services", action="store_true")
    arguments = parser.parse_args()
    try:
        launch(arguments.config, arguments.sim_config, arguments.run_id, arguments.preflight, arguments.check_services)
    except SetupError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
