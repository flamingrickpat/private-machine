"""Launch the four versioned companion scenarios against independent character homes."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
import time
from typing import Literal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.baseline_minecraft import MinecraftSetup
    from simulation.local_world import LocalWorld

import httpx
from pydantic import BaseModel

from pm.host.durable_files import ProcessLock, runtime_status
from pm.llm.config import load_config
from pm.paths import REPO_ROOT
from pm.utils.windows_commit import CommitHeadroom
from simulation.engine import RunEngine
from simulation.personas import DEFAULT_RUNS
from simulation.templates import SimulationTemplate, create_run


class Scenario(StrEnum):
    Chat = "jesse-spamuel"
    Group = "jesse-two-spamuels"
    Multiplayer = "two-spamuels-minecraft"
    Coplay = "jesse-spamuel-minecraft"


class SuiteManifest(BaseModel):
    version: Literal[1]
    reserve_gib: int
    participant_gib: int
    budget_usd: Literal[100]
    scenarios: list[Scenario]


class KeyBudget(BaseModel):
    limit: float | None
    limit_remaining: float | None
    limit_reset: str | None


class KeyStatus(BaseModel):
    data: KeyBudget


class RunProgress(BaseModel):
    status: str
    attempt: int
    step_index: int
    clock: str


class PublicDelivery(BaseModel):
    kind: str
    target: str = ""
    source: str = ""


class CampaignProgress(RunProgress):
    timeline: list[PublicDelivery]

    def has_exchange(self) -> bool:
        waiting: set[str] = set()
        for delivery in self.timeline:
            if delivery.kind == "actor":
                waiting.add(delivery.target)
            if delivery.kind == "speech" and delivery.source in waiting:
                return True
        return False


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    scenario: Scenario
    template: SimulationTemplate
    root: Path

    def progress(self) -> RunProgress:
        saved = RunProgress.model_validate_json((self.root / "state.json").read_bytes())
        saved.status = runtime_status(self.root, saved.status)
        return saved


class BaselineCampaign:
    """One campaign ID fixes the run folders used for status and restart."""

    def __init__(self, campaign_id: str, config: Path, selected: Scenario | None) -> None:
        if re.fullmatch(r"[A-Za-z0-9_-]+", campaign_id) is None:
            raise ValueError("Campaign ID accepts letters, numbers, underscores and hyphens.")
        self.campaign_id: str = campaign_id
        self.config: Path = config.resolve()
        self.runs: list[ScenarioRun] = []
        suite = REPO_ROOT / "test_data" / "simulation" / "baseline-v1"
        self.manifest: SuiteManifest = SuiteManifest.model_validate_json((suite / "suite.json").read_bytes())
        if len(self.manifest.scenarios) != len(Scenario) or set(self.manifest.scenarios) != set(Scenario):
            raise ValueError("The baseline manifest must include each scenario once.")
        for scenario in self.manifest.scenarios:
            if selected is not None and scenario != selected:
                continue
            template_path = suite / f"{scenario}.json"
            template = SimulationTemplate.model_validate_json(template_path.read_bytes())
            run_root = DEFAULT_RUNS / "runs" / f"{campaign_id}-{scenario}"
            self.runs.append(ScenarioRun(scenario, template, run_root))

    def check_inputs(self) -> None:
        endpoints: set[str] = set()
        for run in self.runs:
            plan = run.template.plan
            if plan is None or len({step.at.date() for step in plan.steps}) != 3:
                raise ValueError(f"{run.scenario} needs a fixed three-day plan.")
            for participant in run.template.participants:
                home = Path(participant.home).expanduser()
                for filename in ("character.yaml", "personality.json", "ghost.pickle"):
                    if not (home / filename).is_file():
                        raise FileNotFoundError(home / filename)
                endpoint = participant.minecraft_mcp_url
                if endpoint is not None:
                    if endpoint in endpoints:
                        raise ValueError(f"Two baseline bodies share {endpoint}.")
                    endpoints.add(endpoint)
            print(f"{run.scenario}: {len(plan.steps)} stages; {len(run.template.participants)} participant(s)")

    def check_bodies(self, runs: list[ScenarioRun], layout: MinecraftSetup | None = None) -> None:
        """Connect to each declared Minecraft MCP and verify its observation capability."""
        from simulation.baseline_minecraft import BodyPreflight, MinecraftSetup, prepare_body
        if layout is None:
            layout = MinecraftSetup.load(REPO_ROOT / "test_data/simulation/baseline-v1/minecraft.json")
        probes: list[BodyPreflight] = []
        for run in runs:
            for participant in run.template.participants:
                if participant.minecraft_mcp_url is None:
                    continue
                matches = [body for world in layout.worlds if world.scenario == run.scenario
                           for body in world.bodies if body.participant == participant.id]
                if len(matches) != 1 or matches[0].endpoint != participant.minecraft_mcp_url:
                    raise ValueError(f"Invalid body mapping for {run.scenario}/{participant.id}.")
                probe = prepare_body(matches[0])
                identity = probe.check_identity()
                print(f"{run.scenario}: {identity.username} on {identity.server}, MCP {identity.schema_version}")
                probes.append(probe)
        for probe in probes:
            probe.check_motion()

    def check_capacity(self, participants_to_start: int) -> None:
        required = self.manifest.reserve_gib + self.manifest.participant_gib * participants_to_start
        CommitHeadroom.measure().require(required)

    def check_budget(self) -> None:
        """Verify a single non-resetting OpenRouter limit across all configured inference routes."""
        config = load_config(self.config)
        if not config.profile_cognition:
            raise ValueError("Enable profile_cognition in the campaign config.")
        connections = [preset.resolve() for preset in config.chat.values()]
        connections.append(config.embedding.resolve())
        shared_key = connections[0].api_key
        for connection in connections:
            if connection.base_url.rstrip("/") != "https://openrouter.ai/api/v1" or connection.api_key != shared_key:
                raise ValueError("Use the same capped OpenRouter key for all campaign model routes.")
        authorization = "Bearer " + shared_key.get_secret_value()
        response = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": authorization})  # pm-lint: allow PM005 -- HTTP authorization header at provider boundary
        response.raise_for_status()
        budget = KeyStatus.model_validate_json(response.content).data
        if budget.limit is None or budget.limit > self.manifest.budget_usd or budget.limit_reset is not None:
            raise ValueError("The campaign requires a non-resetting OpenRouter key capped at $100 or less.")
        if budget.limit_remaining is None or budget.limit_remaining <= 0:
            raise ValueError("The campaign key has no remaining credit.")
        print(f"Provider key has ${budget.limit_remaining:.2f} remaining.")

    def measure_first_exchange(self, engine: RunEngine) -> None:
        """Run this scenario alone until its first answered delivery or completed stage, then drain."""
        progress = CampaignProgress.model_validate(engine.view())
        if progress.has_exchange() or progress.status == "completed":
            return
        self.check_capacity(len(engine.state["participants"]))
        engine.start()
        thread = engine.thread
        if thread is None:
            raise RuntimeError(f"No controller thread for {engine.root}")
        try:
            while thread.is_alive():
                self.check_capacity(0)
                progress = CampaignProgress.model_validate(engine.view())
                if progress.has_exchange():
                    if progress.status == "running":
                        engine.pause()
                    break
                time.sleep(1)
        except BaseException:
            if thread.is_alive() and engine.view()["status"] in {"starting", "running"}:
                engine.pause()
            raise
        finally:
            thread.join()
        result = RunProgress.model_validate(engine.view())
        if result.status not in {"paused", "completed"}:
            raise RuntimeError(f"Isolated exchange ended {result.status}: {engine.root}")

    def launch(self, resume: bool, layout: MinecraftSetup | None = None, require_capped_key: bool = True,
               services: LocalWorld | None = None) -> None:
        self.check_inputs()
        engines: list[RunEngine] = []
        locks = ExitStack()
        try:
            for run in self.runs:
                locks.enter_context(ProcessLock(DEFAULT_RUNS / "campaigns" / self.campaign_id / f"{run.scenario}.lock"))
            pending: list[ScenarioRun] = []
            for run in self.runs:
                if run.root.exists():
                    if not resume:
                        raise FileExistsError(run.root)
                    status = run.progress().status
                    if status == "completed":
                        continue
                    if status in {"starting", "running", "pausing", "stopping"}:
                        raise RuntimeError(f"{run.root} already has an active controller ({status}).")
                pending.append(run)
            if not pending:
                return
            self.check_capacity(sum(len(run.template.participants) for run in pending))
            if require_capped_key:
                self.check_budget()
            self.check_bodies(pending, layout)
            for run in pending:
                if run.root.exists():
                    engine = RunEngine(run.root)
                    if Path(str(engine.state["config"])).resolve() != self.config:
                        raise ValueError(f"Resume with the original config for {run.scenario}.")
                else:
                    engine = create_run(run.template, run.root, str(self.config))
                    if services is not None:
                        services.record_run_setup(run.root)
                engines.append(engine)
            if len(engines) > 1:
                for engine in engines:
                    self.measure_first_exchange(engine)
            ready = [engine for engine in engines if engine.view()["status"] != "completed"]
            self.check_capacity(sum(len(engine.state["participants"]) for engine in ready))
            for engine in ready:
                engine.start()
            while any(engine.thread is not None and engine.thread.is_alive() for engine in engines):
                if services is not None:
                    services.check_processes()
                self.check_capacity(0)
                time.sleep(2)
            for engine in engines:
                if engine.thread is None:
                    raise RuntimeError(f"Missing controller thread for {engine.root}.")
                engine.thread.join()
                if engine.view()["status"] != "completed":
                    raise RuntimeError(f"Inspect {engine.root}: {engine.view()['status']}")
        finally:
            for engine in engines:
                if engine.thread is not None and engine.thread.is_alive():
                    if engine.view()["status"] in {"starting", "running"}:
                        engine.pause()
            for engine in engines:
                if engine.thread is not None:
                    engine.thread.join()
            locks.close()

    def export(self, destination: Path) -> None:
        from simulation.baseline_bundle import write_bundle
        roots: list[Path] = []
        for run in self.runs:
            progress = run.progress()
            if progress.status not in {"completed", "failed", "paused", "stopped", "crashed"}:
                raise ValueError(f"Quiesce {run.scenario} before export: {progress.status}")
            roots.append(run.root)
        write_bundle(roots, destination)

    def status(self) -> None:
        for run in self.runs:
            if not run.root.exists():
                print(f"{run.scenario}: awaiting launch")
                continue
            progress = run.progress()
            print(f"{run.scenario}: {progress.status}, attempt {progress.attempt}, world {progress.clock}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["validate", "preflight", "launch", "resume", "status", "metrics", "export", "unpack"])
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--scenario", choices=list(Scenario))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    scenario = Scenario(args.scenario) if args.scenario else None
    campaign = BaselineCampaign(args.campaign, args.config, scenario)
    match args.operation:
        case "export":
            destination = args.bundle or REPO_ROOT / "test_data" / "baselines" / f"{args.campaign}.zip"
            campaign.export(destination)
            print(destination)
        case "unpack":
            if args.bundle is None or args.destination is None:
                parser.error("unpack needs --bundle and --destination")
            from simulation.baseline_bundle import unpack_bundle
            unpack_bundle(args.bundle, args.destination)
        case "validate":
            campaign.check_inputs()
        case "preflight":
            campaign.check_inputs()
            campaign.check_capacity(sum(len(run.template.participants) for run in campaign.runs))
            campaign.check_bodies(campaign.runs)
            campaign.check_budget()
        case "status":
            campaign.status()
        case "metrics":
            from simulation.baseline_metrics import measure_run
            for run in campaign.runs:
                print(run.scenario, measure_run(run.root).model_dump_json(indent=2))  # pm-lint: allow PM007 -- CLI output
        case "launch":
            campaign.launch(resume=False)
        case "resume":
            campaign.launch(resume=True)


if __name__ == "__main__":
    main()
