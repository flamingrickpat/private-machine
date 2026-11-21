"""Own one four-scenario first pass from Windows Task Scheduler, including its shared spending limit."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import httpx
import yaml

from pm.host.durable_files import ProcessLock
from pm.paths import REPO_ROOT
from pm.utils.operating_system import OperatingSystem
from simulation.baseline import BaselineCampaign, KeyStatus
from simulation.current import append_current
from simulation.personas import DEFAULT_RUNS
from simulation.simulation_launcher import Installation


def wait_for_key(campaign: str, path: Path) -> str:
    last_status = ""
    while True:
        key = ""
        if path.is_file():
            try:
                key = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["api_key"]
            except (TypeError, KeyError, yaml.YAMLError):
                raise ValueError(f"Set api_key in {path}.") from None
            if not isinstance(key, str):
                raise ValueError(f"Set api_key to a string in {path}.")
        status = f"waiting for api_key in local config {path}"
        if key:
            try:
                response = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key})
                if response.is_success:
                    budget = KeyStatus.model_validate_json(response.content).data
                    if (budget.limit is not None and 0 < budget.limit <= 20 and budget.limit_reset is None
                            and budget.limit_remaining is not None and budget.limit_remaining > 0):
                        append_current(f"{campaign} · shared provider limit verified · ${budget.limit_remaining:.2f} remaining")
                        return key
                    status = "waiting for a funded provider key with a lifetime cap of $20 or less and no reset"
                else:
                    status = f"OpenRouter key metadata returned HTTP {response.status_code}; waiting for a valid key or service recovery"
            except httpx.TransportError:
                status = "waiting for OpenRouter key metadata service"
        if status != last_status:
            append_current(f"{campaign} · {status}")
            last_status = status
        time.sleep(30)


def campaign_settings(source: Path, root: Path, key: str) -> Path:
    install = Installation.read(source)
    models_path = (source.parent / install.model_config_path.expanduser()).resolve()
    models = yaml.safe_load(models_path.read_text(encoding="utf-8-sig"))
    for route in [*models["chat"].values(), models["embedding"]]:
        if route["base_url"].rstrip("/") != "https://openrouter.ai/api/v1":
            raise ValueError(f"Configure OpenRouter for each model route in {models_path}")
        route.pop("api_key_env", None)
        route["api_key"] = key
        if route.get("startup_script"):
            raise ValueError(f"Remove the local startup script from the OpenRouter route in {models_path}")
    model_copy = root / "endpoint.yaml"
    model_copy.write_text(yaml.safe_dump(models, sort_keys=False), encoding="utf-8")
    install.model_config_path = model_copy
    install.mcp_checkout = (source.parent / install.mcp_checkout.expanduser()).resolve()
    install.server_installation = (source.parent / install.server_installation.expanduser()).resolve()
    if install.base_checkpoint is not None:
        install.base_checkpoint = (source.parent / install.base_checkpoint.expanduser()).resolve()
    install.require_capped_openrouter_key = True
    output = root / "installation.yaml"
    output.write_text(install.model_dump_json(indent=2), encoding="utf-8")  # pm-lint: allow PM007 -- JSON is valid YAML
    return output


def run_campaign(campaign_id: str) -> None:
    source = REPO_ROOT / "config_simulation.yaml"
    suite = BaselineCampaign(campaign_id, source, None)
    root = DEFAULT_RUNS / "campaigns" / campaign_id
    with ProcessLock(root / "detached.lock"):
        if any(run.root.exists() for run in suite.runs):
            raise FileExistsError(f"Campaign already has evidence: {campaign_id}.")
        key = wait_for_key(campaign_id, root / "openrouter-key.yaml")
        config = campaign_settings(source, root, key)
        if not OperatingSystem.install_child_cleanup():
            raise RuntimeError("Windows could not attach the campaign to a cleanup job.")
        workers: dict[str, subprocess.Popen[bytes]] = {}
        for run in suite.runs:
            scenario = run.scenario.value
            template = REPO_ROOT / "test_data/simulation/baseline-v1" / f"{scenario}.json"
            log = root / f"{scenario}.log"
            with log.open("ab") as stream:
                worker = subprocess.Popen([sys.executable, "-u", "-m", "simulation.simulation_launcher",
                                           "--config", str(config), "--sim_config", str(template), "--run-id", campaign_id],
                                          cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW)
            workers[scenario] = worker
            append_current(f"{campaign_id} · {scenario} · launcher PID {worker.pid} · log {log}")
        failed = False
        while workers:
            for scenario, worker in list(workers.items()):
                result = worker.poll()
                if result is not None:
                    append_current(f"{campaign_id} · {scenario} · launcher exit {result} · log {root / (scenario + '.log')}")
                    failed |= result != 0
                    del workers[scenario]
            if workers:
                time.sleep(3)
        if failed:
            raise RuntimeError("One or more scenarios failed. See the scenario logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    args = parser.parse_args()
    try:
        run_campaign(args.campaign)
    except Exception as error:
        append_current(f"{args.campaign} · campaign stopped · {type(error).__name__}. Inspect the local campaign logs.")
        raise
