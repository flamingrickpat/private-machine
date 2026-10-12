"""Check actual death metadata with a separate empty real-server character."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(args) -> None:
    """Retain real protocol evidence without claiming native model acceptance.

    The fixture kills only its separate character, after checking empty
    inventory. It preserves the world and stops the services that it owns.
    Use a fresh ignored output directory. Any failed check retains its receipt.
    """
    from pm.host.durable_files import atomic_json
    from runtime.minecraft_outcomes import GameplayEvidence
    from runtime.minecraft_tools import build_minecraft_tools
    from dev.work_items.alpha_minecraft_gate import RealServerStack
    from tests.live.minecraft_stack import load_env_file

    out, server = args.out.resolve(), args.server.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep real lifecycle evidence under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    properties = load_env_file(server / "server.properties")
    values = load_env_file(ROOT.parent / "pm-minecraft/.env_real")
    values.update({"MINECRAFT_PORT": properties["server-port"], "MINECRAFT_RCON_PORT": properties["rcon.port"],
        "MINECRAFT_RCON_PASSWORD": properties["rcon.password"], "MINECRAFT_PLAYER": "a13_life_probe",
        "MCMCP_TEST_MODE": "true", "MCMCP_AGENT_HOME": str(out / "character"), "MCMCP_INSPECTOR_ENABLED": "false",
        "JAVA_EXE": "C:/Users/rick/AppData/Roaming/PrismLauncher/java/java-runtime-gamma/bin/java.exe",
        "PYTHON_EXE": str(ROOT.parent / "pm-minecraft/.venv/Scripts/python.exe"),
        "TEST_RUN_ROOT": str(out), "SERVER_MEMORY": "1G", "SERVER_START_TIMEOUT_SECONDS": "180"})
    for key in ("MINECRAFT_PORT", "MINECRAFT_RCON_PORT", "MCP_PORT", "MCMCP_BODY_PORT", "MCMCP_VIEWER_PORT"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(values[key])))
    env_path = out / ".env.alpha"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    receipt = {"status": "running", "scope": "real protocol lifecycle boundary, no native model acceptance",
        "reset_world": False, "item_grants": False, "player": values["MINECRAFT_PLAYER"], "rows": [], "checks": {},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    evidence, stack = GameplayEvidence(), RealServerStack(server, out, env_path)
    source = str(out / "receipt.json")

    def save():
        atomic_json(out / "receipt.json", receipt)

    def call(name, arguments):
        observed = tools["minecraft_call"](name, arguments)
        row = {"event": "result", "seq": len(receipt["rows"]) + 1, "time": time.time(), "tool": name,
            "arguments": arguments, "ok": observed["ok"], "result": observed["result"]}
        receipt["rows"].append(row)
        evidence.consume(row, source)
        save()
        return row["result"]

    try:
        save()
        stack.start()
        tools = build_minecraft_tools(stack.mcp_url)
        evidence.consume({"event": "epoch_acquired", "time": time.time()}, source)
        before = call("minecraft_observe", {"include_image": False})
        if before["inventory"] or before["death_count"] != 0 or not before["body_session"]:
            raise ValueError("The lifecycle fixture requires an empty character and a fresh body counter.")
        code = ("import sys; from pathlib import Path; from mcrcon import MCRcon; "
            "v=dict(l.split('=',1) for l in Path(sys.argv[1]).read_text().splitlines() if '=' in l); "
            "r=MCRcon('127.0.0.1',v['MINECRAFT_RCON_PASSWORD'],port=int(v['MINECRAFT_RCON_PORT'])); "
            "r.connect(); print(r.command('kill a13_life_probe')); r.disconnect()")
        killed = subprocess.run([values["PYTHON_EXE"], "-c", code, str(env_path)],
            capture_output=True, text=True, check=True)
        receipt["fixture_command"] = {"command": "kill a13_life_probe", "result": killed.stdout, "time": time.time()}
        deadline = time.monotonic() + 30
        while True:
            info = tools["minecraft_call"]("minecraft_info", {})["result"]
            if info["death_count"] == 1 and info["spawned"]:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("The actual respawn did not finish.")
            time.sleep(0.25)
        after = call("minecraft_observe", {"include_image": False})
        body = after["camera"]["block_position"]
        for offset in (-1, 0, 1):
            call("minecraft_inspect_block", {"position": {**body, "y": body["y"] + offset}, "include_image": False})
        call("minecraft_observe", {"include_image": False})
        call("minecraft_observe", {"include_image": False})
        blocked = evidence.guard("minecraft_craft_item", {"item": "stick"})
        receipt["checks"] = {"actual_death_counter_increased": after["death_count"] == 1,
            "same_body_session": before["body_session"] == after["body_session"],
            "restored_vitals_do_not_clear_recovery": evidence.recovery is not None,
            "death_blocks_optional_mutation": blocked is not None and blocked["reason"] == "death_requires_replan",
            "failed_epoch_retained": evidence.current_failed_recovery() is not None}
        receipt["gameplay"] = evidence.snapshot()
        receipt["status"] = "passed" if all(receipt["checks"].values()) else "failed"
        if receipt["status"] != "passed":
            raise AssertionError("The actual death boundary failed a protocol check.")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stack.stop()
        body_log = out / "character/body.log"
        if body_log.exists():
            shutil.copy2(body_log, out / "services/body.log")
        receipt["finished"] = time.time()
        save()
        print(out / "receipt.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=Path("C:/source/minecraft-server-real"))
    run(parser.parse_args())
