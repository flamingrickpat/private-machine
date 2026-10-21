"""Run an asynchronous public conversation with an independent Jesse model.

The full companion owns its private activity, speech, memory, and Minecraft.
Jesse receives only public history and a human situation. Evidence stays under
.runtime. A runtime error ends the experiment and retains its traceback.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.work_items.alpha_minecraft_gate import RealServerStack
from dev.work_items.shared.evidence import save_json, sha256_file
from dev.tools.alpha_snapshot import capture
from pm.context_render.turns import Role, Turn
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.memory.store import MemoryStore
from tests.live.minecraft_stack import load_env_file
from workflows.companion import build_app

SEED = ROOT / "test_data/personas/spamuel/canon.yaml"
TOPICS = [
    "You had an exhausting week and want a small shared Minecraft adventure. Ask what Spamuel can observe, then suggest gathering wood together.",
    "You are procrastinating on an unfinished creative project. Talk about it, then suggest a small Minecraft outing to get unstuck.",
    "You want to reconnect over an ordinary evening. Ask about the Minecraft surroundings and choose one modest activity together.",
]


def now() -> str:
    """Return an aware UTC timestamp for experiment events."""
    return datetime.now(timezone.utc).isoformat()


def start_minecraft(out: Path, reset_world: bool) -> RealServerStack:
    """Start the real server and MCP with an isolated character.

    Refuse occupied ports. Validate every reset path before the supplied
    PowerShell script removes world data. Credentials stay in the run directory.
    The caller must stop the returned stack in a finally block.
    """
    server = Path("C:/source/minecraft-server-real").resolve()
    properties = load_env_file(server / "server.properties")
    values = load_env_file(ROOT.parent / "pm-minecraft/.env_real")
    values.update({"MINECRAFT_PORT": properties["server-port"],
        "MINECRAFT_RCON_PORT": properties["rcon.port"], "MINECRAFT_RCON_PASSWORD": properties["rcon.password"],
        "MINECRAFT_PLAYER": "tdd_player", "MCMCP_TEST_MODE": "true",
        "MCMCP_AGENT_HOME": str(out / "character"), "MCMCP_INSPECTOR_ENABLED": "false",
        "JAVA_EXE": "C:/Users/rick/AppData/Roaming/PrismLauncher/java/java-runtime-gamma/bin/java.exe",
        "PYTHON_EXE": str(ROOT.parent / "pm-minecraft/.venv/Scripts/python.exe"),
        "TEST_RUN_ROOT": str(out), "SERVER_MEMORY": "2G", "SERVER_START_TIMEOUT_SECONDS": "180"})
    ports = [int(properties["server-port"]), int(properties["rcon.port"]),
             int(values.get("MCP_PORT", "6767")), int(values.get("MCMCP_BODY_PORT", "6768")),
             int(values.get("MCMCP_VIEWER_PORT", "6769"))]
    for port in ports:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    if properties["level-name"] != "world":
        raise ValueError("The selected server must use level-name=world for this fixture.")
    save_json(out / "minecraft-fixture.json", {"server": str(server), "seed": properties["level-seed"],
        "reset_world": reset_world, "server_properties_sha256": sha256_file(server / "server.properties"),
        "mcp_revision": subprocess.check_output(["git", "-C", str(ROOT.parent / "pm-minecraft"), "rev-parse", "HEAD"], text=True).strip(),
        "character": str(out / "character")})
    if reset_world:
        for name in ("world", "logs", ".fabric"):
            target = (server / name).resolve()
            if not target.is_relative_to(server) or target == server:
                raise ValueError(f"Reset leaves the selected server: {target}")
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(server / "wipe-world.ps1")], check=True)
    env_path = out / ".env.minecraft"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    stack = RealServerStack(server, out, env_path)
    try:
        stack.start()
        from workflows.minecraft_body.mcp_client import build_minecraft_tools
        observation = build_minecraft_tools(stack.mcp_url)["minecraft_call"]("minecraft_observe", {"include_image": False})
        save_json(out / "minecraft-initial.json", observation)
    except BaseException:
        stack.stop()
        raise
    os.environ["MINECRAFT_MCP_URL"] = stack.mcp_url
    return stack


def shift_history(out: Path) -> MemoryStore:
    """Import the full seed and shift its structured dates to end yesterday.

    The original file and text stay unchanged. All parsed timestamp fields
    receive the same offset. The manifest records this experimental transform.
    """
    ghost = MemoryStore.from_yaml(SEED)
    data = ghost.snapshot().serialize_knoxels()
    dates = []

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and key.startswith("timestamp") and isinstance(item, str):
                    dates.append(datetime.fromisoformat(item))
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(data)
    end = max(item.replace(tzinfo=None) for item in dates)
    target = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
    offset = target - end

    def shifted(value):
        if isinstance(value, dict):
            return {key: ((datetime.fromisoformat(item) + offset).isoformat()
                          if isinstance(key, str) and key.startswith("timestamp") and isinstance(item, str) else shifted(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [shifted(item) for item in value]
        return value

    from pm.memory.yaml_codec import RecordPayload
    ghost._all_knoxels = {key: RecordPayload.model_validate(shifted(value)).root for key, value in data.items()}
    ghost.states = [type(cycle).model_validate(shifted(cycle.model_dump(mode="json"))) for cycle in ghost.states]
    ghost.rebuild_knoxel_indexes()
    ghost.checkpoint = out / "ghost.pickle"
    save_json(out / "history-transform.json", {"source": str(SEED), "sha256": sha256_file(SEED),
        "offset_seconds": offset.total_seconds(), "original_end": end.isoformat(), "shifted_end": target.isoformat(),
        "records": len(data), "text_policy": "Preserve every text field; shift structured timestamp fields only."})
    return ghost


class ChatCapture:
    """Capture actual public delivery order and separate causal diagnostics."""

    def __init__(self, app, root: Path):
        self.app, self.root = app, root
        self.lock = threading.RLock()
        self.rows: list[dict] = []
        self.seen: set[int] = set()

    def receive(self, envelope) -> None:
        """Append an observed public event once, before any later review."""
        key = int(envelope.payload["knoxel_id"])
        with self.lock:
            if key in self.seen:
                return
            self.seen.add(key)
            with self.app.ghost.lock:
                record = self.app.ghost.get_knoxel_by_id(key)
                row = {"at": now(), "speaker": record.source, "text": str(record.content),
                       "id": key, "correlation": envelope.correlation_id,
                       "cause": envelope.causation_id, "metadata": dict(record.metadata)}
            self.rows.append(row)
            with (self.root / "chat.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{row['at']} {row['speaker']}: {row['text']}", flush=True)

    def public(self) -> list[dict]:
        """Return only the public text and time available to Jesse."""
        with self.lock:
            return [{key: row[key] for key in ("at", "speaker", "text")} for row in self.rows]


def jesse_message(model, history: list[dict], situation: str, topic: str, index: int) -> str:
    """Compose from public chat alone. No private runtime state enters this call."""
    system = ("Pretend to be human Jesse texting your longtime companion Spamuel. "
        "Write exactly one natural message, usually one to three sentences. Use casual lowercase when it fits. "
        "You are tired, warm, occasionally distracted, and willing to disagree. Do not interview or evaluate Spamuel. "
        "Stay responsive to what was said, let the topic develop, and avoid repetitive reassurance. "
        "Never repeat an unanswered question or send several versions of the same request. "
        "You may send a follow-up before an answer arrives. Never write Spamuel's words or stage directions. "
        "Do not mention being an agent or an experiment. Situation: " + situation + " Topic: " + topic)
    messages = [Turn(Role.System, system), Turn(Role.User,
        "Public conversation available as you start composing message " + str(index) + ":\n" + json.dumps(history, ensure_ascii=False))]
    return model.complete(messages, temperature=0.85, enable_thinking=False).strip()


def run_conversation(args, root: Path) -> dict:
    """Run independent composition and delivery beside the full application.

    A deadline ends missing progress as a failure. A short final quiet interval
    lets queued companion messages arrive. It never supplies a dummy reply.
    """
    os.environ["PM_PROMPT_LOG_DIR"] = str(root / "llm-logs")
    rng = random.Random(args.random_seed)
    topic = args.topic or rng.choice(TOPICS)
    situation = ("You shut Spamuel down months ago and have now restarted him. Acknowledge that once at the start. Then let the conversation move naturally."
                 if args.condition == "long-gap" else "You last talked yesterday. This is an ordinary continuation of your shared conversation.")
    ghost = shift_history(root) if args.condition == "recent" else None
    kwargs = {"ghost": ghost} if ghost is not None else {"checkpoint_path": root / "ghost.pickle"}
    app = build_app(seed_path=SEED, runtime_root=root / "runtime", config_path=str(args.config),
                    cognition="real", thought_period=30, world_poll_seconds=120,
                    emit_reply=lambda text: None, **kwargs)
    capture_chat = ChatCapture(app, root)
    app.bus.subscribe("conversation_capture", lambda event: event.topic in {"user.input", "dialogue.reply"}, capture_chat.receive)
    background = app.ghost.dialogue_history()[-20:]
    public_background = [{"at": row["timestamp"], "speaker": row["speaker"], "text": row["text"]} for row in background]
    save_json(root / "jesse-setup.json", {"situation": situation, "topic": topic, "public_history": public_background,
        "condition": args.condition, "random_seed": args.random_seed})
    model = LazyStageModel(resolve_connection(load_config(args.config)), ModelActivation())
    receipt = {"schema": "pm.conversation-review.v1", "started_at": now(), "condition": args.condition,
        "topic": topic, "human_review": "pending", "ok": False, "failures": []}
    started = time.monotonic()
    app.run()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="jesse") as pool:
            for index in range(1, args.messages + 1):
                app.raise_if_error()
                visible = public_background + capture_chat.public()
                save_json(root / "jesse-inputs" / f"{index:02}.json", visible)
                composing_at = now()
                pending = pool.submit(jesse_message, model, visible, situation, topic, index)
                while not pending.done():
                    app.raise_if_error()
                    if time.monotonic() - started > args.timeout:
                        raise TimeoutError("The conversation exceeded its recorded deadline.")
                    time.sleep(0.5)
                text = pending.result()
                if not text:
                    raise ValueError("Jesse returned an empty message.")
                delay = rng.uniform(2, 5) if index in {2, 5, 8} else rng.uniform(10, 22)
                until = time.monotonic() + delay
                while time.monotonic() < until:
                    app.raise_if_error()
                    time.sleep(0.5)
                correlation = app.user_input.submit(text, session=root.name)
                save_json(root / "jesse-inputs" / f"{index:02}-sent.json", {"composing_at": composing_at,
                    "sent_at": now(), "visible_messages": len(visible), "correlation": correlation, "text": text})
                # Most messages allow space for a reply. Deliberate bursts keep
                # independent composition visible without requiring alternation.
                if index not in {1, 4, 7}:
                    before = len(capture_chat.public())
                    deadline = time.monotonic() + args.pause
                    while time.monotonic() < deadline and len(capture_chat.public()) == before:
                        app.raise_if_error()
                        time.sleep(0.5)
            deadline = time.monotonic() + args.settle
            while time.monotonic() < deadline:
                app.raise_if_error()
                time.sleep(0.5)
        app.raise_if_error()
        receipt["ok"] = True
    except BaseException as error:
        receipt["failures"].append(f"{type(error).__name__}: {error}")
        (root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        app.shutdown()
        app.join()
        if app.error is not None:
            failure = f"{type(app.error).__name__}: {app.error}"
            if failure not in receipt["failures"]:
                receipt["failures"].append(failure)
            receipt["ok"] = False
            (root / "runtime-error.txt").write_text("".join(traceback.format_exception(app.error)), encoding="utf-8")
        receipt.update(ended_at=now(), messages=capture_chat.rows, runs=app.supervisor.receipts())
        receipt["user_messages"] = sum(row["speaker"] == "Jesse" for row in capture_chat.rows)
        receipt["companion_messages"] = sum(row["speaker"] == "Spamuel" for row in capture_chat.rows)
        if not receipt["companion_messages"]:
            receipt["ok"] = False
            receipt["failures"].append("No companion message reached public delivery.")
        save_json(root / "receipt.json", receipt)
    return receipt


def write_review(root: Path, receipt: dict) -> Path:
    """Create one review item for the full public transcript, including failures."""
    setup = json.loads((root / "jesse-setup.json").read_text(encoding="utf-8"))
    lines = []
    for row in receipt["messages"]:
        lines.extend([f"**{row['at']} · {row['speaker']}**", "", row["text"], ""])
    references = [{"label": "Completed receipt and causal diagnostics", "path": str((root / "receipt.json").relative_to(ROOT))},
                  {"label": "Jesse setup and public history", "path": str((root / "jesse-setup.json").relative_to(ROOT))},
                  {"label": "Conversation runner", "path": "dev/work_items/conversation_review.py"}]
    notes = [f"Observed {receipt['user_messages']} Jesse messages and {receipt['companion_messages']} Spamuel messages.",
             "Read in delivery order. Overlapping messages need no one-to-one pairing.",
             "Full PM runtime with private activity and enabled Minecraft. Human review is pending.",
             *receipt["failures"]]
    bundle = {"schema": "pm.dev-review.bundle.v1", "id": root.name, "work_item": "A17/A18 hardening",
        "title": "Async conversation · " + receipt["condition"] + " · " + root.name,
        "created_at": receipt["ended_at"], "summary": "A whole conversation with an independent Jesse model and the full companion runtime.",
        "source": {"path": str((root / "receipt.json").relative_to(ROOT)), "sha256": sha256_file(root / "receipt.json")},
        "items": [{"id": "conversation", "title": "Whole conversation", "persona": "Spamuel / preserved history",
            "scenario": receipt["condition"], "context": [{"role": "Situation", "text": setup["situation"]},
                {"role": "Jesse's topic", "text": setup["topic"]}], "content": "\n".join(lines),
            "reviewable": bool(receipt["companion_messages"]), "notes": notes, "references": references}]}
    path = root / "review-bundle.json"
    save_json(path, bundle)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.alpha.openrouter.yaml")
    parser.add_argument("--condition", choices=["long-gap", "recent"], required=True)
    parser.add_argument("--topic", default="")
    parser.add_argument("--random-seed", type=int, default=19)
    parser.add_argument("--messages", type=int, default=10)
    parser.add_argument("--pause", type=float, default=240)
    parser.add_argument("--settle", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=4200)
    parser.add_argument("--start-minecraft", action="store_true")
    parser.add_argument("--reset-world", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if not root.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Conversation evidence belongs under .runtime.")
    if args.reset_world and not args.start_minecraft:
        raise ValueError("A world reset requires an owned Minecraft stack.")
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    save_json(root / "origins.json", {"repository": capture(ROOT, source), "python": sys.version,
        "packages": {name: {"version": metadata.version(name),
            "direct_url": json.loads(metadata.distribution(name).read_text("direct_url.json"))}
                     for name in ("pm-coder", "pm-workflows")}, "seed_sha256": sha256_file(SEED)})
    stack = None
    try:
        stack = start_minecraft(root, args.reset_world) if args.start_minecraft else None
        receipt = run_conversation(args, root)
        write_review(root, receipt)
    except BaseException as error:
        (root / "runner-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        save_json(root / "runner-failure.json", {"ok": False, "at": now(), "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if stack is not None:
            try:
                from workflows.minecraft_body.mcp_client import build_minecraft_tools
                observation = build_minecraft_tools(stack.mcp_url)["minecraft_call"]("minecraft_observe", {"include_image": False})
                save_json(root / "minecraft-final.json", observation)
            finally:
                stack.stop()
    print(root / "receipt.json", flush=True)
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
