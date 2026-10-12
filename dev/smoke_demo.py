"""Scripted demonstration through the real Ghost, Kernel, Bash, and bus.

Every check runs real modules and real workflow invocations. Fixed outputs are
dummy behavior, labeled as such. The process exits nonzero if any check fails.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm_core.model.knoxel_common import Feature
from pm_core.model.knoxel_enums import FeatureType, InterlocusType
from pm_workflows.kernel import Kernel
from pm_workflows.runtime import WorkflowRuntime
from pm_workflows.working_target import CommandResult

import ghost_mcp
from pm_core.ghost.service import Ghost
from main import BANNER, build_app
from bus.envelopes import Envelope, _new_id
from working_target import FolderTarget, GhostBashWrapper

SEED = ROOT / "data" / "spamuel_agentic_canon.yaml"

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    CHECKS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class Recorder:
    def __init__(self, bus, topic=None):
        self.events = []
        self.topic = topic
        bus.subscribe(f"demo-{id(self)}", self._accepts, self._deliver)

    def _accepts(self, envelope):
        if envelope.kind == "control":
            return False
        return self.topic is None or envelope.topic == self.topic

    def _deliver(self, envelope):
        self.events.append(envelope)


class _MemoryEnvironment:
    def __init__(self):
        self.files = {}

    def read_text(self, path):
        return self.files[path]

    def write_text(self, path, text):
        self.files[path] = text

    def move(self, source, destination):
        self.files[destination] = self.files.pop(source)

    def list_files(self, directory):
        return sorted(k for k in self.files if k.startswith(directory))

    def exec(self, command):
        return CommandResult("", "", 0)


_MANIFEST = """\
---
name: demo-role
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: role.py
    result_contract:
      type: json
      schema: {status: {enum: [done]}, summary: string}
phases:
  - name: work
    kind: role
    role: work
    on_status: {done: stop}
    on_invalid: {action: stop}
---
"""


def _store_feature(ghost, content):
    return ghost.store_knoxel(Feature(
        content=content, feature_type=FeatureType.Dialogue,
        interlocus=InterlocusType.Public, causal=True, override_source="demo",
    )).id


def _run_kernel(role_code, *, propagate=False, with_runtime=True) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="pmprobe-"))
    runtime = None
    if with_runtime:
        runtime = WorkflowRuntime(
            environment=_MemoryEnvironment(),
            stop_event=threading.Event(),
            tools={},
            propagate_python_errors=propagate,
        )
    (tmp / "role.py").write_text(role_code, encoding="utf-8")
    (tmp / "w.workflow.md").write_text(_MANIFEST, encoding="utf-8")
    kernel = Kernel(
        manifest_path=tmp / "w.workflow.md",
        workspace=tmp / "repo",
        task_id="demo",
        task_text="demo",
        base_dir=tmp,
        kernel_data_root=tmp / "kernel_data",
        runtime=runtime,
    )
    return kernel.run()


def run() -> int:
    print(BANNER, end="")
    clock = ManualClock()
    tmp = Path(tempfile.mkdtemp(prefix="pmnext-"))

    # 1. seed
    ghost = Ghost.from_yaml(SEED)
    ids = list(ghost.all_knoxels)
    check("1. seed: unique ids", len(ids) == len(set(ids)))

    # 2. virtual bash read + rejected write
    env = GhostBashWrapper(ghost, "read").open("c2")
    kid = min(env.visible_knoxels)
    check("2. bash read", env.exec(f"cat /work/ghost/knoxels/{kid}.yaml").exit_code == 0)
    check("2. bash write rejected", env.exec(f"echo x > /work/ghost/knoxels/{kid}.yaml").exit_code != 0)

    # 3. new feature seen by next invocation
    new_id = _store_feature(ghost, "freshly stored")
    check("3. next env sees new memory", new_id in GhostBashWrapper(ghost, "read").open("c3").visible_knoxels)

    # 4. restricted scope: bash and MCP hide an omitted record
    all_ids = set(ghost.all_knoxels)
    omitted = max(all_ids)
    kept = [i for i in sorted(all_ids) if i != omitted]
    scoped = GhostBashWrapper(ghost, "read", visible_ids=kept).open("c4")
    check("4. bash hides omitted", scoped.exec(f"cat /work/ghost/knoxels/{omitted}.yaml").exit_code != 0)
    try:
        ghost_mcp.build_tools(scoped)["ghost_get"](omitted)
        check("4. mcp hides omitted", False)
    except KeyError:
        check("4. mcp hides omitted", True)

    # 5/6. real Python role runs; default kernel works without runtime
    result = _run_kernel("def run(context):\n    return {'status': 'done', 'summary': 'x'}\n")
    check("5. real Python role runs", result["terminal_status"] == "done")
    bare = _run_kernel(
        "def run(context):\n    return {'status': 'done', 'summary': 'x'}\n",
        with_runtime=False,
    )
    check("6. default kernel without runtime", bare["terminal_status"] == "done")

    # App scenarios (manual clock, no threads).
    replies: list[str] = []
    app = build_app(seed_path=SEED, runtime_root=tmp, clock=clock,
                    emit_reply=replies.append, ghost=ghost)

    # 7. fanout
    rec_a = Recorder(app.bus, topic="topic.x")
    rec_b = Recorder(app.bus, topic="topic.x")
    app.bus.put(Envelope(id="e1", source="x", kind="data", topic="topic.x", correlation_id="", payload={}))
    check("7. fanout", len(rec_a.events) == len(rec_b.events) == 1 and rec_a.events[0].id == "e1")

    # 8. two inputs -> one GWT, two dummy actions
    r_actions = Recorder(app.bus, topic="action.dialogue")
    app.user_input.submit("hello")
    app.user_input.submit("world")
    app.main_cog.step(block=False)
    tick = app.ghost.current_tick_id
    check("8. two dummy actions from one batch", len(r_actions.events) == 2)
    check("8. one GWT recorded", tick in app.ghost.gwt_turns)

    # 9. two public replies
    app.dialogue.step(block=False)
    app.user_input.step(block=False)
    check("9. two public replies", replies == ["I heard you, Jesse.", "I heard you, Jesse."])

    # 10. /demo-task ordered telemetry
    telemetry = Recorder(app.bus, topic="task.telemetry")
    app.user_input.submit("/demo-task")
    app.main_cog.step(block=False)
    app.scheduler.step(block=False)
    clock.advance(1.0)
    app.scheduler.step(block=False)
    clock.advance(1.0)
    app.scheduler.step(block=False)
    statuses = [e.payload["status"] for e in telemetry.events]
    check("10. accepted/running/completed order", statuses == ["accepted", "running", "completed"])

    # 11. /memory reply count matches scoped search
    r_mem_reply = Recorder(app.bus, topic="dialogue.reply")
    app.user_input.submit("/memory")
    app.main_cog.step(block=False)
    app.tool_dialogue.step(block=False)
    scope_env = app.tool_dialogue.last_environment
    expected = len(ghost_mcp.build_tools(scope_env)["ghost_search"]("Jesse"))
    mem_feature = app.ghost.get_knoxel_by_id(r_mem_reply.events[-1].payload["knoxel_id"])
    check("11. memory count matches scoped search",
          mem_feature.content == f"I found {expected} memories about Jesse.")

    # 12. a thought never prints as public dialogue
    before_replies = len(replies)
    clock.advance(30.0)
    app.inner_thought.step(block=False)
    check("12. thought is not public dialogue", len(replies) == before_replies)

    # 13. terminate an idle module and join its thread
    terminations = Recorder(app.bus, topic="module.terminated")
    app.status.run()
    app.bus.put(Envelope(id=_new_id(), source="demo", kind="control", topic="terminate",
                         correlation_id="", payload={}, target="status"))
    app.status.join()
    check("13. idle module terminates cleanly", len(terminations.events) == 1)

    # 14-17. cooperative cancellation and error propagation
    cancelled = _run_kernel(
        "def run(context):\n"
        "    context.runtime.stop_event.set()\n"
        "    context.runtime.check_cancelled()\n"
        "    return {'status': 'done', 'summary': 'x'}\n",
    )
    check("14. active workflow accepts cancellation", cancelled["terminal_status"] == "terminated")

    propagated = False
    try:
        _run_kernel("def run(context):\n    raise ValueError('demo boom')\n", propagate=True)
    except ValueError:
        propagated = True
    check("16. ValueError propagates", propagated)

    converted = _run_kernel("def run(context):\n    raise ValueError('demo boom')\n", propagate=False)
    check("17. legacy conversion without flag", converted["ok"] is False)

    # 18. real folder target
    bash = shutil.which("bash")
    if bash is None:
        check("18. folder bash", False)
    else:
        fenv = FolderTarget(tmp / "folders", bash).open("f1")
        fenv.write_text("a.txt", "alpha\n")
        fenv.move("a.txt", "b.txt")
        ex = fenv.exec("echo done && cat b.txt")
        check("18. folder write/move/read",
              ex.exit_code == 0 and "done" in ex.stdout and "alpha" in ex.stdout)
        check("18. folder list", fenv.list_files("") == ["b.txt"])

    failed = [name for name, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())