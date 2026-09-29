"""Construction (``build_app``) and the interactive terminal session.

``python main.py`` starts the terminal session. ``python demo.py`` runs the
finite scenario checks through the same ``build_app`` construction.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from pm_core.ghost.service import Ghost
from ghost_mcp import build_tools
from modules import (
    CheckSubprocess,
    DummyDialogue,
    DummyMainCogLoop,
    InnerThoughtLoop,
    MainCogLoop,
    ScheduleSubprocesses,
    StoryDialogue,
    TaskTable,
    ToolPlusDialogue,
    UserInputProxy,
)
from bus.bus import BusSystem
from runtime.shell import Application
from working_target import GhostBashWrapper
from workflow_adapter import PROJECT_ROOT

BANNER = "DUMMY COGNITION \u2014 no LLM calls\n"
REAL_BANNER = "REAL COGNITION \u2014 model activates on first request\n"

DEFAULT_SEED = Path(__file__).resolve().parent / "data" / "spamuel_agentic_canon.yaml"


def default_config_path() -> str:
    if os.environ.get("PM_MODEL_CONFIG"):
        return os.environ["PM_MODEL_CONFIG"]
    if Path("config.local.yaml").is_file():
        return "config.local.yaml"
    return "config.example.yaml"


def _cognition_modules(cognition, *, ghost, target, bus, on_error, clock,
                      runtime_root, workflow_paths, stage_model, config_path):
    """Return the (main_cog, dialogue) pair for the requested cognition path."""
    if cognition == "real":
        from workflows.dialogue_story.module_tools import build_story_tools
        from workflows.main_cognition.module_tools import build_main_cognition_tools

        if stage_model is None:
            from runtime.cognitive_model import LazyStageModel
            from runtime.model_activation import ModelActivation
            from runtime.models import load_config, resolve_connection

            config = load_config(config_path)
            stage_model = LazyStageModel(resolve_connection(config), ModelActivation())
        main_cog = MainCogLoop(
            "main_cog", bus, on_error, clock, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "main_cognition" / "main_cognition.workflow.md",
            tools_factory=build_main_cognition_tools(ghost, runtime_root, stage_model.complete),
            runtime_root=runtime_root,
            ghost=ghost,
        )
        dialogue = StoryDialogue(
            "dialogue", bus, on_error, clock, None,
            working_target=target,
            workflow=PROJECT_ROOT / "workflows" / "dialogue_story" / "dialogue_story.workflow.md",
            tools_factory=build_story_tools(stage_model),
            runtime_root=runtime_root,
            ghost=ghost,
        )
        return main_cog, dialogue
    main_cog = DummyMainCogLoop(
        "main_cog", bus, on_error, clock, None,
        **workflow_paths("main_cognition"),
    )
    dialogue = DummyDialogue(
        "dialogue", bus, on_error, clock, None,
        **workflow_paths("dialogue"),
    )
    return main_cog, dialogue


def build_app(
    seed_path,
    runtime_root,
    thought_period: float = 30.0,
    task_period: float = 0.1,
    clock=time.monotonic,
    emit_reply=print,
    ghost=None,
    checkpoint_path=None,
    *,
    cognition: str = "dummy",
    stage_model=None,
    config_path: str | None = None,
) -> Application:
    if cognition not in {"real", "dummy"}:
        raise ValueError("cognition must be 'real' or 'dummy'.")
    if ghost is not None and checkpoint_path is not None:
        raise ValueError("Supply a Ghost or a checkpoint path, not both.")
    resolved_config = config_path or default_config_path()

    if ghost is None:
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        if checkpoint is not None and checkpoint.exists():
            ghost = Ghost.from_checkpoint(checkpoint)
        else:
            ghost = Ghost.from_yaml(Path(seed_path), checkpoint=checkpoint)
    target = GhostBashWrapper(ghost, mode="read")
    bus = BusSystem()
    tasks = TaskTable(clock)
    app = Application(bus, [])

    on_error = app.on_error
    runtime_root = Path(runtime_root)

    def workflow_paths(name: str):
        return {
            "working_target": target,
            "workflow": PROJECT_ROOT / "workflows" / f"{name}.workflow.md",
            "tools_factory": build_tools,
            "runtime_root": runtime_root,
            "ghost": ghost,
        }

    user_input = UserInputProxy("user_input", bus, on_error, clock, ghost, emit_reply)
    main_cog, dialogue = _cognition_modules(
        cognition, ghost=ghost, target=target, bus=bus, on_error=on_error,
        clock=clock, runtime_root=runtime_root, workflow_paths=workflow_paths,
        stage_model=stage_model, config_path=resolved_config,
    )
    inner_thought = InnerThoughtLoop(
        "inner_thought", bus, on_error, clock, thought_period,
        **workflow_paths("inner_thought"),
    )
    scheduler = ScheduleSubprocesses(
        "scheduler", bus, on_error, clock, task_period,
        **workflow_paths("schedule"), tasks=tasks,
    )
    status = CheckSubprocess(
        "status", bus, on_error, clock, None,
        **workflow_paths("status"), tasks=tasks,
    )
    tool_dialogue = ToolPlusDialogue(
        "tool_dialogue", bus, on_error, clock, None,
        **workflow_paths("tool_dialogue"),
    )

    modules = [
        user_input, main_cog, dialogue, inner_thought,
        scheduler, status, tool_dialogue,
    ]
    for module in modules:
        bus.subscribe(module.module_id, module.accepts, module.deliver)
    app.modules = modules

    app.user_input = user_input
    app.main_cog = main_cog
    app.dialogue = dialogue
    app.inner_thought = inner_thought
    app.scheduler = scheduler
    app.status = status
    app.tool_dialogue = tool_dialogue
    app.ghost = ghost
    app.bus = bus
    app.tasks = tasks
    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spamuel cognition shell")
    parser.add_argument("--seed", default=str(DEFAULT_SEED))
    parser.add_argument("--runtime-root", default=str(Path.cwd() / ".runtime"))
    parser.add_argument("--cognition", choices=["real", "dummy"], default="real",
                        help="Real model-backed cognition, or the labeled dummy fixture.")
    parser.add_argument("--config", default=None, help="Model config path. Default: PM_MODEL_CONFIG or config.local.yaml")
    persistence = parser.add_mutually_exclusive_group()
    persistence.add_argument("--checkpoint", type=Path, help="Accepted Ghost checkpoint. Default: <runtime-root>/ghost.json")
    persistence.add_argument("--ephemeral", action="store_true", help="Run without Ghost persistence")
    args = parser.parse_args(argv)

    app = build_app(
        seed_path=args.seed,
        runtime_root=args.runtime_root,
        emit_reply=print,
        checkpoint_path=None if args.ephemeral else (args.checkpoint or Path(args.runtime_root) / "ghost.json"),
        cognition=args.cognition,
        config_path=args.config,
    )
    print(BANNER if args.cognition == "dummy" else REAL_BANNER, flush=True)
    app.run()

    def read_stdin() -> None:
        try:
            for line in sys.stdin:
                text = line.rstrip("\n")
                if text.strip():
                    app.user_input.submit(text)
        except (EOFError, ValueError):
            pass
        finally:
            app.shutdown()

    reader = threading.Thread(target=read_stdin, daemon=True)
    reader.start()

    try:
        while reader.is_alive() and app.error is None:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        app.join()
        app.raise_if_error()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
