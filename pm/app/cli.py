"""Terminal entry point: run one application with stdin input and an optional web chat.

The caller passes the assembly that builds the application, so this module
never imports ``workflows``.

Two ways to start:

- ``--seed`` / ``--checkpoint`` / ``--ephemeral``: the terminal session as
  before. The runtime root defaults to ``./.runtime``.
- ``--home DIR``: the character home folder. Memory loads from and saves to
  ``DIR/ghost.pickle``, which must exist. The runtime root is ``DIR/runtime``.
  The persona comes from the home files. The web chat starts by default.

``--web`` starts the web chat in the terminal mode too.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import logging
from pathlib import Path

from pydantic import BaseModel

from pm.constants import WEB_CHAT_PORT


class LaunchPlan(BaseModel):
    """The resolved start options of one application run.

    ``home`` is the character home folder, or None in the terminal mode.
    ``seed`` is the seed file of the terminal mode; it is None with a home.
    ``checkpoint`` is the accepted Ghost checkpoint; None means an explicit
    in-memory session without persistence (``--ephemeral``). With a home,
    the checkpoint is ``home/ghost.pickle``.
    """

    home: Path | None
    seed: Path | None
    runtime_root: Path
    checkpoint: Path | None
    cognition: str
    config: str | None
    web: bool
    web_port: int


def build_parser(default_seed: Path) -> argparse.ArgumentParser:
    """Return the argument parser. ``default_seed`` is the assembly's seed file."""
    parser = argparse.ArgumentParser(description="pm cognition shell")
    parser.add_argument("--seed", default=None, help=f"Seed persona file. Default: {default_seed}")
    parser.add_argument("--runtime-root", default=None, help="Runtime root. Default: ./.runtime, or <home>/runtime")
    parser.add_argument("--cognition", choices=["real", "dummy"], default="real",
                        help="Real model-backed cognition, or the labeled dummy fixture.")
    parser.add_argument("--config", default=None, help="Model config path. Default: PM_MODEL_CONFIG or config.local.yaml")
    persistence = parser.add_mutually_exclusive_group()
    persistence.add_argument("--home", type=Path, help="Character home folder. Loads and saves <home>/ghost.pickle.")
    persistence.add_argument("--checkpoint", type=Path, help="Accepted Ghost checkpoint. Default: <runtime-root>/ghost.pickle")
    persistence.add_argument("--ephemeral", action="store_true", help="Run without Ghost persistence")
    parser.add_argument("--web", action="store_true", help="Start the web chat. Always on with --home.")
    parser.add_argument("--web-port", type=int, default=WEB_CHAT_PORT, help=f"Web chat port. Default: {WEB_CHAT_PORT}")
    return parser


def launch_plan(argv, default_seed: Path) -> LaunchPlan:
    """Parse ``argv`` into a launch plan. Invalid combinations exit through argparse.

    ``--home`` excludes ``--seed`` and ``--runtime-root``, because the home
    folder supplies the persona and the runtime root. The parser group
    already excludes ``--checkpoint`` and ``--ephemeral``.
    """
    parser = build_parser(default_seed)
    args = parser.parse_args(argv)
    if args.home is not None:
        if args.seed is not None:
            parser.error("--home excludes --seed: the persona comes from the home folder.")
        if args.runtime_root is not None:
            parser.error("--home excludes --runtime-root: the runtime root is <home>/runtime.")
        home = args.home.resolve()
        return LaunchPlan(home=home, seed=None, runtime_root=home / "runtime", checkpoint=home / "ghost.pickle",
                          cognition=args.cognition, config=args.config, web=True, web_port=args.web_port)
    runtime_root = Path(args.runtime_root) if args.runtime_root is not None else Path.cwd() / ".runtime"
    return LaunchPlan(
        home=None, seed=Path(args.seed) if args.seed is not None else Path(default_seed),
        runtime_root=runtime_root,
        checkpoint=None if args.ephemeral else (args.checkpoint or runtime_root / "ghost.pickle"),
        cognition=args.cognition, config=args.config, web=args.web, web_port=args.web_port,
    )


def build_from_plan(plan: LaunchPlan, *, assembly):
    """Build the application of ``plan`` without starting it.

    With a home, the character home opens the store; it raises
    ``FileNotFoundError`` when ``ghost.pickle`` is missing. Every accepted
    commit of that store saves ``ghost.pickle``.
    """
    if plan.home is not None:
        from pm.app.character_home import CharacterHome

        ghost = CharacterHome(plan.home).open_store()
        return assembly.build_app(
            seed_path=None, runtime_root=plan.runtime_root, emit_reply=print, ghost=ghost,
            cognition=plan.cognition, config_path=plan.config,
        )
    return assembly.build_app(
        seed_path=plan.seed, runtime_root=plan.runtime_root, emit_reply=print,
        checkpoint_path=plan.checkpoint, cognition=plan.cognition, config_path=plan.config,
    )


def main(argv=None, *, assembly) -> int:
    """Own diagnostics and child cleanup for the full terminal session."""
    from pm.host.diagnostics import configure_logging, close_logging
    from pm.host.processes import ensure_process_cleanup

    plan = launch_plan(argv, assembly.DEFAULT_SEED)
    configure_logging(plan.home or plan.runtime_root)
    ensure_process_cleanup()
    try:
        from pm.llm.prompt_log import backfill_cache_logs
        backfill_cache_logs()
        return _run_session(plan, assembly=assembly)
    except BaseException:
        logging.getLogger(__name__).exception("Terminal session stopped with an error")
        raise
    finally:
        close_logging()


def _run_session(plan: LaunchPlan, *, assembly) -> int:
    """Run one session of ``assembly`` until stdin closes (terminal mode), Ctrl+C, or an error.

    Shutdown signals modules and workers, closes the web chat, and joins work.
    A clean stop records a shutdown event.
    Every accepted commit already saved the checkpoint, so Ctrl+C loses no
    accepted state. A cycle that has not committed is not accepted state.
    """
    app = build_from_plan(plan, assembly=assembly)
    print(assembly.BANNER if plan.cognition == "dummy" else assembly.REAL_BANNER, flush=True)
    if plan.home is not None:
        print(f"Character home: {plan.home}", flush=True)
    web = None

    def read_stdin() -> None:
        try:
            for line in sys.stdin:
                text = line.rstrip("\n")
                if text.strip():
                    app.user_input.submit(text)
        except (EOFError, ValueError):
            pass
        finally:
            # Without the web chat, the end of stdin ends the session. With
            # it, the browser stays the input and Ctrl+C ends the session.
            if web is None:
                app.shutdown()

    # SIGTERM and, on Windows, Ctrl+Break or a closed console window stop
    # the session like Ctrl+C. A second signal forces exit after a thread dump.
    # Windows ends the process about 5 seconds after
    # a console window closes, so a long model call can still cut the stop.
    stop_signals = 0

    def stop_on_signal(signum, frame) -> None:
        nonlocal stop_signals
        stop_signals += 1
        if stop_signals > 1:
            from pm.host.diagnostics import exit_after_repeated_signal
            exit_after_repeated_signal()
        app.shutdown()

    previous = {signal.SIGINT: signal.signal(signal.SIGINT, stop_on_signal),
                signal.SIGTERM: signal.signal(signal.SIGTERM, stop_on_signal)}
    if sys.platform == "win32":
        previous[signal.SIGBREAK] = signal.signal(signal.SIGBREAK, stop_on_signal)

    try:
        if plan.web:
            from pm.app.web_chat import WebChatServer
            web = WebChatServer(app.ghost, app.user_input.submit, port=plan.web_port, on_error=app.on_error)
        app.run()
        if web is not None:
            web.start()
            print(f"Web chat: {web.url}", flush=True)
        reader = threading.Thread(target=read_stdin, name="stdin", daemon=True)
        reader.start()
        while not app.shutdown_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        try:
            if web is not None:
                web.stop()
        finally:
            try:
                app.join()
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
        app.raise_if_error()
    return 0
