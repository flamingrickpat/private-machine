"""Terminal entry point: read stdin lines into one running application.

The caller passes the assembly that builds the application, so this module
never imports ``workflows``.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path



def main(argv=None, *, assembly) -> int:
    """Run the terminal session of ``assembly`` until stdin closes or an error stops it."""
    parser = argparse.ArgumentParser(description="pm cognition shell")
    parser.add_argument("--seed", default=str(assembly.DEFAULT_SEED))
    parser.add_argument("--runtime-root", default=str(Path.cwd() / ".runtime"))
    parser.add_argument("--cognition", choices=["real", "dummy"], default="real",
                        help="Real model-backed cognition, or the labeled dummy fixture.")
    parser.add_argument("--config", default=None, help="Model config path. Default: PM_MODEL_CONFIG or config.local.yaml")
    persistence = parser.add_mutually_exclusive_group()
    persistence.add_argument("--checkpoint", type=Path, help="Accepted Ghost checkpoint. Default: <runtime-root>/ghost.json")
    persistence.add_argument("--ephemeral", action="store_true", help="Run without Ghost persistence")
    args = parser.parse_args(argv)

    app = assembly.build_app(
        seed_path=args.seed,
        runtime_root=args.runtime_root,
        emit_reply=print,
        checkpoint_path=None if args.ephemeral else (args.checkpoint or Path(args.runtime_root) / "ghost.json"),
        cognition=args.cognition,
        config_path=args.config,
    )
    print(assembly.BANNER if args.cognition == "dummy" else assembly.REAL_BANNER, flush=True)
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

