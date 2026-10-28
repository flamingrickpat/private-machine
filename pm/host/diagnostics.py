"""Keep process diagnostics in the agent home and record stalled threads.

One process owns one UTF-8 log file with a formatter that names the UTC time,
the process, and the Python thread. Native crash traces and unhandled thread or
process exceptions go into the same file, so a stopped session leaves one
artifact to read.

:func:`configure_logging` installs the handler and hooks that had to be
replaced, and :func:`close_logging` restores them. The module keeps that state,
so call it once per process, at startup and after every worker stopped.
:func:`join_thread` reports a stuck thread's stack instead of waiting silently,
and :func:`exit_after_repeated_signal` dumps and exits when a second stop signal
arrives during a wedged shutdown.
"""
from __future__ import annotations

from collections.abc import Callable
import faulthandler
import logging
import os
from pathlib import Path
import sys
import threading
import traceback
from datetime import datetime, timezone
from types import TracebackType

#: The owned log handler, the hooks it replaced, and the levels it changed.
#: The hooks return ``object`` because the interpreter ignores a hook's result.
_HANDLER: logging.FileHandler | None = None
_THREAD_HOOK: Callable[[threading.ExceptHookArgs], object] | None = None
_EXCEPTION_HOOK: Callable[[type[BaseException], BaseException, TracebackType | None], object] | None = None
_FAULTS_WERE_ENABLED = False
_LEVELS: dict[str, int] = {}


def configure_logging(home: Path) -> Path:
    """Open one UTF-8 log per process under ``home/logs``.

    Each line includes UTC time, process ID, Python thread ID, and thread name.
    Native crash traces use the same file. File errors stop startup.
    """
    global _HANDLER, _THREAD_HOOK, _EXCEPTION_HOOK, _FAULTS_WERE_ENABLED
    if _HANDLER is not None:
        close_logging()
    directory = Path(home) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"runtime_{stamp}_{os.getpid()}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s pid=%(process)d thread=%(thread)d "
        "[%(threadName)s] %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    import time
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    for name in ("pm", "workflows"):
        _LEVELS[name] = logging.getLogger(name).level
        logging.getLogger(name).setLevel(logging.DEBUG)
    _HANDLER = handler
    _THREAD_HOOK = threading.excepthook
    _EXCEPTION_HOOK = sys.excepthook

    def thread_error(args: threading.ExceptHookArgs) -> None:
        """Record an unhandled thread exception, then run the previous hook."""
        logging.getLogger(__name__).error("Unhandled thread error thread=%s", args.thread.name,
                                         exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        if _THREAD_HOOK is not None:
            _THREAD_HOOK(args)

    def process_error(kind: type[BaseException], error: BaseException,
                      trace: TracebackType | None) -> None:
        """Record an unhandled main-thread exception, then run the previous hook."""
        logging.getLogger(__name__).error("Unhandled process error", exc_info=(kind, error, trace))
        if _EXCEPTION_HOOK is not None:
            _EXCEPTION_HOOK(kind, error, trace)

    threading.excepthook = thread_error
    sys.excepthook = process_error
    _FAULTS_WERE_ENABLED = faulthandler.is_enabled()
    faulthandler.enable(file=handler.stream, all_threads=True)
    logging.getLogger(__name__).info("Process started log=%s native_thread=%s", path, threading.get_native_id())
    return path


def close_logging() -> None:
    """Flush and close the owned handler after all application workers stop."""
    global _HANDLER, _THREAD_HOOK, _EXCEPTION_HOOK
    if _HANDLER is None:
        return
    logging.getLogger(__name__).info("Process diagnostics closed")
    faulthandler.disable()
    if _FAULTS_WERE_ENABLED:
        faulthandler.enable(all_threads=True)
    logging.getLogger().removeHandler(_HANDLER)
    _HANDLER.close()
    _HANDLER = None
    for name, level in _LEVELS.items():
        logging.getLogger(name).setLevel(level)
    _LEVELS.clear()
    if _THREAD_HOOK is not None:
        threading.excepthook = _THREAD_HOOK
        _THREAD_HOOK = None
    if _EXCEPTION_HOOK is not None:
        sys.excepthook = _EXCEPTION_HOOK
        _EXCEPTION_HOOK = None


def exit_after_repeated_signal() -> None:
    """Record all thread stacks and exit with code 130 after a second signal.

    This path avoids logging locks during a stuck shutdown. Windows closes
    the process job at exit, which also removes remaining owned children.
    """
    stream = _HANDLER.stream if _HANDLER is not None else sys.stderr
    os.write(stream.fileno(), b"Repeated stop signal: forced exit after thread dump.\n")
    faulthandler.dump_traceback(file=stream, all_threads=True)
    os._exit(130)


def join_thread(thread: threading.Thread) -> None:
    """Join a worker. Record its stack every five seconds until it stops."""
    logger = logging.getLogger(__name__)
    logger.info("Thread join start name=%s ident=%s native_id=%s", thread.name, thread.ident, thread.native_id)
    while thread.is_alive():
        thread.join(timeout=5)
        if thread.is_alive():
            frame = sys._current_frames().get(thread.ident)
            logger.warning("Thread still active name=%s ident=%s stack=\n%s", thread.name, thread.ident,
                           "".join(traceback.format_stack(frame)) if frame is not None else "unavailable")
    logger.info("Thread join end name=%s", thread.name)
