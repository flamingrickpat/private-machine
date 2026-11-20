"""Append local connection details without interleaving concurrent simulation writers."""
from datetime import datetime
import socket
import time

from pm.host.durable_files import ProcessLock
from pm.paths import REPO_ROOT


def append_current(message: str) -> None:
    root = REPO_ROOT / ".runtime"
    lock = ProcessLock(root / "current.lock").acquire(wait=True)
    try:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        while True:
            try:
                with (root / "current.md").open("a", encoding="utf-8") as stream:
                    stream.write(f"- {stamp}  {message}\n")
                return
            except PermissionError as error:
                if error.winerror not in (32, 33):
                    raise
                time.sleep(0.2)
    finally:
        lock.close()


def log_web_address(port: int) -> None:
    host = socket.gethostname()
    append_current(f"Simulation workspace ready. [Home network](http://{socket.gethostbyname(host)}:{port}/) "
                   f"· [This PC](http://127.0.0.1:{port}/) · [Hostname](http://{host}:{port}/)")
