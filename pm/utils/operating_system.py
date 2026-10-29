"""The shared operating-system boundary, without runtime-package dependencies.

``OperatingSystem`` owns native process, signal, lock, and filesystem operations.
Callers retain application policy: which command to run, when to stop, which
resource to lock, and whether cancellation should interrupt a wait. This module
imports only the standard library at startup, so memory persistence can use it
without importing the application or workflow assembly. The historical RSS
measurement loads its existing optional psutil provider only when requested.

Windows cleanup retains one noninherited Job Object handle in this process.
Closing its last handle terminates its remaining members. This binds descendants
to this runtime, not the runtime to an external shell that launched it. Existing
model services and their autorestart scripts remain separately owned.

Platform branches live here. Existing non-Windows behavior is preserved; Linux
and macOS lifecycle guarantees require their own implementation and real checks.
Native handle failures retain their OS errors. A rejected cleanup-job install
keeps its existing warning and returns false. No domain state is persisted here.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, MutableMapping, Sequence
import ctypes
from ctypes import wintypes
import logging
import os
from pathlib import Path
from importlib import import_module
import shutil
import signal
import subprocess
import sys
from types import FrameType
from typing import ClassVar, IO, NoReturn, Protocol, TypeAlias, cast


SignalHandler: TypeAlias = Callable[[int, FrameType | None], None]
SignalBindings: TypeAlias = dict[int, SignalHandler | int | None]
ProcessStream: TypeAlias = int | IO[str] | IO[bytes] | None


class OperatingSystem:
    """One typed entry point for native host behavior and process creation."""

    environment: ClassVar[MutableMapping[str, str]] = os.environ
    seek_from_end: ClassVar[int] = os.SEEK_END
    pipe: ClassVar[int] = subprocess.PIPE
    discard_output: ClassVar[int] = subprocess.DEVNULL
    merged_output: ClassVar[int] = subprocess.STDOUT
    _job_handle: ClassVar[int | None] = None

    class _ResidentInfo(Protocol):
        rss: int

    class _ResidentProcess(Protocol):
        def memory_info(self) -> OperatingSystem._ResidentInfo: ...

    class _PsutilModule(Protocol):
        @staticmethod
        def Process() -> OperatingSystem._ResidentProcess: ...

    @staticmethod
    def copy_environment() -> dict[str, str]:
        """Snapshot the current environment for one child launch."""
        return dict(OperatingSystem.environment)

    @staticmethod
    def resident_memory_mb() -> float:
        """Preserve the live harness's optional RSS provider and Unix fallback.

        The optional foreign module is adapted to its narrow public contract.
        It is loaded only when this existing measurement is requested.
        """
        try:
            provider = cast(OperatingSystem._PsutilModule, import_module("psutil"))
            return provider.Process().memory_info().rss / 1e6
        except ImportError:
            pass
        if sys.platform == "win32":
            raise ModuleNotFoundError("Resident-memory measurement requires psutil on Windows.")
        try:
            with Path("/proc/self/statm").open() as stream:
                return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
        except (OSError, ValueError, AttributeError):
            pass
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1e6 if sys.platform == "darwin" else 1e3)

    @staticmethod
    def is_windows() -> bool:
        """Return whether this interpreter uses the Windows native interfaces."""
        return os.name == "nt"

    @staticmethod
    def process_id() -> int:
        """Return this process's OS identity for diagnostics and provenance."""
        return os.getpid()

    @classmethod
    def python_executable(cls, *, bypass_launcher: bool = False) -> str:
        """Return Python, optionally bypassing the Windows virtualenv launcher.

        The base interpreter does not load virtualenv packages. It is useful
        when a process-lifetime check must observe the actual interpreter PID.
        """
        if bypass_launcher and cls.is_windows():
            return str(Path(sys.base_prefix) / "python.exe")
        return sys.executable

    @staticmethod
    def sync_file(file_descriptor: int) -> None:
        """Flush an already written file descriptor to durable storage."""
        os.fsync(file_descriptor)

    @staticmethod
    def replace_file(source: Path, destination: Path) -> None:
        """Atomically replace the destination with the caller's staged file."""
        os.replace(source, destination)

    @staticmethod
    def link_file(source: Path, destination: Path) -> None:
        """Publish a hard link. An existing destination remains an error."""
        os.link(source, destination)

    @staticmethod
    def write_descriptor(file_descriptor: int, data: bytes) -> int:
        """Write directly to a descriptor without acquiring a logging lock."""
        return os.write(file_descriptor, data)

    @staticmethod
    def exit_immediately(exit_code: int) -> NoReturn:
        """Exit without Python cleanup, releasing OS-owned handles."""
        os._exit(exit_code)

    @classmethod
    def _creation_flags(cls, hidden: bool) -> int:
        """Keep Windows console-window flags out of application callers."""
        return subprocess.CREATE_NO_WINDOW if hidden and cls.is_windows() else 0

    @staticmethod
    def script_command(script: Path) -> list[str]:
        """Select the existing interpreter command for a supported startup script."""
        suffix = script.suffix.lower()
        if suffix == ".ps1":
            return [shutil.which("pwsh") or "powershell", "-NoProfile", "-NonInteractive",
                    "-ExecutionPolicy", "Bypass", "-File", str(script)]
        if suffix == ".py":
            return [sys.executable, str(script)]
        if suffix == ".sh":
            return ["bash", str(script)]
        raise ValueError(f"Unsupported startup script suffix: {suffix}.")

    @classmethod
    def start_binary_process(
        cls, command: Sequence[str], *, cwd: Path | str | None = None,
        stdin: ProcessStream = None, stdout: ProcessStream = None, stderr: ProcessStream = None,
        env: Mapping[str, str] | None = None, hidden: bool = False, new_process_group: bool = False,
    ) -> subprocess.Popen[bytes]:
        """Start an existing binary-I/O helper with its caller's stream wiring."""
        return subprocess.Popen(command, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
                                env=env, creationflags=cls._creation_flags(hidden),
                                start_new_session=new_process_group and not cls.is_windows())

    @classmethod
    def start_text_process(
        cls, command: Sequence[str], *, cwd: Path | str | None = None,
        stdin: ProcessStream = None, stdout: ProcessStream = None, stderr: ProcessStream = None,
        env: Mapping[str, str] | None = None, hidden: bool = False,
        encoding: str | None = None, errors: str | None = None, bufsize: int = -1,
    ) -> subprocess.Popen[str]:
        """Start an existing text-I/O helper without changing its stream policy."""
        return subprocess.Popen(command, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
                                env=env, text=True, encoding=encoding, errors=errors, bufsize=bufsize,
                                creationflags=cls._creation_flags(hidden))

    @classmethod
    def run_binary(
        cls, command: Sequence[str], *, cwd: Path | str | None = None,
        stdout: ProcessStream = None, stderr: ProcessStream = None,
        env: Mapping[str, str] | None = None, capture_output: bool = False,
        timeout_seconds: float | None = None, check: bool = False, hidden: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a binary command with the existing capture or inherited-output policy."""
        return subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, env=env,
                              capture_output=capture_output, timeout=timeout_seconds, check=check,
                              creationflags=cls._creation_flags(hidden))

    @classmethod
    def start_logged_process(
        cls, command: Sequence[str], *, cwd: Path, log: IO[bytes],
        environment: Mapping[str, str],
    ) -> subprocess.Popen[bytes]:
        """Start a hidden child with closed stdin and merged binary log output."""
        return subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, env=environment,
            creationflags=cls._creation_flags(True),
        )

    @classmethod
    def start_piped_process(
        cls, command: Sequence[str], *, cwd: Path, error_log: IO[str],
    ) -> subprocess.Popen[str]:
        """Start a hidden worker with UTF-8 stdin/stdout and a separate log."""
        return subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=error_log, text=True, encoding="utf-8", bufsize=1,
            creationflags=cls._creation_flags(True),
        )

    @classmethod
    def run_text(
        cls, command: Sequence[str], *, cwd: Path | str | None = None,
        timeout_seconds: float | None = None, check: bool = False,
        encoding: str | None = None, errors: str | None = None, hidden: bool = False,
        stdout: ProcessStream = None, stderr: ProcessStream = None,
        env: Mapping[str, str] | None = None, capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a finite command and retain its real text output and exit code."""
        return subprocess.run(
            command, cwd=cwd, capture_output=capture_output, text=True, stdout=stdout, stderr=stderr, env=env,
            timeout=timeout_seconds, check=check, encoding=encoding, errors=errors,
            creationflags=cls._creation_flags(hidden),
        )

    @staticmethod
    def output_bytes(
        command: Sequence[str], *, cwd: Path | None = None, suppress_errors: bool = False,
    ) -> bytes:
        """Capture finite command bytes. A nonzero exit raises its real error."""
        return subprocess.check_output(
            command, cwd=cwd, stderr=subprocess.DEVNULL if suppress_errors else None,
        )

    @staticmethod
    def output_text(command: Sequence[str], *, cwd: Path | None = None, suppress_errors: bool = False) -> str:
        """Capture finite command text using the existing locale decoding."""
        return subprocess.check_output(command, cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL if suppress_errors else None)

    @classmethod
    def kill_process_group(
        cls, process: subprocess.Popen[str] | subprocess.Popen[bytes], *, log: IO[str],
    ) -> None:
        """Preserve the historical case worker's tree/group kill and exit race."""
        if sys.platform == "win32":
            result = cls.run_binary(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                    stdout=log, stderr=cls.merged_output)
            if result.returncode != 0 and process.poll() is None:
                raise RuntimeError(f"Could not stop owned worker tree {process.pid}. Inspect its log and service ports.")
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # It exited between the caller's poll and the native signal.

    @classmethod
    def open_shared_read_handle(cls, path: Path, *, share_writes: bool = False) -> int:
        """Hold a real Windows read handle that denies atomic replacement.

        ``share_writes`` preserves tests that permit in-place writes but deny
        deletion. The caller closes this handle after the observed operation.
        """
        if not cls.is_windows():
            raise NotImplementedError("Windows file-sharing handles require Windows.")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        handle = cast(int | None, kernel.CreateFileW(str(path), 0x80000000, 3 if share_writes else 1, None, 3, 0, None))
        if handle == ctypes.c_void_p(-1).value or handle is None:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    @staticmethod
    def close_native_handle(handle: int) -> None:
        """Close a Windows handle obtained through this utility."""
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    @classmethod
    async def start_async_process(cls, command: Sequence[str], *, cwd: Path) -> asyncio.subprocess.Process:
        """Start a hidden finite command without forwarding its output."""
        return await asyncio.create_subprocess_exec(
            *command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=cls._creation_flags(True),
        )

    @staticmethod
    async def terminate_async_process(process: asyncio.subprocess.Process) -> None:
        """Terminate and reap an asynchronous process that is still running."""
        if process.returncode is None:
            process.terminate()
            await process.wait()

    @classmethod
    def kill_process_tree(cls, process: subprocess.Popen[str] | subprocess.Popen[bytes]) -> None:
        """Force Windows descendants to exit, or preserve the other-host kill path."""
        if cls.is_windows():
            cls.run_text(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True, hidden=True)
        else:
            process.kill()

    @staticmethod
    def terminate_process(process: subprocess.Popen[str] | subprocess.Popen[bytes]) -> None:
        """Preserve a caller's direct-process termination policy."""
        process.terminate()

    @staticmethod
    def kill_process(process: subprocess.Popen[str] | subprocess.Popen[bytes]) -> None:
        """Force one direct process to exit without changing its caller's wait."""
        process.kill()

    @staticmethod
    def lock_file(stream: IO[bytes]) -> None:
        """Acquire the existing nonblocking native file lock at this position."""
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @classmethod
    def register_stop_signals(cls, handler: SignalHandler, *, console_break: bool = False) -> SignalBindings:
        """Install the caller's shutdown callback and return the prior handlers."""
        signals = [signal.SIGINT, signal.SIGTERM]
        if console_break and cls.is_windows():
            signals.append(signal.SIGBREAK)
        previous: SignalBindings = {}
        for number in signals:
            previous[number] = signal.signal(number, handler)
        return previous

    @staticmethod
    def restore_signals(previous: Mapping[int, SignalHandler | int | None]) -> None:
        """Restore exactly the signal handlers replaced by the caller."""
        for number, handler in previous.items():
            signal.signal(number, handler)

    @classmethod
    def open_source_file(cls, path: Path) -> None:
        """Open Markdown by association and executable source in the existing editor."""
        if path.suffix == ".md":
            os.startfile(str(path))
            return
        editor = shutil.which("subl")
        if editor is None:
            for candidate in (
                Path("C:/Program Files/Sublime Text/sublime_text.exe"),
                Path("C:/Program Files/Sublime Text 3/sublime_text.exe"),
            ):
                if candidate.is_file():
                    editor = str(candidate)
                    break
        if editor is None:
            raise ValueError("Sublime Text is required to open Python and YAML source safely.")
        subprocess.Popen([editor, str(path)], creationflags=cls._creation_flags(True))

    @classmethod
    def child_cleanup_installed(cls) -> bool:
        """Report whether this process retains its Windows cleanup-job handle."""
        return cls._job_handle is not None

    @classmethod
    def terminate_process_id(cls, process_id: int, *, exit_code: int) -> None:
        """Force one explicitly owned process to exit, preserving native errors."""
        if sys.platform != "win32":
            os.kill(process_id, signal.SIGKILL)
            return
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = cast(int | None, kernel.OpenProcess(0x0001, False, process_id))
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel.TerminateProcess(handle, exit_code):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)

    @classmethod
    def install_child_cleanup(cls) -> bool:
        """Retain the existing Windows kill-on-close job for this runtime's children.

        Native structure declarations stay inside the boundary. Their fields
        follow the Windows ABI; they are not application records. A host that
        rejects job installation keeps the existing warning-and-join behavior.
        """
        if cls._job_handle is not None:
            return True
        if not cls.is_windows():
            return False

        class BasicLimits(ctypes.Structure):
            process_time: int
            job_time: int
            flags: int
            minimum: int
            maximum: int
            active: int
            affinity: int
            priority: int
            scheduling: int
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
                        ("active", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            read_ops: int
            write_ops: int
            other_ops: int
            read_bytes: int
            write_bytes: int
            other_bytes: int
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            basic: BasicLimits
            io: IoCounters
            process_memory: int
            job_memory: int
            peak_process: int
            peak_job: int
            _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        # ctypes restypes define this external adapter's concrete return value.
        job = cast(int | None, kernel.CreateJobObjectW(None, None))
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
                or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())):
            error = ctypes.get_last_error()
            if job:
                kernel.CloseHandle(job)
            logging.getLogger(__name__).warning("Child process cleanup job unavailable windows_error=%s", error)
            return False
        cls._job_handle = job
        logging.getLogger(__name__).info("Child process cleanup job installed")
        return True

    @classmethod
    def wait_for_process_exit(cls, process_id: int, *, timeout_seconds: float) -> bool:
        """Wait on a real Windows process handle without hiding access failures."""
        if not cls.is_windows():
            raise NotImplementedError("Native process-exit observation is implemented for Windows.")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = cast(int | None, kernel.OpenProcess(0x00100000, False, process_id))
        if not handle:
            error = ctypes.get_last_error()
            if error in {87, 1168}:  # The process no longer exists.
                return True
            raise ctypes.WinError(error)
        try:
            result = kernel.WaitForSingleObject(handle, int(timeout_seconds * 1000))
            if result == 0:
                return True
            if result == 258:
                return False
            raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)
