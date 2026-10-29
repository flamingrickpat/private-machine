"""Exercise the OS utility through real jobs, children, files, and locks."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from types import FrameType

from pm.host.durable_files import ProcessLock
from pm.utils.operating_system import OperatingSystem


ROOT = Path(__file__).resolve().parents[4]
PYTHON = OperatingSystem.python_executable(bypass_launcher=True)


def setUpModule() -> None:
    """Keep even failed test children inside the test runner's lifetime."""
    if OperatingSystem.is_windows() and not OperatingSystem.install_child_cleanup():
        raise RuntimeError("Process lifetime checks require an installed Windows cleanup job.")


@unittest.skipUnless(OperatingSystem.is_windows(), "Windows Job Objects require Windows.")
class ProcessCleanupTests(unittest.TestCase):
    def test_parent_exit_removes_all_descendants(self) -> None:
        grandchild = "import time; time.sleep(300)"
        child = (
            "import subprocess,sys,time,json,os; from pm.utils.operating_system import OperatingSystem; "
            f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}],stdout=OperatingSystem.discard_output,stderr=OperatingSystem.discard_output); "
            "print(json.dumps([os.getpid(),p.pid]),flush=True); time.sleep(300)"
        )
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), TemporaryDirectory() as directory:
                parent = (
                    "import subprocess,sys; from pm.utils.operating_system import OperatingSystem; "
                    "assert OperatingSystem.install_child_cleanup(); "
                    "assert OperatingSystem.child_cleanup_installed(); "
                    f"p=subprocess.Popen([sys.executable,'-u','-c',{child!r}],stdout=OperatingSystem.pipe,text=True); "
                    "print(p.stdout.readline(),flush=True); sys.stdin.readline(); "
                    + ("OperatingSystem.exit_immediately(7)" if exit_code else "p.stdout.close()")
                )
                with (Path(directory) / "stderr.log").open("w+", encoding="utf-8") as errors:
                    process = OperatingSystem.start_piped_process([PYTHON, "-u", "-c", parent], cwd=ROOT, error_log=errors)
                    try:
                        output, _ = process.communicate("exit\n", timeout=15)
                        errors.seek(0)
                        self.assertEqual(process.returncode, exit_code, errors.read())
                        children: list[int] = json.loads(output)
                        self.assertEqual(len(children), 2)
                        for process_id in children:
                            self.assertTrue(OperatingSystem.wait_for_process_exit(process_id, timeout_seconds=5),
                                            f"Child {process_id} survived.")
                    finally:
                        if process.poll() is None:
                            OperatingSystem.kill_process_tree(process)
                            process.communicate(timeout=10)

    def test_forced_parent_death_removes_its_child(self) -> None:
        parent = (
            "import subprocess,sys,time; from pm.utils.operating_system import OperatingSystem; "
            "assert OperatingSystem.install_child_cleanup(); "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']); "
            "print(p.pid,flush=True); time.sleep(300)"
        )
        with TemporaryDirectory() as directory:
            with (Path(directory) / "stderr.log").open("w+", encoding="utf-8") as errors:
                process = OperatingSystem.start_piped_process([PYTHON, "-u", "-c", parent], cwd=ROOT, error_log=errors)
                try:
                    assert process.stdout is not None
                    child_id = int(process.stdout.readline())
                    OperatingSystem.terminate_process_id(process.pid, exit_code=9)
                    process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 9)
                    self.assertTrue(OperatingSystem.wait_for_process_exit(child_id, timeout_seconds=5))
                finally:
                    if process.poll() is None:
                        OperatingSystem.kill_process_tree(process)
                        process.communicate(timeout=10)


class OperatingSystemTests(unittest.TestCase):
    def test_legacy_text_and_binary_pipe_wiring_retains_the_actual_bytes(self) -> None:
        binary = OperatingSystem.start_binary_process(
            [PYTHON, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()[::-1]); sys.stderr.buffer.write(b'seen')"],
            stdin=OperatingSystem.pipe, stdout=OperatingSystem.pipe, stderr=OperatingSystem.pipe, hidden=True,
        )
        output, errors = binary.communicate(b"abc", timeout=5)
        self.assertEqual((binary.returncode, output, errors), (0, b"cba", b"seen"))
        text = OperatingSystem.start_text_process(
            [PYTHON, "-c", "import sys; print(sys.stdin.read().upper())"],
            stdin=OperatingSystem.pipe, stdout=OperatingSystem.pipe, stderr=OperatingSystem.pipe, hidden=True,
        )
        spoken, diagnostics = text.communicate("hello", timeout=5)
        self.assertEqual((text.returncode, spoken.strip(), diagnostics), (0, "HELLO", ""))
        result = OperatingSystem.run_binary([PYTHON, "-c", "print('binary-result')"], capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), b"binary-result")
        self.assertEqual(OperatingSystem.output_text([PYTHON, "-c", "print('text-result')"]).strip(), "text-result")

    def test_the_historical_worker_stop_removes_real_descendants(self) -> None:
        from dev.work_items.a18b.supervision import run_worker

        with TemporaryDirectory() as directory:
            root = Path(directory)
            program = (
                "import subprocess,sys,json,os,time; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']); "
                f"Path({str(root / 'pids.json')!r}).write_text(json.dumps([os.getpid(),p.pid])); "
                f"Path({str(root / 'STOP')!r}).touch(); time.sleep(300)"
            )
            with (root / "worker.log").open("w+", encoding="utf-8") as log:
                result = run_worker([PYTHON, "-c", program], cwd=ROOT, case=root, log=log,
                                    timeout_seconds=10, cleanup_seconds=0)
            self.assertTrue(result.forced)
            self.assertEqual(result.termination_reason, "operator STOP")
            self.assertNotEqual(result.returncode, 0)
            process_ids: list[int] = json.loads((root / "pids.json").read_text(encoding="utf-8"))
            for process_id in process_ids:
                self.assertTrue(OperatingSystem.wait_for_process_exit(process_id, timeout_seconds=5))

    def test_low_level_file_publication_preserves_the_old_linked_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.bin"
            linked = root / "published.bin"
            replacement = root / "replacement.bin"
            with original.open("wb", buffering=0) as stream:
                self.assertEqual(OperatingSystem.write_descriptor(stream.fileno(), b"original"), 8)
                OperatingSystem.sync_file(stream.fileno())
            OperatingSystem.link_file(original, linked)
            replacement.write_bytes(b"replacement")
            OperatingSystem.replace_file(replacement, linked)
            self.assertEqual(original.read_bytes(), b"original")
            self.assertEqual(linked.read_bytes(), b"replacement")
            self.assertFalse(replacement.exists())

    def test_stop_handlers_restore_the_actual_previous_signal_bindings(self) -> None:
        def stop(number: int, frame: FrameType | None) -> None:
            raise AssertionError("The registration check must not send a signal.")

        previous = OperatingSystem.register_stop_signals(stop, console_break=True)
        try:
            replaced = OperatingSystem.register_stop_signals(stop)
            self.assertTrue(replaced)
            for handler in replaced.values():
                self.assertIs(handler, stop)
        finally:
            OperatingSystem.restore_signals(previous)

    def test_real_finite_startup_script_writes_its_receipt(self) -> None:
        from pm.llm.activation import run_startup

        with TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "finite-startup.py"
            script.write_text("from pathlib import Path\nPath('receipt.txt').write_text('finished', encoding='utf-8')\n",
                              encoding="utf-8")
            asyncio.run(run_startup(script))
            self.assertEqual((root / "receipt.txt").read_text(encoding="utf-8"), "finished")

    def test_real_text_and_binary_commands_retain_output_and_exit_codes(self) -> None:
        result = OperatingSystem.run_text(
            [PYTHON, "-c", "import sys; print('text-output'); print('error-output',file=sys.stderr); sys.exit(3)"],
            cwd=ROOT, timeout_seconds=5,
        )
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout.strip(), "text-output")
        self.assertEqual(result.stderr.strip(), "error-output")
        output = OperatingSystem.output_bytes([PYTHON, "-c", "import sys; sys.stdout.buffer.write(b'bytes-output')"])
        self.assertEqual(output, b"bytes-output")

    def test_real_asynchronous_process_finishes_and_reaps(self) -> None:
        async def run() -> int:
            process = await OperatingSystem.start_async_process([PYTHON, "-c", "raise SystemExit(4)"], cwd=ROOT)
            code = await process.wait()
            await OperatingSystem.terminate_async_process(process)
            return code

        self.assertEqual(asyncio.run(run()), 4)

    def test_a_competing_process_observes_native_lock_ownership_and_release(self) -> None:
        command = (
            "from pathlib import Path\n"
            "import sys\n"
            "from pm.host.durable_files import BodyBusy, ProcessLock\n"
            "try:\n"
            "    lock = ProcessLock(Path(sys.argv[1])).acquire()\n"
            "except BodyBusy:\n"
            "    print('busy')\n"
            "else:\n"
            "    print('owned')\n"
            "    lock.close()\n"
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ownership.lock"
            with ProcessLock(path):
                competing = OperatingSystem.run_text([sys.executable, "-c", command, str(path)], cwd=ROOT,
                                                     timeout_seconds=5)
                self.assertEqual(competing.returncode, 0, competing.stderr)
                self.assertEqual(competing.stdout.strip(), "busy")
            released = OperatingSystem.run_text([sys.executable, "-c", command, str(path)], cwd=ROOT,
                                                timeout_seconds=5)
            self.assertEqual(released.returncode, 0, released.stderr)
            self.assertEqual(released.stdout.strip(), "owned")


if __name__ == "__main__":
    unittest.main()
