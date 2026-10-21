"""Verify Windows cleanup with real child and grandchild processes."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(os.name == "nt", "Windows Job Objects require Windows.")
class ProcessCleanupTests(unittest.TestCase):
    def test_parent_exit_removes_all_descendants(self):
        grandchild = "import time; time.sleep(300)"
        child = (
            "import subprocess,sys,time,json,os; "
            f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            "print(json.dumps([os.getpid(),p.pid]),flush=True); time.sleep(300)"
        )
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code):
                parent = (
                    "import subprocess,sys,os; import pm.host.processes as owned; "
                    "owned.ensure_process_cleanup(); assert owned._JOB is not None; "
                    f"p=subprocess.Popen([sys.executable,'-u','-c',{child!r}],stdout=subprocess.PIPE,text=True); "
                    "print(p.stdout.readline(),flush=True); sys.stdin.readline(); "
                    + ("os._exit(7)" if exit_code else "p.stdout.close()")
                )
                # The base interpreter avoids a venv launcher between the test and its child.
                process = subprocess.Popen([sys._base_executable, "-u", "-c", parent],
                    cwd=Path(__file__).resolve().parents[4], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    output, error = process.communicate("exit\n", timeout=15)
                    self.assertEqual(process.returncode, exit_code, error)
                    children = json.loads(output)
                    self.assertEqual(len(children), 2)
                    for pid in children:
                        handle = kernel.OpenProcess(0x00100000, False, pid)
                        if handle:
                            try:
                                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0, f"Child {pid} survived.")
                            finally:
                                kernel.CloseHandle(handle)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate(timeout=10)
