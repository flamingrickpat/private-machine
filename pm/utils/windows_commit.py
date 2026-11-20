"""Measure machine-wide Windows commit before admitting additional PM participants."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass


class PerformanceInformation(ctypes.Structure):
    """Win32 PERFORMANCE_INFORMATION. Counters are expressed in system pages."""

    _fields_ = [
        ("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t), ("HandleCount", wintypes.DWORD),
        ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD),
    ]


@dataclass(frozen=True, slots=True)
class CommitHeadroom:
    available_bytes: int
    limit_bytes: int

    @classmethod
    def measure(cls) -> CommitHeadroom:
        counters = PerformanceInformation()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        if not psapi.GetPerformanceInfo(ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        available = (counters.CommitLimit - counters.CommitTotal) * counters.PageSize
        return cls(available, counters.CommitLimit * counters.PageSize)

    def require(self, minimum_gib: int) -> None:
        available_gib = self.available_bytes / (1024 ** 3)
        if available_gib < minimum_gib:
            raise MemoryError(f"Windows commit headroom is {available_gib:.1f} GiB; campaign needs {minimum_gib} GiB.")
