from dataclasses import dataclass
import time
import threading
from typing import Optional, Tuple

from pydantic import BaseModel


def now_ms() -> int:
    # wall clock in ms; if you fear jumps you can use time.time_ns() // 1_000_000
    return int(time.time() * 1000)

@dataclass(frozen=True)
class HLC:
    # Hybrid Logical Clock
    pt_ms: int        # physical time in ms
    lc: int           # logical counter

class HLCClock:
    """
    Single-process HLC suitable for multithreaded use.
    """
    def __init__(self, node_id: str):
        self._node_id = node_id
        self._lock = threading.Lock()
        self._hlc = HLC(pt_ms=now_ms(), lc=0)

    @property
    def node_id(self) -> str:
        return self._node_id

    def _update(self, local_pt: int, external: Optional[HLC]) -> HLC:
        # Algorithm per Kulkarni et al. (used by CockroachDB, etc.)
        a = self._hlc
        b = external or a
        max_pt = max(a.pt_ms, b.pt_ms, local_pt)
        if max_pt == a.pt_ms == b.pt_ms:
            # concurrent step on same max physical time
            new_lc = max(a.lc, b.lc) + 1
        elif max_pt == a.pt_ms:
            # local dominates, keep or bump local lc if equal times
            new_lc = a.lc + (1 if a.pt_ms == local_pt else 0)
        elif max_pt == b.pt_ms:
            new_lc = b.lc + 1
        else:
            # local wall clock moved forward beyond both
            new_lc = 0
        return HLC(pt_ms=max_pt, lc=new_lc)

    def send(self) -> HLC:
        """Use when *emitting* an event (no external dependency)."""
        with self._lock:
            self._hlc = self._update(local_pt=now_ms(), external=None)
            return self._hlc

    def recv(self, ext: HLC) -> HLC:
        """Use when *consuming* events stamped with ext, then emitting a follow-up."""
        with self._lock:
            self._hlc = self._update(local_pt=now_ms(), external=ext)
            return self._hlc

class CausalStamp(BaseModel):
    hlc: HLC
    lane: str         # "verbal" | "body" | "thought" | "lida" ...
    lane_seq: int     # per-lane seq number (monotone)
    node_id: str      # from HLCClock

class HLCAuthority:
    def __init__(self, node_id: str = "ghost-0"):
        self._clock = HLCClock(node_id=node_id)
        self._lane_seq = {
            "verbal": 0,
            "body": 0,
            "thought": 0,
            "lida": 0,
        }
        self._lock = threading.RLock()
        self._latest_ccq = {
            "verbal": None,  # stores HLC of the last emitted CCQ
            "body": None,
        }

    # ---- stamping helpers ----
    def _next_seq(self, lane: str) -> int:
        with self._lock:
            self._lane_seq[lane] += 1
            return self._lane_seq[lane]

    def stamp_emit(self, lane: str) -> CausalStamp:
        """
        Call when a lane emits something *not directly caused by a specific input*
        (e.g., proactive thought, UI message).
        """
        hlc = self._clock.send()
        return CausalStamp(hlc=hlc, lane=lane, lane_seq=self._next_seq(lane), node_id=self._clock.node_id)

    def stamp_after_inputs(self, lane: str, input_stamps: list[CausalStamp]) -> CausalStamp:
        """
        Call when a lane produces output *because of* inputs.
        Ensures output is causally after all of them.
        """
        if input_stamps:
            max_hlc = max((s.hlc for s in input_stamps), key=lambda h: (h.pt_ms, h.lc))
            hlc = self._clock.recv(max_hlc)
        else:
            hlc = self._clock.send()
        return CausalStamp(hlc=hlc, lane=lane, lane_seq=self._next_seq(lane), node_id=self._clock.node_id)

    # ---- CCQ management ----
    def publish_ccq_verbal(self, ccq_payload, caused_by: list[CausalStamp]):
        stamp = self.stamp_after_inputs("lida", caused_by)
        # attach stamp to CCQ
        ccq_payload.causal = stamp
        with self._lock:
            self._latest_ccq["verbal"] = stamp.hlc
        return ccq_payload

    def publish_ccq_body(self, ccq_payload, caused_by: list[CausalStamp]):
        stamp = self.stamp_after_inputs("lida", caused_by)
        ccq_payload.causal = stamp
        with self._lock:
            self._latest_ccq["body"] = stamp.hlc
        return ccq_payload

    def latest_ccq_hlc(self, lane: str) -> Optional[HLC]:
        with self._lock:
            return self._latest_ccq[lane]

def ccq_includes(self, lane: str, input_hlc: HLC) -> bool:
    latest = self.ghost.latest_ccq_hlc(lane)
    if not latest:
        return False
    return (latest.pt_ms, latest.lc) >= (input_hlc.pt_ms, input_hlc.lc)