import threading
import time
import queue
from enum import StrEnum, Enum, auto
from queue import Queue
from typing import List, Dict, Any, Callable, Optional
from pydantic import BaseModel, Field

from pm.ghosts.ghost_lida import GhostLida


class TargetThread(StrEnum):
    Communication = "Communication"
    World = "World"
    Thought = "Thought"   # fix typo


class Triage(Enum):
    Trivial = auto()
    Normal = auto()
    Critical = auto()


class ShellInput(BaseModel):
    targets: List[TargetThread]
    source: str
    content: str
    causal_order: int = 0
    metadata: Dict[str, Any] | None = Field(default=None)


class BaseShell:
    """
    A lean, event-driven shell around a LIDA-like ghost.

    Thread model
    ------------
    - router:      pulls all external inputs from `inbox` and fans out to per-domain inputs + LIDA ingest.
    - verbal:      low-latency comms (chat). Wakes on new input *or* CCQ update event.
    - body:        low-latency world/avatar control. Wakes on new input *or* CCQ update event.
    - thought:     internal monologue / proactive cognition (timer-driven + event hooks).
    - lida:        heavyweight cognitive cycle; serializes broadcasts; raises CCQ events.
    - background:  timers/alarms/maintenance (e.g., consolidation, metacog monitors).

    Queues & Events
    ---------------
    - inbox:             ONLY external ingress (UI, STT, vision, tools).
    - lida_inbox:        items that should be visible to the cognitive cycle.
    - input_verbal/body: small domain inboxes to trim latency.
    - ccq_updated_{verbal,body}: events to notify workers of new CCQs.
    - out_subs:          per-target pub/sub for downstream sinks.
    """

    def __init__(self, ghost: GhostLida, db_path_local: Optional[str] = None):
        self.ghost = ghost
        self.db_path_local = db_path_local

        # --- Ingress / dispatch ---
        self.inbox: "Queue[ShellInput]" = Queue(maxsize=4096)
        self.lida_inbox: "Queue[Any]" = Queue(maxsize=4096)

        # Domain-specific low-latency inputs
        self.input_verbal: "Queue[Any]" = Queue(maxsize=1024)
        self.input_body: "Queue[Any]" = Queue(maxsize=1024)

        # CCQ update Events (raised by lida thread)
        self.ccq_updated_verbal = threading.Event()
        self.ccq_updated_body = threading.Event()

        # Subscribers for broadcast output per target
        self._out_subs: Dict[TargetThread, List[Queue]] = {
            TargetThread.Communication: [],
            TargetThread.World: [],
            TargetThread.Thought: [],
        }

        # Lifecycle
        self._active = threading.Event()
        self._active.set()

        # Threads
        self._threads: List[threading.Thread] = []

        # Internal lock for any shared-light operations
        self._lock = threading.RLock()

    # ------------------ Public API ------------------

    def submit(self, item: ShellInput) -> None:
        """External producers push all inputs here; non-blocking if possible."""
        self.inbox.put(item, block=True)

    def subscribe(self, target: TargetThread, q: Queue) -> None:
        """Downstream consumers (terminal, web UI, AR) register to receive broadcasts."""
        with self._lock:
            self._out_subs[target].append(q)

    def start(self) -> None:
        self._spawn(self.router_in_thread, "router")
        self._spawn(self.verbal_communication_thread, "verbal")
        self._spawn(self.environmental_action_thread, "body")
        self._spawn(self.internal_thought_thread, "thought")
        self._spawn(self.lida_thread, "lida")
        self._spawn(self.background_task_thread, "background")

    def stop(self) -> None:
        self._active.clear()
        # nudge blocking waits
        self.ccq_updated_verbal.set()
        self.ccq_updated_body.set()
        # Put sentinels to unblock any .get()
        for q_ in (self.inbox, self.lida_inbox, self.input_verbal, self.input_body):
            try:
                q_.put_nowait(None)  # type: ignore
            except queue.Full:
                pass

    def join(self, timeout: Optional[float] = None) -> None:
        for t in self._threads:
            t.join(timeout=timeout)

    # ------------------ Workers ------------------

    def router_in_thread(self):
        """Fan-out: All external inputs → domain lanes (+ feed LIDA)."""
        while self._active.is_set():
            item = self.inbox.get()
            if item is None:
                break

            # Fan out to domains and LIDA ingest
            for tgt in item.targets:
                if tgt == TargetThread.Communication:
                    self.input_verbal.put(item)
                elif tgt == TargetThread.World:
                    self.input_body.put(item)
                elif tgt == TargetThread.Thought:
                    # Thought lane is mostly self-driven; still route to LIDA
                    pass

            # Everything is also visible to the cognitive cycle
            self.lida_inbox.put(item)

    def verbal_communication_thread(self):
        """
        Low-latency chat path:
        - opportunistically prefills CCQ (if updated) for minimal latency,
        - triages inputs to decide whether to wait for LIDA updates,
        - completes & broadcasts a response,
        - also hands the action back into LIDA for learning / global consistency.
        """
        # Optional: prime with last CCQ snapshot
        self._prefill_verbal_ccq()

        while self._active.is_set():
            item = self._get_either(self.input_verbal, self.ccq_updated_verbal, wait_s=0.25)
            if item is None:
                # CCQ update ping → prefill; no user input
                self._prefill_verbal_ccq()
                continue

            # 1) Quick triage
            triage = self.ghost.verbal_quick_triage([item])  # expects list
            if triage == Triage.Critical:
                # Wait briefly for the next CCQ produced *after* this input
                self._drain_ccq_until(self.ccq_updated_verbal, min_causal=item.causal_order)
                self._prefill_verbal_ccq()

            # 2) Completion on verbal model (fast path)
            action = self.ghost.verbal_completion([item])

            # 3) Broadcast and feed to LIDA
            self.broadcast(TargetThread.Communication, action)
            self.lida_inbox.put(action)

    def environmental_action_thread(self):
        """Low-latency body control similar to verbal lane."""
        self._prefill_body_ccq()

        while self._active.is_set():
            item = self._get_either(self.input_body, self.ccq_updated_body, wait_s=0.25)
            if item is None:
                self._prefill_body_ccq()
                continue

            triage = self.ghost.body_quick_triage([item])
            if triage == Triage.Critical:
                self._drain_ccq_until(self.ccq_updated_body, min_causal=item.causal_order)
                self._prefill_body_ccq()

            action = self.ghost.get_body_action([item])
            self.broadcast(TargetThread.World, action)
            self.lida_inbox.put(action)

    def internal_thought_thread(self):
        """
        Self-driven cognition: reflection, planning, MCP calls, etc.
        Keep it timer-based with backoff to avoid hot loops.
        """
        backoff = 0.5
        while self._active.is_set():
            try:
                # Let the ghost decide if anything proactive should happen
                act = self.ghost.thought_tick()
                if act:
                    self.broadcast(TargetThread.Thought, act)
                    self.lida_inbox.put(act)
                time.sleep(backoff)
            except Exception:
                # don't kill the thread on ghost exceptions
                time.sleep(1.0)

    def lida_thread(self):
        """
        The heavy global cognitive cycle. It ingests accumulated inputs,
        runs codelets, attention competition, selects actions, and emits CCQs.
        """
        buffer: List[Any] = []
        while self._active.is_set():
            # Gather what arrived; block a little to batch
            try:
                itm = self.lida_inbox.get(timeout=0.1)
                if itm is None:
                    break
                buffer.append(itm)
                # Drain without blocking
                while True:
                    itm2 = self.lida_inbox.get_nowait()
                    if itm2 is None:
                        break
                    buffer.append(itm2)
            except queue.Empty:
                pass

            if not buffer:
                continue

            # Update ghost & tick cognitive cycle
            self.ghost.add_sensory_data(buffer)
            artifacts = self.ghost.tick_cognitive_cycle()
            buffer.clear()

            # Raise CCQ events and let low-latency threads prefill
            if getattr(artifacts, "ccq_verbal", None):
                self.ghost.verbal_com_prefill(artifacts.ccq_verbal)
                self.ccq_updated_verbal.set()
            if getattr(artifacts, "ccq_body", None):
                self.ghost.body_schema_prefill(artifacts.ccq_body)
                self.ccq_updated_body.set()

            # Clear events soon after to ensure next update can wake listeners
            # (listeners don't rely on staying set)
            self.ccq_updated_verbal.clear()
            self.ccq_updated_body.clear()

    def background_task_thread(self):
        """
        Timers / alarms / consolidation / health checks.
        Keep loops short and cooperative.
        """
        while self._active.is_set():
            try:
                self.ghost.maintenance_tick()  # sleep, consolidation, telemetry, etc.
            except Exception:
                pass
            time.sleep(1.0)

    # ------------------ Helpers ------------------

    def broadcast(self, target: TargetThread, payload: Any):
        """Fan-out to all subscribers of given target."""
        with self._lock:
            subs = list(self._out_subs.get(target, []))
        for q_ in subs:
            try:
                q_.put_nowait(payload)
            except queue.Full:
                # Drop if slow consumer; optionally add metrics
                pass

    def _prefill_verbal_ccq(self):
        try:
            ccq = self.ghost.get_latest_ccq_verbal()
            if ccq:
                self.ghost.verbal_com_prefill(ccq)
        except Exception:
            pass

    def _prefill_body_ccq(self):
        try:
            ccq = self.ghost.get_latest_ccq_body()
            if ccq:
                self.ghost.body_schema_prefill(ccq)
        except Exception:
            pass

    @staticmethod
    def _get_either(q_in: Queue, evt: threading.Event, wait_s: float):
        """
        Wait up to wait_s for either queue item or event.
        Returns: queue item if any; None if event fired; None if timeout.
        """
        deadline = time.time() + wait_s
        while time.time() < deadline:
            try:
                return q_in.get_nowait()
            except queue.Empty:
                if evt.is_set():
                    return None
                time.sleep(0.01)
        return None

    @staticmethod
    def _drain_ccq_until(evt: threading.Event, min_causal: int, timeout_s: float = 1.0):
        """
        Wait briefly for a CCQ update expected to be >= min_causal.
        Implementation note: we don't have direct causal numbers for CCQs here,
        so we just wait a bit; wire-through causal numbers if needed.
        """
        evt.wait(timeout=timeout_s)

    def _spawn(self, fn: Callable[[], None], name: str):
        t = threading.Thread(target=fn, name=f"shell-{name}", daemon=True)
        t.start()
        self._threads.append(t)
