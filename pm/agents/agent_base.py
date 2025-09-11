from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Protocol
from pydantic import BaseModel, ValidationError
import random
import time
from datetime import datetime

from pm.agents.agent_manager import AgentManager
from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_proxy import LlmManagerProxy
from pm.utils.duplex_utils import DuplexSignalFinish, DuplexSignalTerminate, DuplexStartGenerationTool, DuplexStartGenerationText, DuplexSignalEog

# ──────────────────────────────────────────────────────────────────────────────
# Controller protocol (adapt this to your LlmManagerLLama)
# ──────────────────────────────────────────────────────────────────────────────

Role = str  # "system" | "user" | "assistant"
Message = Tuple[Role, str]

class Session(Protocol):
    def complete_text(self, *, max_tokens:int=512, temperature:float=0.7, stop:Optional[List[str]]=None) -> str: ...
    def complete_json(self, schema:Type[BaseModel], *, max_tokens:int=512, temperature:float=0.2) -> BaseModel: ...
    def send_user(self, text:str) -> None: ...
    def halt(self) -> None: ...
    # (optional) if you expose KV helpers:
    def mark(self) -> int: ...
    def rollback_to(self, pos:int) -> None: ...

class CompletionController(Protocol):
    def open_session(self, initial_messages:List[Message], *, few_shots:List[Message]|None=None) -> Session: ...
    # optional: your controller can auto-detect prefix subset & keep KV hot

# ──────────────────────────────────────────────────────────────────────────────
# Prompt DSL
# ──────────────────────────────────────────────────────────────────────────────

class OpKind(Enum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    LITERAL = auto()
    EXAMPLE_BEGIN = auto()
    EXAMPLE_END = auto()
    COMPLETION_TEXT = auto()
    COMPLETION_JSON = auto()

@dataclass
class PromptOp:
    kind: OpKind
    content: Optional[str] = None
    target_key: Optional[str] = None
    schema: Optional[Type[BaseModel]] = None

# Convenience constructors
def System(text:str) -> PromptOp: return PromptOp(OpKind.SYSTEM, text)
def User(text:str) -> PromptOp: return PromptOp(OpKind.USER, text)
def Assistant(text:str) -> PromptOp: return PromptOp(OpKind.ASSISTANT, text)
def Literal(text:str) -> PromptOp: return PromptOp(OpKind.LITERAL, text)
def ExampleBegin() -> PromptOp: return PromptOp(OpKind.EXAMPLE_BEGIN)
def ExampleEnd() -> PromptOp: return PromptOp(OpKind.EXAMPLE_END)
def CompletionText(target_key:str) -> PromptOp: return PromptOp(OpKind.COMPLETION_TEXT, target_key=target_key)
def CompletionJSON(schema:Type[BaseModel], target_key:str) -> PromptOp:
    return PromptOp(OpKind.COMPLETION_JSON, target_key=target_key, schema=schema)

# ──────────────────────────────────────────────────────────────────────────────
# BaseAgent
# ──────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Orchestrates a plan of PromptOps with:
    - single static preamble before the first completion (KV-friendly)
    - multi-step text → JSON completions
    - example recording & optional rating
    """
    name: str = "BaseAgent"

    def __init__(self, llm: LlmManagerProxy, manager: AgentManager):
        self.llm = llm
        self.agent_manager = manager
        self.input: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self._recording: bool = False
        self._current_example: List[Message] = []
        self._examples_collected: List[List[Message]] = []

    # ── hooks to override ─────────────────────────────────────────────────────
    def get_system_prompts(self) -> List[str]:
        return []

    def get_default_few_shots(self) -> List[Message]:
        return []

    def get_rating_system_prompt(self) -> str:
        return ""

    def get_rating_probability(self) -> float:
        return 0.0

    def build_plan(self) -> List[PromptOp]:
        raise NotImplementedError

    @classmethod
    def execute(cls, input_dict:Dict[str, Any], llm: LlmManagerProxy, manager: AgentManager) -> Dict[str, Any]:
        self = cls(llm, manager)
        self.input = input_dict
        self.results = {}
        plan = self.build_plan()

        # 1) construct the static preamble (everything up to the first completion)
        preamble: List[Message] = []
        few_shots = self.get_default_few_shots()
        example_mode = False

        for op in plan:
            if op.kind == OpKind.EXAMPLE_BEGIN:
                example_mode = True
                self._current_example = []
                continue
            if op.kind == OpKind.EXAMPLE_END:
                example_mode = False
                if self._current_example:
                    self._examples_collected.append(self._current_example[:])
                    self._current_example = []
                continue
            if op.kind in (OpKind.COMPLETION_TEXT, OpKind.COMPLETION_JSON):
                break  # stop – we’ll send preamble once, then live-complete
            # accumulate messages (system/user/assistant/literal)
            if op.kind in (OpKind.SYSTEM, OpKind.USER, OpKind.ASSISTANT, OpKind.LITERAL):
                role = "system" if op.kind == OpKind.SYSTEM else \
                       "user" if op.kind == OpKind.USER else \
                       "assistant" if op.kind == OpKind.ASSISTANT else \
                       "user"  # Literal defaults to user-turn payload
                preamble.append((role, op.content or ""))
                if example_mode:
                    self._current_example.append((role, op.content or ""))

        # 2) open one long-lived session (your manager keeps KV hot)
        # session = controller.open_session(preamble, few_shots=few_shots)

        q_from_ai = Queue()
        q_from_user = Queue()

        def get_from_queue_until_end():
            buffer = []
            while True:
                res = q_from_ai.get(block=True, timeout=None)
                if isinstance(res, DuplexSignalEog):
                    break
                buffer.append(res)
            return "".join(buffer)

        def t():
            content = self.llm.completion_text(LlmPreset.Default, preamble, CommonCompSettings(temperature=0.3, max_tokens=1024, duplex=True, queue_from_user=q_from_user, queue_to_user=q_from_ai, wait_for_start_signal=True))
            self.results["_full_output"] = content

        thr = threading.Thread(target=t, daemon=True)
        thr.start()

        # 3) now walk the plan from the first completion onward
        started = False
        for op in plan:
            if op.kind in (OpKind.SYSTEM, OpKind.ASSISTANT, OpKind.LITERAL, OpKind.EXAMPLE_BEGIN, OpKind.EXAMPLE_END):
                # already included in the preamble (or handled above); skip here
                continue

            started = True

            if op.kind in (OpKind.USER,):
                q_from_user.put(op.content)
            elif op.kind == OpKind.COMPLETION_TEXT:
                q_from_user.put(DuplexStartGenerationText())
                text = get_from_queue_until_end()
                self.results[op.target_key or "text"] = text
                if self._is_recording():
                    self._append_example(("assistant", text))
            elif op.kind == OpKind.COMPLETION_JSON and op.schema is not None:
                q_from_user.put(DuplexStartGenerationTool(op.schema))
                text = get_from_queue_until_end()
                try:
                    obj = op.schema.model_validate_json(text)
                except ValidationError as e:
                    # If your manager returns raw text sometimes, store error and raw
                    obj = None
                    self.results[(op.target_key or "json") + "_error"] = str(e)

                self.results[op.target_key or "json"] = obj
                if self._is_recording():
                    self._append_example(("assistant", obj.json() if obj else ""))

        q_from_user.put(DuplexSignalTerminate())

        # 4) optional rating/mining
        #self._maybe_rate_example(controller, session)

        return self.results

    # ── helpers ───────────────────────────────────────────────────────────────
    def _is_recording(self) -> bool:
        return bool(self._current_example)

    def _append_example(self, msg:Message) -> None:
        self._current_example.append(msg)

    def _maybe_rate_example(self, controller:CompletionController, session:Session) -> None:
        if not self._examples_collected:
            return
        p = self.get_rating_probability()
        if p <= 0 or random.random() > p:
            return

        # build a single text block to rate (last example)
        ex = self._examples_collected[-1]
        # rating prompt
        rating_msgs: List[Message] = [
            ("system", self.get_rating_system_prompt()),
            ("user", "Please rate the following agent output (dialogue + summary + category):\n\n" +
                     "\n".join(f"{r.upper()}: {c}" for r, c in ex))
        ]
        rating_session = controller.open_session(rating_msgs, few_shots=[])
        try:
            class Rating(BaseModel):
                rating: int
                reason: str
            res = rating_session.complete_json(Rating, max_tokens=128, temperature=0.0)
            rating = int(getattr(res, "rating", 0))
            reason = getattr(res, "reason", "")
        except Exception:
            rating = 0
            reason = "rating failed"
        rating_session.halt()

        # persist via your agent_manager if present
        try:
            self.agent_manager.save_example_to_db(
                self.name, datetime.now(), "callstack_unavailable", 0.0,
                prompt_repr="\n".join(f"{r}: {c}" for r, c in ex),
                rating=rating, reason=reason
            )
        except Exception:
            pass
