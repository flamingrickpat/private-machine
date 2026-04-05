from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Protocol, Callable

from json_repair import repair_json
from pydantic import BaseModel, ValidationError
import random
import time
from datetime import datetime

from pm.agents.agent_manager import AgentManager
from pm.system.llm.llm_common import CommonCompSettings, LlmPreset
from pm.system.llm.llm_proxy import LlmManagerProxy
from pm.utils.duplex_utils import DuplexSignalFinish, DuplexSignalTerminate, DuplexStartGenerationTool, DuplexStartGenerationText, DuplexSignalEog, DuplexSignalFinished, DuplexJsonBegin, DuplexAssistantInjectBegin, DuplexAssistantInjectEnd

logger = logging.getLogger(__name__)

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
    FUNCTION_CALL = auto()

@dataclass
class PromptOp:
    kind: OpKind
    content: Optional[str] = None
    source_key: Optional[str] = None
    target_key: Optional[str] = None
    schema: Optional[Type[BaseModel]] = None
    delegate: Callable[[str], str] = None


# Convenience constructors
def System(text: str) -> PromptOp:
    return PromptOp(OpKind.SYSTEM, text)

def User(text: str = None, source_key: str = None) -> PromptOp:
    return PromptOp(OpKind.USER, text, source_key=source_key)

def Assistant(text: str) -> PromptOp:
    return PromptOp(OpKind.ASSISTANT, text)

def Literal(text: str) -> PromptOp:
    return PromptOp(OpKind.LITERAL, text)

def ExampleBegin() -> PromptOp:
    return PromptOp(OpKind.EXAMPLE_BEGIN)

def ExampleEnd() -> PromptOp:
    return PromptOp(OpKind.EXAMPLE_END)

def CompletionText(target_key: str) -> (PromptOp):
    return PromptOp(OpKind.COMPLETION_TEXT, target_key=target_key)

def CompletionJSON(schema: Type[BaseModel], target_key: str) -> PromptOp:
    return PromptOp(OpKind.COMPLETION_JSON, target_key=target_key, schema=schema)

def FunctionCall(source_key: str = None, target_key: str = None, delegate: Callable[[str], str] = None) -> PromptOp:
    return PromptOp(OpKind.FUNCTION_CALL, source_key=source_key, target_key=target_key, delegate=delegate)

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

    def get_log_name(self) -> str:
        return self.name

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

        first_comp_op_idx = -1
        for i in range(len(plan)):
            op = plan[i]
            if op.kind == OpKind.EXAMPLE_BEGIN:
                example_mode = True
                self._current_example = []
                continue
            elif op.kind == OpKind.EXAMPLE_END:
                example_mode = False
                if self._current_example:
                    self._examples_collected.append(self._current_example[:])
                    self._current_example = []
                continue
            elif op.kind in (OpKind.COMPLETION_TEXT, OpKind.COMPLETION_JSON):
                first_comp_op_idx = i
                break  # stop – we’ll send preamble once, then live-complete
            # accumulate messages (system/user/assistant/literal)
            elif op.kind in (OpKind.SYSTEM, OpKind.USER, OpKind.ASSISTANT, OpKind.LITERAL):
                role = "system" if op.kind == OpKind.SYSTEM else \
                       "user" if op.kind == OpKind.USER else \
                       "assistant" if op.kind == OpKind.ASSISTANT else \
                       "user"  # Literal defaults to user-turn payload
                preamble.append((role, op.content or ""))
                if example_mode:
                    self._current_example.append((role, op.content or ""))
            else:
                raise ValueError(f"Unhandled Agent operation kind: {op.kind}")

        if first_comp_op_idx >= 0:
            plan = plan[first_comp_op_idx:]

        # 2) open one long-lived session (your manager keeps KV hot)
        # session = controller.open_session(preamble, few_shots=few_shots)

        cnt = 0
        succ = False
        while True:
            q_from_ai = llm.get_queue()
            q_from_user = llm.get_queue()

            def get_from_queue_until_end():
                buffer = []
                while True:
                    try:
                        res = q_from_ai.get(block=False)
                        if isinstance(res, DuplexSignalEog) or isinstance(res, DuplexSignalFinished):
                            break
                        buffer.append(res)
                    except queue.Empty:
                        time.sleep(1)
                res = "".join(buffer)
                if "</think>" in res:
                    res = res.split("</think>")[1]
                return res

            def get_json_from_queue_until_end():
                buffer = []
                recording = False
                while True:
                    try:
                        res = q_from_ai.get(block=False)
                        if isinstance(res, DuplexSignalEog) or isinstance(res, DuplexSignalFinished):
                            break
                        if isinstance(res, DuplexJsonBegin):
                            recording = True
                        elif recording:
                            buffer.append(res)
                    except queue.Empty:
                        time.sleep(1)
                res = "".join(buffer)
                if "</think>" in res:
                    res = res.split("</think>")[1]
                return res

            thr = None
            try:
                def t():
                    content = self.llm.completion_text(LlmPreset.Default, preamble, CommonCompSettings(temperature=0.3, max_tokens=1024, duplex=True, queue_from_user=q_from_user, queue_to_user=q_from_ai, wait_for_start_signal=True, caller_id=self.get_log_name(), enable_thinking=False, seed=time.time_ns() + cnt))
                    self.results["_full_output"] = content

                thr = threading.Thread(target=t, daemon=True)
                thr.name = "base_agent_thread_for_duplex_completion"
                thr.start()

                # 3) now walk the plan from the first completion onward
                for op in plan:
                    if op.kind in (OpKind.SYSTEM, OpKind.EXAMPLE_BEGIN, OpKind.EXAMPLE_END):
                        # already included in the preamble (or handled above); skip here
                        continue
                    elif op.kind == OpKind.USER:
                        if op.content is not None:
                            q_from_user.put(op.content)
                        elif op.source_key is not None:
                            q_from_user.put(self.results[op.source_key])
                        else:
                            raise ValueError("OpKind.USER needs content or source_key.")
                    elif op.kind == OpKind.ASSISTANT:
                        q_from_user.put(DuplexAssistantInjectBegin())
                        q_from_user.put(op.content)
                        q_from_user.put(DuplexAssistantInjectEnd())
                    elif op.kind == OpKind.COMPLETION_TEXT:
                        q_from_user.put(DuplexStartGenerationText())
                        text = get_from_queue_until_end()
                        self.results[op.target_key or "text"] = text.strip()
                        if self._is_recording():
                            self._append_example(("assistant", text))
                    elif op.kind == OpKind.COMPLETION_JSON and op.schema is not None:
                        q_from_user.put(DuplexStartGenerationTool(op.schema))
                        text = get_json_from_queue_until_end()
                        good_json_string = repair_json(text)
                        try:
                            obj = op.schema.model_validate_json(good_json_string)
                        except:
                            logger.critical("ERROR Bad JSON: " + text)
                            logger.critical("ERROR Bad JSON Repaired: " + good_json_string)
                            raise
                        self.results[op.target_key or "json"] = obj
                        if self._is_recording():
                            self._append_example(("assistant", obj.json() if obj else ""))
                    elif op.kind == OpKind.FUNCTION_CALL:
                        self.results[op.target_key] = op.delegate(self.results[op.source_key])
                    else:
                        raise ValueError(f"Unhandled Agent operation kind: {op.kind}")
                # loop completed, all good, terminate in finally
                break
            except Exception as e:
                cnt += 1
                logger.error(f"AgentBase for {self.name} failed for the {cnt} time: {e}")
            finally:
                q_from_user.put(DuplexSignalTerminate())
                if thr:
                    thr.join()
                with q_from_user.mutex:
                    q_from_user.queue.clear()
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
        self.agent_manager.save_example_to_db(
            self.name, get_real_datetime(), "callstack_unavailable", 0.0,
            prompt_repr="\n".join(f"{r}: {c}" for r, c in ex),
            rating=rating, reason=reason
        )
