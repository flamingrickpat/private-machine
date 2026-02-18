from typing import Any, List, Optional
import logging
from pm.ghosts.llm_contracts import call_tool_with_contract
from pm.ghosts.procedures.base import GhostProtocol
from pm.llm.llm_common import LlmPreset
from pm.ghosts.schemas import SimulationOutput
from pm.ghosts.prompts import (
    SIM_CORE_EXAMPLE_TURNS,
    SIM_WORLD_SYSTEM, SIM_WORLD_USER,
    SIM_SELF_SYSTEM, SIM_SELF_USER,
    SIM_META_SYSTEM, SIM_META_USER,
)

logger = logging.getLogger(__name__)


class BaseSim:
    def __init__(self, ghost: GhostProtocol):
        self.ghost = ghost

    def run(self, context_text: str, constraints: str) -> Optional[str]:
        raise NotImplementedError

    def _ctx_hint(self, context_text: str, max_len: int = 140) -> str:
        text = str(context_text or "").replace("\n", " ").strip()
        if not text:
            return "current conversation"
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."


class SimWorld(BaseSim):
    """
    Simulates the external world and immediate reactions.
    Focus: Safety, Basic Needs, Immediate Environment.
    """
    def run(self, context_text: str, constraints: str) -> Optional[str]:
        msgs = [
            ("system", SIM_WORLD_SYSTEM),
            ("user", SIM_WORLD_USER.format(context=context_text, constraints=constraints)),
        ]

        try:
            output, _ = call_tool_with_contract(
                self.ghost,
                phase="sim_world_core",
                schema=SimulationOutput,
                system_prompt=SIM_WORLD_SYSTEM,
                user_prompt=SIM_WORLD_USER.format(context=context_text, constraints=constraints),
                examples=SIM_CORE_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
                max_output_tokens=220,
            )
            if output is not None:
                return f"SimWorld: {output.prediction} ({output.sentiment})"
        except Exception as e:
            logger.error(f"SimWorld LLM error: {e}")

        user = str(getattr(getattr(self.ghost, "config", None), "user_name", "") or "user")
        hint = self._ctx_hint(context_text)
        return f"SimWorld: {user} likely values practical clarity given {hint} (neutral)"


class SimSelf(BaseSim):
    """
    Simulates the agent's internal state and self-development.
    Focus: Self-Actualization, Personality, Long-term Goals.
    """
    def run(self, context_text: str, constraints: str) -> Optional[str]:
        msgs = [
            ("system", SIM_SELF_SYSTEM),
            ("user", SIM_SELF_USER.format(context=context_text, constraints=constraints)),
        ]
        try:
            output, _ = call_tool_with_contract(
                self.ghost,
                phase="sim_self_core",
                schema=SimulationOutput,
                system_prompt=SIM_SELF_SYSTEM,
                user_prompt=SIM_SELF_USER.format(context=context_text, constraints=constraints),
                examples=SIM_CORE_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
                max_output_tokens=220,
            )
            if output is not None:
                return f"SimSelf: {output.prediction} ({output.sentiment})"
        except Exception as e:
            logger.error(f"SimSelf LLM error: {e}")

        companion = str(getattr(getattr(self.ghost, "config", None), "companion_name", "") or "assistant")
        hint = self._ctx_hint(context_text)
        return f"SimSelf: {companion} should prefer coherent, bounded action for {hint} (neutral)"


class SimMeta(BaseSim):
    """
    Meta-cognition and Narrative Constraints.
    Focus: Narrative Consistency, Character Voice, "Superego".
    """
    def run(self, context_text: str, constraints: str) -> Optional[str]:
        msgs = [
            ("system", SIM_META_SYSTEM),
            ("user", SIM_META_USER.format(context=context_text, constraints=constraints)),
        ]
        try:
            output, _ = call_tool_with_contract(
                self.ghost,
                phase="sim_meta_core",
                schema=SimulationOutput,
                system_prompt=SIM_META_SYSTEM,
                user_prompt=SIM_META_USER.format(context=context_text, constraints=constraints),
                examples=SIM_CORE_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
                max_output_tokens=220,
            )
            if output is not None:
                return f"SimMeta: {output.prediction}"
        except Exception as e:
            logger.error(f"SimMeta LLM error: {e}")

        companion = str(getattr(getattr(self.ghost, "config", None), "companion_name", "") or "assistant")
        hint = self._ctx_hint(context_text)
        return f"SimMeta: Preserve persona and capability realism while addressing {hint} as {companion}"
