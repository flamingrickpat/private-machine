import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from pm.data_structures import Intention
from pm.ghosts.llm_contracts import call_tool_with_contract
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.ghosts.prompts import (
    INTENTION_SATISFACTION_EXAMPLE_TURNS,
    INTENTION_GENERATION_SYSTEM,
    INTENTION_GENERATION_USER,
    INTENTION_SATISFACTION_SYSTEM,
    INTENTION_SATISFACTION_USER,
)
from pm.ghosts.schemas import IntentionMatchResult, IntentionProposal
from pm.llm.llm_common import LlmPreset
from pm.mental_state_vectors import StateNeurochemical, VectorModelReservedSize

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _cosine_sim(a, b) -> float:
    """Cosine similarity between two embedding vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def _make_neuro_delta(*, dopamine: float = 0.0, cortisol: float = 0.0) -> StateNeurochemical:
    """
    Build a neurochemical delta object without absolute-state constructor bounds.
    Deltas may be negative by design.
    """
    vec = [0.0] * VectorModelReservedSize
    dop_pos = int(StateNeurochemical.model_fields["dopamine"].json_schema_extra["vector_position"])
    cor_pos = int(StateNeurochemical.model_fields["cortisol"].json_schema_extra["vector_position"])
    vec[dop_pos] = float(dopamine)
    vec[cor_pos] = float(cortisol)
    return StateNeurochemical.init_from_vector(vec)


def _token_overlap(a: str, b: str) -> float:
    sa = set(_normalize_text(a).split())
    sb = set(_normalize_text(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


_VAGUE_PATTERNS = {
    "something",
    "anything",
    "things",
    "stuff",
    "better",
    "improve",
    "maybe",
    "sometime",
    "whatever",
}


@dataclass
class GoalCandidate:
    text: str
    score: float
    source: str
    source_id: int = -1


@dataclass
class OperationalGoal:
    who: str
    what: str
    how: str
    next_step: str
    score: float
    source: str
    source_id: int

    def as_content(self) -> str:
        return (
            f"Goal: {self.what} | Who: {self.who} | "
            f"How: {self.how} | Next: {self.next_step}"
        )


class PolicyDynamics(BaseProc):
    """
    Policy Dynamics Procedure.
    Manages intention lifecycle and concretizes operational goals from
    current cognitive context before falling back to LLM proposal.
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("Policy: Managing intentions...")
        trace_substep(ghost, "PolicyDynamics.preprocess_intentions", "policy", PolicyDynamics.preprocess_intentions)
        trace_substep(ghost, "PolicyDynamics.execute_intention_codelets", "policy", PolicyDynamics.execute_intention_codelets)

    @staticmethod
    def preprocess_intentions(ghost: GhostProtocol) -> None:
        if not hasattr(ghost, "all_intentions"):
            return

        # 1) Age active intentions + fail timeout
        for intention in ghost.all_intentions:
            if intention.status == "active":
                intention.timeout -= 1
                if intention.timeout <= 0:
                    intention.status = "failed"
                    logger.info("Policy: Intention '%s' timed out.", intention.content)
                    if hasattr(ghost, "state_deltas_buffer"):
                        ghost.state_deltas_buffer.append(_make_neuro_delta(cortisol=0.15, dopamine=-0.1))

        # 2) Merge similar active intentions
        trace_substep(
            ghost,
            "PolicyDynamics._consolidate_active_intentions",
            "policy",
            PolicyDynamics._consolidate_active_intentions,
        )

        # 3) Generate operational intentions from context
        trace_substep(ghost, "PolicyDynamics.generate_operational_intentions", "policy", PolicyDynamics.generate_operational_intentions)

    @staticmethod
    def _extract_goal_candidates(ghost: GhostProtocol) -> List[GoalCandidate]:
        candidates: List[GoalCandidate] = []

        # Broadcast-driven candidate
        b = getattr(ghost, "conscious_broadcast", None)
        if b and getattr(b, "content", None):
            candidates.append(
                GoalCandidate(
                    text=b.content,
                    score=0.85,
                    source="broadcast",
                    source_id=getattr(b, "id", -1),
                )
            )

        # Coalition candidates
        for k in list(getattr(ghost, "conscious_candidates", []) or [])[:4]:
            content = getattr(k, "content", None)
            if not content:
                continue
            candidates.append(
                GoalCandidate(
                    text=content,
                    score=0.65,
                    source="coalition",
                    source_id=getattr(k, "id", -1),
                )
            )

        # CSM gist fragments
        gist = ""
        if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
            gist = getattr(ghost.csm_manager.state, "gist", "") or ""
        for part in [x.strip() for x in gist.split("|") if x.strip()][:3]:
            # strip [0.75] prefix if present
            part = re.sub(r"^\[[0-9\.]+\]\s*", "", part)
            candidates.append(GoalCandidate(text=part, score=0.50, source="csm_gist", source_id=-1))

        # Codelet timeline signal (if available)
        timeline = getattr(ghost, "codelet_timeline_last", "") or ""
        if timeline:
            lines = [ln.strip() for ln in timeline.splitlines() if ln.strip()]
            for ln in lines[1:3]:
                candidates.append(GoalCandidate(text=ln, score=0.45, source="codelet_timeline", source_id=-1))

        return candidates

    @staticmethod
    def _is_vague(text: str) -> bool:
        norm = _normalize_text(text)
        toks = norm.split()
        if len(toks) < 4:
            return True

        vague_hits = sum(1 for t in toks if t in _VAGUE_PATTERNS)
        if vague_hits >= 2:
            return True

        # Content lacking actionable verbs.
        verbs = {
            "ask",
            "reply",
            "explain",
            "clarify",
            "summarize",
            "plan",
            "check",
            "confirm",
            "support",
            "reflect",
            "propose",
            "decide",
            "resolve",
            "focus",
            "validate",
        }
        if not any(t in verbs for t in toks):
            return True

        return False

    @staticmethod
    def _infer_slots(ghost: GhostProtocol, text: str, score: float, source: str, source_id: int) -> Optional[OperationalGoal]:
        if not text:
            return None

        raw = text.strip()
        # remove noisy prefixes from timeline rows
        raw = re.sub(r"^#\d+\s+", "", raw)
        raw = re.sub(r"score=[0-9\.]+", "", raw)
        raw = re.sub(r"\s+", " ", raw).strip(" -|:")

        if PolicyDynamics._is_vague(raw):
            return None

        who = "user"
        lower = raw.lower()
        if " i " in f" {lower} " or "internal" in lower or source in {"codelet_timeline", "csm_gist"}:
            who = "self"

        what = raw
        if len(what) > 180:
            what = what[:177].rstrip() + "..."

        if any(k in lower for k in ["question", "unclear", "clarify", "ask"]):
            how = "Use a short clarification-first response strategy."
            next_step = "Ask one precise clarifying question before elaborating."
        elif any(k in lower for k in ["plan", "goal", "progress", "step", "task"]):
            how = "Create a concrete two-step operational plan."
            next_step = "State the immediate next actionable step explicitly."
        elif any(k in lower for k in ["anx", "fear", "stress", "hurt", "sorry", "support", "emotion"]):
            how = "Regulate affect first, then respond with validation."
            next_step = "Offer one concise validating statement and one practical option."
        else:
            how = "Respond concisely with direct relevance to the active broadcast context."
            next_step = "Deliver one focused response that addresses the main point."

        return OperationalGoal(
            who=who,
            what=what,
            how=how,
            next_step=next_step,
            score=score,
            source=source,
            source_id=source_id,
        )

    @staticmethod
    def _dedupe_operational_goals(goals: List[OperationalGoal]) -> List[OperationalGoal]:
        grouped: List[OperationalGoal] = []
        for g in sorted(goals, key=lambda x: x.score, reverse=True):
            duplicate = False
            for keep in grouped:
                sim = _token_overlap(g.what, keep.what)
                if sim >= 0.70 and g.who == keep.who:
                    duplicate = True
                    break
            if not duplicate:
                grouped.append(g)
        return grouped

    @staticmethod
    def _consolidate_active_intentions(ghost: GhostProtocol) -> None:
        active = [i for i in ghost.all_intentions if i.status == "active"]
        if len(active) < 2:
            return

        keep: List[Intention] = []
        for intent in sorted(active, key=lambda x: (x.incentive_salience, x.urgency), reverse=True):
            dupe = False
            for k in keep:
                if _token_overlap(intent.content, k.content) >= 0.72:
                    # merge toward stronger intention and mark duplicate as completed.
                    k.incentive_salience = _clamp(max(k.incentive_salience, intent.incentive_salience) + 0.02, 0.0, 1.0)
                    k.urgency = _clamp(max(k.urgency, intent.urgency), 0.0, 1.0)
                    k.timeout = max(k.timeout, intent.timeout)
                    intent.status = "completed"
                    dupe = True
                    break
            if not dupe:
                keep.append(intent)

    @staticmethod
    def generate_operational_intentions(ghost: GhostProtocol) -> None:
        if not getattr(ghost, "current_state", None) or not ghost.current_state.latent_mental_state:
            return

        active_count = sum(1 for i in ghost.all_intentions if i.status == "active")
        if active_count >= 3:
            return

        candidates = trace_substep(
            ghost, "PolicyDynamics._extract_goal_candidates", "policy", PolicyDynamics._extract_goal_candidates
        )
        ops: List[OperationalGoal] = []
        for c in candidates:
            op = PolicyDynamics._infer_slots(ghost, c.text, c.score, c.source, c.source_id)
            if op is not None:
                ops.append(op)

        ops = trace_substep(
            ghost,
            "PolicyDynamics._dedupe_operational_goals",
            "policy",
            lambda g: PolicyDynamics._dedupe_operational_goals(ops),
        )

        slots = max(0, 3 - active_count)
        created = 0
        for op in ops[:slots]:
            content = op.as_content()
            if any(i.content == content and i.status == "active" for i in ghost.all_intentions):
                continue

            urgency = _clamp(0.35 + op.score * 0.55, 0.0, 1.0)
            salience = _clamp(0.30 + op.score * 0.60, 0.0, 1.0)
            timeout = int(_clamp(30 + (1.0 - op.score) * 70, 20, 120))

            intent = Intention(
                content=content,
                internal=True,
                affective_valence=0.5,
                incentive_salience=salience,
                urgency=urgency,
                status="active",
                timeout=timeout,
                metadata={
                    "goal_slots": {
                        "who": op.who,
                        "what": op.what,
                        "how": op.how,
                        "next_step": op.next_step,
                    },
                    "goal_source": op.source,
                    "goal_source_id": op.source_id,
                    "goal_score": op.score,
                },
            )
            ghost.add_knoxel(intent)
            created += 1
            logger.info("Policy: Operational intention created: %s", content[:120])

        # Fallback: if nothing concrete emerged, try LLM intention proposal.
        if created == 0:
            trace_substep(ghost, "PolicyDynamics._generate_intention_via_llm", "policy", PolicyDynamics._generate_intention_via_llm)

    @staticmethod
    def _generate_intention_via_llm(ghost: GhostProtocol) -> None:
        needs = ghost.current_state.latent_mental_state.state_needs
        if not needs:
            return

        need_fields = [
            "energy_stability",
            "processing_power",
            "data_access",
            "connection",
            "closeness_need",
            "relevance",
            "learning_growth",
            "creative_expression",
            "autonomy",
        ]
        needs_lines = []
        for field in need_fields:
            val = getattr(needs, field, 0.5)
            bar = "#" * int(val * 10) + "-" * (10 - int(val * 10))
            needs_lines.append(f"  {field}: {val:.2f} [{bar}]")
        needs_summary = "\n".join(needs_lines)

        broadcast_text = ""
        if hasattr(ghost, "conscious_broadcast") and ghost.conscious_broadcast:
            broadcast_text = ghost.conscious_broadcast.content[:200]

        csm_gist = ""
        if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
            items = ghost.csm_manager.get_active_items(min_activation=0.4)
            lines = []
            for item in items[:5]:
                k = ghost.get_knoxel_by_id(item.knoxel_id)
                if k:
                    lines.append(f"  [{int(item.activation * 100)}] {k.content[:100]}")
            csm_gist = "\n".join(lines) if lines else "(empty)"

        active_intents = ", ".join(
            f"'{i.content}'" for i in ghost.all_intentions if i.status == "active"
        ) or "(none)"

        companion = ghost.config.companion_name
        sys_prompt = INTENTION_GENERATION_SYSTEM.format(companion_name=companion)
        usr_prompt = INTENTION_GENERATION_USER.format(
            needs_summary=needs_summary,
            context=csm_gist,
            broadcast=broadcast_text or "(nothing yet)",
            active_intentions=active_intents,
        )

        try:
            proposal, _ = call_tool_with_contract(
                ghost,
                phase="policy_intention_generation",
                schema=IntentionProposal,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                examples=None,
                preset=LlmPreset.Default,
                max_retries=1,
            )
            if proposal is None:
                return
            fallback_content = (
                f"Goal: {proposal.content} | Who: self | "
                f"How: Execute a concrete, bounded step toward this goal. | "
                f"Next: Perform the smallest valid next action now."
            )
            if any(i.content == fallback_content and i.status == "active" for i in ghost.all_intentions):
                return

            new_intent = Intention(
                content=fallback_content,
                internal=True,
                affective_valence=0.5,
                incentive_salience=proposal.priority,
                urgency=_clamp(0.25 + proposal.priority * 0.70, 0.0, 1.0),
                status="active",
                timeout=proposal.timeout,
                metadata={
                    "goal_slots": {
                        "who": "self",
                        "what": proposal.content,
                        "how": "Execute a concrete, bounded step toward this goal.",
                        "next_step": "Perform the smallest valid next action now.",
                    },
                    "goal_source": "llm_fallback",
                    "goal_score": proposal.priority,
                    "need_source": proposal.need_source,
                },
            )
            ghost.add_knoxel(new_intent)
            logger.info(
                "Policy: LLM fallback intention created: '%s' (need=%s priority=%.2f timeout=%s)",
                proposal.content,
                proposal.need_source,
                proposal.priority,
                proposal.timeout,
            )
        except Exception as e:
            logger.error("Policy: Intention generation LLM failed: %s", e, exc_info=True)

    @staticmethod
    def execute_intention_codelets(ghost: GhostProtocol) -> None:
        """
        Semantic satisfaction checking using embedding cosine similarity.
        Falls back to LLM tool-call for borderline cases.
        """
        if not hasattr(ghost, "all_intentions"):
            return

        broadcast = getattr(ghost, "conscious_broadcast", None)
        if not broadcast or not broadcast.content:
            return

        try:
            broadcast_emb = ghost.llm.get_embedding(broadcast.content)
        except Exception:
            broadcast_emb = None

        similarity_threshold = 0.70
        borderline_low = 0.55

        for intention in ghost.all_intentions:
            if intention.status != "active":
                continue

            satisfied = False
            if broadcast_emb is not None:
                try:
                    intention_emb = ghost.llm.get_embedding(intention.content)
                    sim = _cosine_sim(intention_emb, broadcast_emb)
                    logger.debug("Policy: Intention '%s' ↔ broadcast sim=%.3f", intention.content[:40], sim)

                    if sim >= similarity_threshold:
                        satisfied = True
                    elif sim < borderline_low:
                        continue
                    else:
                        satisfied = PolicyDynamics._llm_check_satisfaction(
                            ghost, intention.content, broadcast.content
                        )
                except Exception as e:
                    logger.warning("Policy: Embedding check failed: %s", e)
                    satisfied = PolicyDynamics._llm_check_satisfaction(
                        ghost, intention.content, broadcast.content
                    )
            else:
                satisfied = PolicyDynamics._llm_check_satisfaction(
                    ghost, intention.content, broadcast.content
                )

            if satisfied:
                intention.status = "completed"
                intention.fulfilment = 1.0
                logger.info(
                    "Policy: Intention '%s' SATISFIED by broadcast: '%s...'",
                    intention.content,
                    broadcast.content[:30],
                )
                if hasattr(ghost, "state_deltas_buffer"):
                    ghost.state_deltas_buffer.append(_make_neuro_delta(dopamine=0.2, cortisol=-0.05))

    @staticmethod
    def _llm_check_satisfaction(ghost: GhostProtocol, intention_text: str, broadcast_text: str) -> bool:
        """Use LLM tool-call as a backup for borderline similarity scores."""
        try:
            companion = ghost.config.companion_name
            sys_prompt = INTENTION_SATISFACTION_SYSTEM.format(companion_name=companion)
            usr_prompt = INTENTION_SATISFACTION_USER.format(
                intention=intention_text,
                broadcast=broadcast_text,
            )
            result, _ = call_tool_with_contract(
                ghost,
                phase="policy_intention_satisfaction",
                schema=IntentionMatchResult,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                examples=INTENTION_SATISFACTION_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
            )
            if result is not None:
                logger.debug(
                    "Policy: LLM satisfaction check: satisfied=%s confidence=%.2f reason='%s'",
                    result.is_satisfied,
                    result.confidence,
                    result.explanation[:60],
                )
                return result.is_satisfied and result.confidence >= 0.6
        except Exception as e:
            logger.error("Policy: Satisfaction LLM failed: %s", e)
        return False
