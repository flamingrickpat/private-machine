"""
Simulation Codelets.
Dedicated codelet runners for simulation-specific tasks:
  1. Consequence Simulation — "What happens if I do X?"
  2. Emotional Forecast — "How will I feel after doing X?"
  3. Social Impact — "How will user perceive X?"

These are distinct from the general-purpose simulators (SimWorld/SimSelf/SimMeta)
because they leverage the CodeletRegistry for context-aware activation and
produce SimulationCodeletResult percepts rather than simple prediction strings.
"""
import logging
from typing import List, Optional

from pm.ghosts.llm_contracts import call_tool_with_contract
from pm.ghosts.procedures.base import GhostProtocol
from pm.llm.llm_common import LlmPreset
from pm.ghosts.schemas import SimulationCodeletResult
from pm.ghosts.prompts import (
    SIM_CODELET_EXAMPLE_TURNS,
    SIM_CODELET_CONSEQUENCE_SYSTEM, SIM_CODELET_CONSEQUENCE_USER,
    SIM_CODELET_EMOTIONAL_SYSTEM, SIM_CODELET_EMOTIONAL_USER,
    SIM_CODELET_SOCIAL_SYSTEM, SIM_CODELET_SOCIAL_USER,
)
from pm.data_structures import Feature, FeatureType

logger = logging.getLogger(__name__)


class SimulationCodeletRunner:
    """
    Runs simulation-specific codelets that augment the core simulators.
    Each method represents a type of mental simulation the agent performs
    before committing to an action.
    """

    @staticmethod
    def run_all(ghost: GhostProtocol, proposed_action: str) -> List[SimulationCodeletResult]:
        """
        Run all simulation codelets for a proposed action and return aggregated results.
        """
        results: List[SimulationCodeletResult] = []

        # Build shared context
        context = SimulationCodeletRunner._build_context(ghost)
        mental_state_summary = SimulationCodeletRunner._mental_state_summary(ghost)

        # 1. Consequence Simulation
        consequence = SimulationCodeletRunner._run_consequence(
            ghost, proposed_action, context, mental_state_summary
        )
        if consequence:
            results.append(consequence)

        # 2. Emotional Forecast
        emotional = SimulationCodeletRunner._run_emotional_forecast(
            ghost, proposed_action, context
        )
        if emotional:
            results.append(emotional)

        # 3. Social Impact
        social = SimulationCodeletRunner._run_social_impact(
            ghost, proposed_action, context
        )
        if social:
            results.append(social)

        # Store results as Features in CSM
        for result in results:
            f = Feature(
                content=f"[Sim:{result.scenario}] {result.predicted_outcome} "
                        f"(risk={result.risk:.2f}, benefit={result.benefit:.2f}, "
                        f"feeling={result.emotional_forecast})",
                feature_type=FeatureType.CodeletOutput,
                source="SimulationCodelet",
                causal=False,
            )
            ghost.add_knoxel(f)
            if hasattr(ghost, 'csm_manager'):
                from pm.csm.csm import CSMItem
                ghost.csm_manager.add_or_boost(CSMItem(
                    knoxel_id=f.id,
                    first_tick=ghost.current_tick_id,
                    last_tick=ghost.current_tick_id,
                    activation=0.8,
                ))

        logger.info(f"SimulationCodelets: Ran {len(results)} simulations for action: '{proposed_action[:60]}'")
        return results

    @staticmethod
    def _build_context(ghost: GhostProtocol) -> str:
        """Build a compact context string from CSM."""
        if not hasattr(ghost, 'csm_manager'):
            return "(no context)"
        items = ghost.csm_manager.get_active_items(min_activation=0.3)
        items.sort(key=lambda x: x.activation, reverse=True)
        lines = []
        for item in items[:8]:
            k = ghost.get_knoxel_by_id(item.knoxel_id)
            if k:
                lines.append(f"[{int(item.activation * 100)}] {k.content[:120]}")
        return "\n".join(lines) if lines else "(empty)"

    @staticmethod
    def _mental_state_summary(ghost: GhostProtocol) -> str:
        """Compact mental state summary."""
        if not ghost.current_state or not ghost.current_state.latent_mental_state:
            return "(unknown)"
        ms = ghost.current_state.latent_mental_state
        parts = []
        if ms.state_core:
            parts.append(f"V:{ms.state_core.valence:.2f} A:{ms.state_core.arousal:.2f}")
        if ms.state_neurochemical:
            parts.append(f"DA:{ms.state_neurochemical.dopamine:.2f} CORT:{ms.state_neurochemical.cortisol:.2f}")
        return ", ".join(parts) if parts else "(unknown)"

    @staticmethod
    def _run_consequence(
        ghost: GhostProtocol, action: str, context: str, mental_state: str
    ) -> Optional[SimulationCodeletResult]:
        try:
            model, _ = call_tool_with_contract(
                ghost,
                phase="sim_codelet_consequence",
                schema=SimulationCodeletResult,
                system_prompt=SIM_CODELET_CONSEQUENCE_SYSTEM,
                user_prompt=SIM_CODELET_CONSEQUENCE_USER.format(
                    action=action, context=context, mental_state=mental_state
                ),
                examples=SIM_CODELET_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
            )
            if model is not None:
                return model
        except Exception as e:
            logger.error(f"SimCodelet consequence failed: {e}")
        return None

    @staticmethod
    def _run_emotional_forecast(
        ghost: GhostProtocol, action: str, context: str
    ) -> Optional[SimulationCodeletResult]:
        valence = 0.0
        arousal = 0.0
        if ghost.current_state and ghost.current_state.latent_mental_state:
            core = ghost.current_state.latent_mental_state.state_core
            if core:
                valence = core.valence
                arousal = core.arousal

        try:
            model, _ = call_tool_with_contract(
                ghost,
                phase="sim_codelet_emotional",
                schema=SimulationCodeletResult,
                system_prompt=SIM_CODELET_EMOTIONAL_SYSTEM,
                user_prompt=SIM_CODELET_EMOTIONAL_USER.format(
                    action=action, valence=valence, arousal=arousal, context=context
                ),
                examples=SIM_CODELET_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
            )
            if model is not None:
                return model
        except Exception as e:
            logger.error(f"SimCodelet emotional forecast failed: {e}")
        return None

    @staticmethod
    def _run_social_impact(
        ghost: GhostProtocol, action: str, context: str
    ) -> Optional[SimulationCodeletResult]:
        # Build relationship context from self-model or narratives
        relationship = "(standard relationship)"
        if hasattr(ghost, 'self_model') and ghost.self_model:
            bio = ghost.self_model.biography or ""
            relationship = bio[-200:] if len(bio) > 200 else bio
            if not relationship:
                relationship = "(no relationship history)"

        try:
            model, _ = call_tool_with_contract(
                ghost,
                phase="sim_codelet_social",
                schema=SimulationCodeletResult,
                system_prompt=SIM_CODELET_SOCIAL_SYSTEM,
                user_prompt=SIM_CODELET_SOCIAL_USER.format(
                    action=action, relationship=relationship, context=context
                ),
                examples=SIM_CODELET_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
            )
            if model is not None:
                return model
        except Exception as e:
            logger.error(f"SimCodelet social impact failed: {e}")
        return None
