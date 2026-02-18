import logging
from typing import Optional
from pm.ghosts.llm_contracts import call_tool_with_contract
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.llm.llm_common import LlmPreset
from pm.data_structures import Feature, FeatureType
from pm.ghosts.schemas import BehaviorOutput
from pm.ghosts.reply_blueprints import choose_reply_blueprint
from pm.ghosts.persona_memory import collect_persona_signals
from pm.ghosts.agent_context import (
    AgentContextComposer,
    AgentContextConfig,
    AgentContextWeights,
    compute_agent_context_budget,
)
from pm.ghosts.capabilities import capability_context_text, enforce_behavior_capability
from pm.ghosts.prompts import (
    ACTION_SELECTION_EXAMPLE_TURNS,
    ACTION_SELECTION_SYSTEM, ACTION_SELECTION_USER,
    ACTION_SELECTION_USER_SIMPLE,
)

logger = logging.getLogger(__name__)


class ActionSelectionProc(BaseProc):
    """
    Action Selection Procedure.
    Responsible for choosing the next action based on the conscious broadcast,
    simulation predictions, and procedural memory.
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("Action Selection: choosing behavior...")

        # 1. Gather context from Conscious Broadcast
        broadcast_content = ""
        if hasattr(ghost, 'conscious_broadcast') and ghost.conscious_broadcast:
            broadcast_content = ghost.conscious_broadcast.content
            logger.info(f"Action Selection: Based on broadcast: {broadcast_content[:100]}...")
        else:
            broadcast_content = "No conscious content available."
            logger.warning("Action Selection: No conscious content.")

        # 2. Use simulation output generated earlier in this same tick.
        sim_prediction = ""
        try:
            sim_prediction = trace_substep(
                ghost,
                "ActionSelectionProc.sim_prediction_from_bundle",
                "action",
                ActionSelectionProc._sim_prediction_from_bundle,
                broadcast_content,
            )
            if sim_prediction:
                logger.info(f"Action Selection: Simulation prediction: {sim_prediction[:100]}...")
        except Exception as e:
            logger.warning(f"Action Selection: Simulation bundle pre-check failed: {e}")

        # 3. Build prompt using central templates
        companion = ghost.config.companion_name
        sys_prompt = ACTION_SELECTION_SYSTEM.format(companion_name=companion)
        sys_prompt += (
            "\nHard architecture capability boundary:\n"
            + capability_context_text(ghost)
            + "\nNever choose actions that violate unsupported capabilities."
        )
        persona_signals = trace_substep(
            ghost,
            "ActionSelectionProc.collect_persona_signals",
            "action",
            lambda g: collect_persona_signals(ghost, query=broadcast_content, limit=5),
        )
        prompt_budget = compute_agent_context_budget(
            ghost,
            output_tokens=420,
            min_budget=600,
        )
        context_packet = trace_substep(
            ghost,
            "ActionSelectionProc.build_agent_context",
            "action",
            lambda g: AgentContextComposer.build(
                ghost,
                focus_text="\n".join(
                    [
                        broadcast_content,
                        str(getattr(getattr(ghost, "primary_stimulus", None), "content", "") or ""),
                        str(getattr(ghost, "ego_directive", "") or ""),
                        str(getattr(ghost, "thought_blueprint_directive", "") or ""),
                    ]
                ),
                config=AgentContextConfig(
                    max_tokens=prompt_budget,
                    weights=AgentContextWeights.from_config(
                        dict(getattr(getattr(ghost, "config", None), "agent_context_weights", {}) or {})
                    ),
                    latest_messages=int(getattr(getattr(ghost, "config", None), "agent_context_latest_messages", 40) or 40),
                    workspace_items=int(getattr(getattr(ghost, "config", None), "agent_context_workspace_items", 48) or 48),
                    min_section_tokens=int(getattr(getattr(ghost, "config", None), "agent_context_min_section_tokens", 96) or 96),
                ),
                purpose="action_selection",
            ),
        )
        ego_directive = str(getattr(ghost, "ego_directive", "") or "").strip()
        ego_dissonance = float((getattr(ghost, "ego_decision_last", {}) or {}).get("dissonance", 0.0) or 0.0)
        thought_directive = str(getattr(ghost, "thought_blueprint_directive", "") or "").strip()
        thought_hints = list(getattr(ghost, "thought_blueprint_action_hints", []) or [])
        qualia_action_bias = float(getattr(ghost, "qualia_action_bias", 0.0) or 0.0)
        qualia_action_directive = str(getattr(ghost, "qualia_action_directive", "") or "").strip()
        token_cap = int(getattr(ghost, "generation_token_cap", 0) or 0)
        budget_hint = ""
        if token_cap > 0:
            budget_hint = f"\nResponse budget: keep final user-facing speech within about {token_cap} tokens."

        if sim_prediction:
            valence = 0.0
            arousal = 0.0
            if ghost.current_state and ghost.current_state.latent_mental_state:
                core = ghost.current_state.latent_mental_state.state_core
                if core:
                    valence = core.valence
                    arousal = core.arousal

            usr_prompt = ACTION_SELECTION_USER.format(
                broadcast=broadcast_content,
                sim_prediction=sim_prediction,
                valence=valence,
                arousal=arousal,
            )
        else:
            usr_prompt = ACTION_SELECTION_USER_SIMPLE.format(
                broadcast=broadcast_content,
            )
        if ego_directive:
            usr_prompt += (
                f"\nEgo Directive: {ego_directive}\n"
                f"Arbitration Dissonance: {ego_dissonance:.2f}\n"
                "Respect this directive unless it conflicts with safety."
            )
        if thought_directive:
            usr_prompt += (
                f"\nThought Blueprint Directive: {thought_directive}\n"
                f"Thought Action Hints: {','.join(thought_hints) or 'none'}\n"
                "Use this thought path as a primary planning signal."
            )
        if qualia_action_directive:
            usr_prompt += (
                f"\nQualia Action Directive: {qualia_action_directive}\n"
                f"Qualia Action Bias: {qualia_action_bias:.2f}\n"
            )
        if persona_signals:
            usr_prompt += (
                "\nPersisted Persona Quirks:\n- "
                + "\n- ".join(persona_signals)
                + "\nBias action style toward these learned tendencies unless context strongly conflicts."
            )
        if context_packet and getattr(context_packet, "text", ""):
            usr_prompt += (
                "\nRelevant lived history:\n"
                + str(getattr(context_packet, "text", "") or "")
            )
        usr_prompt += budget_hint
        usr_prompt += (
            "\nArchitecture Capability Profile:\n"
            + capability_context_text(ghost)
        )

        # 4. Call LLM
        try:
            behavior, _ = trace_substep(
                ghost,
                "ActionSelectionProc.call_tool_with_contract",
                "action",
                lambda g: call_tool_with_contract(
                ghost,
                phase="action_selection",
                schema=BehaviorOutput,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                examples=ACTION_SELECTION_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
                max_output_tokens=420,
                ),
            )
            if behavior is not None:
                behavior, capability_guard = enforce_behavior_capability(
                    ghost,
                    behavior,
                    user_request=str(getattr(getattr(ghost, "primary_stimulus", None), "content", "") or broadcast_content),
                )
                logger.info(f"Action Selected: {behavior.action_description}")

                # Store selected behavior schema for downstream reply generation.
                ghost.selected_action_schema = behavior
                blueprint = trace_substep(
                    ghost,
                    "ActionSelectionProc.choose_reply_blueprint",
                    "action",
                    lambda g: choose_reply_blueprint(
                        tick=int(getattr(ghost, "current_tick_id", 0) or 0),
                        behavior=behavior,
                        broadcast=broadcast_content,
                        sim_prediction=sim_prediction,
                        simulation_bundle=dict(getattr(ghost, "simulation_bundle_last", {}) or {}),
                        history=list(getattr(ghost, "reply_blueprint_history", []) or []),
                    ),
                )
                ghost.reply_blueprint_last = blueprint
                if not hasattr(ghost, "reply_blueprint_history"):
                    ghost.reply_blueprint_history = []
                ghost.reply_blueprint_history.append(
                    {
                        **blueprint,
                        "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
                        "action_description": str(getattr(behavior, "action_description", "") or ""),
                    }
                )
                if len(ghost.reply_blueprint_history) > 100:
                    ghost.reply_blueprint_history = ghost.reply_blueprint_history[-100:]

                # Create a Feature for the action
                f = Feature(
                    content=str(behavior.model_dump_json()),
                    feature_type=FeatureType.Action,
                    source="ActionProc",
                    causal=True,                    
                    metadata={
                        "ego_directive": ego_directive,
                        "arbitration_dissonance": ego_dissonance,
                        "thought_blueprint_directive": thought_directive,
                        "thought_blueprint_hints": thought_hints,
                        "qualia_action_bias": qualia_action_bias,
                        "qualia_action_directive": qualia_action_directive,
                        "sim_prediction": sim_prediction[:240],
                        "action_schema": True,
                        "reply_blueprint": blueprint,
                        "persona_signals": persona_signals,
                        "capability_guard": capability_guard or {},
                    },
                )
                ghost.add_knoxel(f)

                # Add to CSM
                if hasattr(ghost, 'csm_manager'):
                    from pm.csm.csm import CSMItem
                    ghost.csm_manager.add_or_boost(CSMItem(
                        knoxel_id=f.id,
                        first_tick=ghost.current_tick_id,
                        last_tick=ghost.current_tick_id,
                        activation=1.0,
                    ))

                return behavior
            else:
                logger.warning("Action Selection: LLM produced no output.")

        except Exception as e:
            logger.error(f"Action Selection LLM error: {e}", exc_info=True)

        return None

    @staticmethod
    def _sim_prediction_from_bundle(ghost: GhostProtocol, broadcast_content: str) -> str:
        bundle = getattr(ghost, "simulation_bundle_last_model", None)
        if bundle is None:
            raw = dict(getattr(ghost, "simulation_bundle_last", {}) or {})
            if raw:
                try:
                    from pm.ghosts.schemas import SimBundle

                    bundle = SimBundle.model_validate(raw)
                except Exception:
                    bundle = None

        if bundle is not None:
            winner = str(getattr(bundle, "winner_lens", "") or "")
            outcomes = list(getattr(bundle, "outcomes", []) or [])
            winner_obj = next((o for o in outcomes if str(getattr(o, "lens", "")) == winner), None)
            if winner_obj is None and outcomes:
                winner_obj = outcomes[0]
            if winner_obj is not None:
                repellers = list(getattr(winner_obj, "repellers", []) or [])
                return (
                    f"[{str(getattr(winner_obj, 'lens', '') or 'world')}] "
                    f"{str(getattr(winner_obj, 'summary', '') or '').strip()} "
                    f"(utility={float(getattr(winner_obj, 'utility', 0.0) or 0.0):.2f}, "
                    f"polarity={str(getattr(winner_obj, 'polarity', 'neutral') or 'neutral')}, "
                    f"risk_signals={','.join(repellers[:2]) or 'none'})"
                )

        # Fallback only when simulation phase data is absent.
        from pm.ghosts.procedures.simulations import SimulationProc

        return SimulationProc.predict_action_outcome(ghost, broadcast_content)
