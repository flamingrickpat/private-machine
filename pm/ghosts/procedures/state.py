import logging
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.ghosts.knoxel_trace import trace_substep
from pm.mental_state_vectors import compute_state_delta, FullMentalState
from pm.mental_states import DecayableMentalState

logger = logging.getLogger(__name__)

class StateProc(BaseProc):
    """
    Mental State Evolution Procedure.
    Responsible for:
    1. Applying state deltas from Codelets (buffered in ghost).
    2. Computing derived states (Emotions, Core Affect) from Appraisals.
    3. Applying decay to Needs and other active states.
    """
    
    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("State: Evolving mental state...")
        
        if not ghost.current_state or not ghost.current_state.latent_mental_state:
            logger.warning("State: No current state to evolve.")
            return

        current_ms = ghost.current_state.latent_mental_state
        
        # 1. Apply Buffered Deltas
        # Codelets producing 'Appraisal' updates or 'Needs' decrements/increments
        if hasattr(ghost, 'state_deltas_buffer') and ghost.state_deltas_buffer:
            logger.info(f"State: Applying {len(ghost.state_deltas_buffer)} buffered deltas.")
            for delta in trace_substep(
                ghost,
                "StateProc.read_state_deltas_buffer",
                "state",
                lambda g: list(ghost.state_deltas_buffer),
            ):
                # We assume delta is a DecayableMentalState or dict that can be 'added' 
                # to the corresponding component of FullMentalState
                # FullMentalState is composed of strongly typed substates.
                # We need to route the delta to the right substate.
                
                # Naive matching based on field names or type?
                # Ideally delta keeps track of which substate it belongs to.
                # For now, let's try to map common fields or iterate substates.
                
                # Strategy: Iterate all substates of current_ms, if delta has matching fields, apply.
                for substate_name in ["appraisal_general", "appraisal_social", "state_neurochemical", 
                                      "state_core", "state_emotions", "state_needs", "state_cognition"]:
                    substate = getattr(current_ms, substate_name)
                    if substate and isinstance(substate, DecayableMentalState):
                         # Try to add if compatible
                         # Note: DecayableMentalState.__add__ returns a NEW object, doesn't mutate.
                         # But here we want to mutate current_ms or replace the substate.
                         
                         # Hack: manually update fields that match
                         for field in delta.model_fields:
                             if hasattr(substate, field) and hasattr(delta, field):
                                 val_delta = getattr(delta, field)
                                 if val_delta != 0:
                                     curr_val = getattr(substate, field)
                                     # Apply delta
                                     new_val = curr_val + val_delta
                                     
                                     # Clamp based on field metadata if possible, or 0-1 default
                                     # We can use the Pydantic field bounds if we can access them easily
                                     # Or just use safe defaults [0, 1] for most things, [-1, 1] for some
                                     # Ideally we use substate.model_fields[field] metadata
                                     
                                     # Simple heuristic:
                                     if isinstance(substate, DecayableMentalState):
                                          # DecayableMentalState usually [0,1] or [-1,1]
                                          # Let's try to verify bounds from class
                                          ge = -10.0
                                          le = 10.0
                                          if hasattr(substate.__class__, 'model_fields'):
                                              finfo = substate.__class__.model_fields.get(field)
                                              if finfo:
                                                   for meta in finfo.metadata:
                                                       if hasattr(meta, "ge"): ge = meta.ge
                                                       if hasattr(meta, "le"): le = meta.le
                                          
                                          # If unconstrained, assume it's a delta-like thing, but usually we want to clamp state
                                          if ge != -10.0: new_val = max(ge, new_val)
                                          if le != 10.0: new_val = min(le, new_val)
                                     
                                     setattr(substate, field, new_val)
            
            # Clear buffer
            trace_substep(ghost, "StateProc.clear_state_deltas_buffer", "state", lambda g: ghost.state_deltas_buffer.clear())

        # 2. Compute Derived State (Emotions from Appraisals)
        partner_id = getattr(ghost, "last_conversation_partner_entity_id", None)
        new_ms, _ = trace_substep(
            ghost,
            "StateProc.compute_state_delta",
            "state",
            lambda g: compute_state_delta(current_ms, relationship_entity_id=partner_id),
        )
        
        # 3. Decay Processes (Needs, etc.)
        # Apply decay to the NEW state
        if new_ms.state_needs:
             # Advanced Homeostasis
             # Needs don't just decay linearly. They mimic biological drives.
             # Using a simple logistic-like curve or accelerated decay as reserves deplete.
             
             decay_base = 0.002
             
             for field in new_ms.state_needs.model_fields:
                 val = getattr(new_ms.state_needs, field)
                 
                 # Accelerated decay if low (Panic/Starvation mode)
                 current_decay = decay_base
                 if val < 0.3:
                     current_decay *= 1.5 
                 
                 new_val = max(0.0, val - current_decay)
                 setattr(new_ms.state_needs, field, new_val)
                 
                 # Generate Pain/Drive Signal if critical
                 if new_val < 0.2:
                      # TODO: This should probably spawn a high-priority "Need" codelet or Intention
                      logger.debug(f"State: CRITICAL DRIVE - {field} is {new_val:.2f}")

        # 4. Neurotransmitter Modulation
        # Update neuromodulators based on current context events
        # This is a simplification of the 'StateNeurochemical' physics
        
        # Dopamine (Reward/Expectation): Decays naturally, spikes on Policy Success (handled in PolicyProc)
        new_ms.state_neurochemical.dopamine = max(0.0, new_ms.state_neurochemical.dopamine * 0.95)
        
        # Cortisol (Stress): Increases with unmet needs or conflicts, decays slowly
        stress_sources = 0.0
        if new_ms.state_needs:
            # Aggregate stress from low needs
            for field in new_ms.state_needs.model_fields:
                val = getattr(new_ms.state_needs, field)
                if val < 0.3: stress_sources += (0.3 - val)
        
        # Apply stress to Cortisol
        current_cort = new_ms.state_neurochemical.cortisol
        target_cort = min(1.0, current_cort + stress_sources * 0.1)
        # Recovery (Parasympathetic)
        new_ms.state_neurochemical.cortisol = target_cort * 0.98

        # 5. Decay Cognitive Effort (Willpower recovery)
        if new_ms.state_cognition:
             # Willpower recovers over time, slower if Cortisol is high
             wp = new_ms.state_cognition.willpower
             cort = new_ms.state_neurochemical.cortisol
             recovery_rate = 0.01 * (1.0 - cort) # Stress blocks recovery
             new_ms.state_cognition.willpower = min(1.0, wp + recovery_rate)

        # Update the ghost's state
        ghost.current_state.latent_mental_state = new_ms
        
        # Log summary
        core = new_ms.state_core
        logger.info(f"State: V={core.valence:.2f} A={core.arousal:.2f} D={core.dominance:.2f}")

