from enum import auto, StrEnum

from pm.data_structures import KnoxelHaver


class ControlVariables(StrEnum):
    HasInput = auto()

class GhostCodelets(KnoxelHaver):
    pass

class ShellBase:
    pass

class BaseProc:
    @staticmethod
    def cognitive_cycle(ghost: GhostCodelets, shell: ShellBase):
        # Ensure the ghost is up to date
        BaseProc.load(ghost, shell)
        # Create the ghost state from the previous state, or init a new character instance with a first state
        BaseProc.init_cycle(ghost)
        # Read all user inputs, world happenings, thoughts from the buffer
        BaseProc.read_initial_buffer(ghost, shell)
        if not ghost.get_bool(ControlVariables.HasInput):
            return

        # Pre-fill the ghost state of this tick with important structures such as current mental state, whose latent state needs to be determined dynamically
        BaseProc.update_dynamic_states(ghost)

        # Load the previous Current Situational Model or build a new one with algorithmically chosen weights
        CsmProc.prepare_csm(ghost)
        # Decay valence of current state.
        CsmProc.decay(ghost)
        # Get a latent concept feature from the current context. The latest N features contributing to the latest theme are selected
        CsmProc.update_situation(ghost)

        # Gather relevant memories to the current situation with query expansion
        PamCodeletProc.query_memories_situation(ghost)
        PamCodeletProc.query_memories_mental_state(ghost)
        # Merge new memories with memories already in CSM, boosting memories alredy existing and replacing memories that become less relevant
        PamCodeletProc.update_memories(ghost)

        # Build latent prompts for all kinds of future codelets for different context sizes
        CsmProc.build_latent_prompts(ghost)

        # Triage the current situation/input to determine if the input allows "fast path" completion with previous state, or world agent needs to wait for CCQ udpate
        # With fast pathing less important input can be quickly replied to, while also generating less mental features asynchronally for narrative context in future promtps and personality development
        TriageProc.triage_stimulus(ghost)
        # Send to listening threads
        BaseProc.update_triage(ghost, shell)

        # PolicyDynamics: Create intentions of short term timeframe. Mostly about the current situation, dialoge, train of thought.
        # Use embeddings and simple logic to merge duplicate Intentions, creating new ones and setting the old ones to non-causal
        PolicyDynamics.preprocess_intentions()
        # Execute the codelets responsible for checking if intentions, expectations or aversions are fulfilled by new input and appraising the deltas
        PolicyDynamics.execute_intention_codelets()

        # Determine what kind of appraisal or feature generating codelets need to be run based on the input.
        # Appraisal, coping strategies, intention management, fantasy, etc. There are like 400 codlets already to spice the mental landscape up with weirdly specific mental features.
        CodeletProc.determine_codelet_families(ghost)
        # Pre-filter the available features from the CSM. For example, if the user asks a question how the AI feels, introspective features (interlocus) will be favored over world facts.
        CodeletProc.pre_filter_csm(ghost)
        # Execute the codelets. Add the resulting features to CSM (not causal yet), boost codelet paths, get context from codelet-specific vector-search + preceding codelet features.
        CodeletProc.execute_codelets(ghost)

        # Merge new intentions, if any were produced.
        PolicyDynamics.postprocess_intentions()

        # Remove old features from CSM whose valence falls below a threshold to limit data for possible coalitions.
        AttentionProc.purge_features(ghost)
        # Build coaltions by clustering relevant features around intentions, or common themes and keywords if there are no high-valence intentions.
        AttentionProc.build_coalitions(ghost)

        # Simulations: Create intentions of medium term timeframe. About the general direction of the situation.
        # Based on state and coalition, build a steering vector for the simulations.
        # Use current state (valence, arousal, dominance) and intention to create a simulation-prompt where the story writing agent is instructed to steer towards bad, neutral or good outcomes for the AI character.
        SimulationProc.determine_steering_vector(ghost)
        # Simulate the world (user, virtual world) if not in loopback-logic. Has simultion for theory-of-mind or general world events. Rules about world are updated during learning, improving future predictions.
        SimulationProc.simulate_world(ghost)
        # Simulate self with more focus on internal features (goals) over external features (user). Simulation includes failing, stalling and reaching personal goals/aversions
        SimulationProc.simulate_self(ghost)
        # Simulation with strong focus on self-model + actual capabilities. To filter coalitions that attempt to do something that narratively makes sense, but doesn't in this architecture (controlling a robot, using a webcam to see, etc)
        SimulationProc.simulate_meta(ghost)
        # Use self-model to modify weights (valence, salience) to simulate "conscious" control against unconscious control. Like use self model to decide when to focus on bad or good outcomes.
        QualiaProc.apply_self_model_simulation(ghost)
        # Combine all weights, select the best one or maybe even the top 2 conflicting ones.
        SimulationProc.select_simulation(ghost)

        # Combine codelet coaltion valence + simulation valence and all that.
        AttentionProc.rate_coalitions(ghost)
        # Modify the weights based on self model slightly.
        QualiaProc.apply_self_model_attention(ghost)
        # Select coaltions to make it to the workspace.
        AttentionProc.select_coalitions(ghost)

        # Generate subjective representation of the current situation based on all the internal features, in a compressed form.
        # At this state its pretty much about how the characters manages their attention.
        QualiaProc.generate_subjective_experience(ghost)
        # Retrieve similar, past experiences.
        QualiaProc.match_subjective_experience(ghost)
        # Instruct a "robot-psychologist" agent to reconstruct the basis behind the latent features to create a model of the ai character. Try to steer it into the right direction, but not too much. Use delta to form new rules.
        QualiaProc.update_self_model_attention(ghost)

        # Update GWT by removing less-saleint features and replacing it with the coaltions content.
        WorkspaceProc.integrate_coalitions(ghost)
        # Update causality of the added features.
        WorkspaceProc.update_causality(ghost)
        # Generate different prompts for all listeners based on new CCQ.
        WorkspaceProc.generate_ccq(ghost)

        # Check out what past behaviors could be reused "answer helpfully" "try to call tools" "be passive aggresive with user".
        # This is a general steering behavior for the world thread prompt, nothing too concrete.
        ActionSelectionProc.create_possible_behaviors()
        # Modify weights slightly
        QualiaProc.apply_self_model_action(ghost)
        # Select general behavior.
        ActionSelectionProc.get_behavior()

        # Describe the course of action in a compressed form.
        QualiaProc.generate_subjective_action_descision(ghost)
        # Instruct a "robot-psychologist" agent to reconstruct WHY the character decided what to do.
        QualiaProc.update_self_model(ghost)

        # Send CCQ and behaviors to listening threads.
        # Based on behavior some threads become more or less active. If a loopback-thought is initiated, the world thread prompt might be instructed to "make the ai tell the user they're processing a thought".
        # The CCQ is a container for all kinds of different prompts for different threads.
        # World Thread: Talk to user, many unconscious features, behaviors and dialog to make the story really immersive where answers are really based on the internal description.
        # Thought Thread: Almost all of this is composed of the subjective experiences. By managing attention through tool calls, interlocus to actual inner mechanics of its operation might be gained.
        # This is a free-writing extension of the qualia system, and its results might be referenced by world thread if asked, for reporting its subjective state.
        BaseProc.update_closing_buffer(ghost, shell)
        BaseProc.update_ccq(ghost, shell)

        # Some procedures for creating new rules, cleaning up, etc.
        AttentionProc.learn_attention_behavior(ghost)
        QualiaProc.learn_self_model(ghost)
        PolicyDynamics.learn_expectation_outcome(ghost)
        SimulationProc.learn_simulation_result(ghost)

        # Memory procs for hierarchical summarized memory.
        MemoryProc.consolidate_memory(ghost)
        MemoryProc.update_narratives(ghost)
        # Optimize prompts or train a LORA (optional)
        OptimizationProc.improve_prompts(ghost, shell)
        OptimizationProc.learn_lora(ghost, shell)

        # Dump JSON of Ghost
        BaseProc.save(ghost, shell)




