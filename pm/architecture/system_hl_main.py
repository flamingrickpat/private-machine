def get_graph():
    graph = Flow([
        START,
        start_transaction,
        tasks_create,
        tasks_delegate,
        Select([
            Branch(lambda x: x["next_agent"] == "task_memory", [
                memory_cluster,
                memory_extract_facts
            ]),
            Branch(lambda x: x["next_agent"] == "task_no_sensation", [
                internal_thought_create_seed,
                internal_thought_contextualize_seed,
                internal_thought_prefilter,
                Select([
                    Branch(lambda x: x["next_agent"] == "thought_ok", [
                        internal_thought_add_sensation,
                        Goto(label="has_sensation")
                    ]),
                    Branch(lambda x: x["next_agent"] == "thought_not_ok", [END])
                ])
            ]),
            Branch(lambda x: x["next_agent"] == "task_idle", [END]),
            Branch(lambda x: x["next_agent"] == "task_sensation", Flow(label="has_sensation", flow=[
                sensation_filter_necessary,
                prepare_history_and_context,
                Select([
                    Branch(lambda x: x["next_agent"] == "sensation_filter_yes", Flow([
                        sensation_filter_yes,
                        sensation_gather_metadata,
                        Parallel([
                            sensation_meta_check,
                            sensation_relevance_check,
                            sensation_urgency_check,
                            sensation_task_relevance_check,
                            sensation_redundancy_check,
                            sensation_focus_check,
                            sensation_tool_impact_check
                        ]),
                        sensation_decide_if_response_necessary,
                        Select([
                            Branch(lambda x: x["next_agent"] == "response_necessary_yes", [
                                Goto(label="response_necessary")
                            ]),
                            Branch(lambda x: x["next_agent"] == "response_necessary_no", [
                                generate_thought_response_unnecssary,
                                sensation_postprocess,
                                END
                            ])
                        ])
                    ])),
                    Branch(lambda x: x["next_agent"] == "sensation_filter_yes", [END])
                ]),
                id_superego_check,
                Select([
                    Branch(lambda x: x["next_agent"] == "id_superego_check_necessary", Flow([
                        Flow(label="response_necessary", flow=[
                            prepare_gwt_workspace,
                            Flow(label="id", flow=[
                                Parallel([
                                    id_pleasure_seeking,
                                    id_pain_avoidance,
                                    id_drive_fulfilment,
                                    id_impulse,
                                    id_fantasy,
                                    id_emotional_resonance,
                                    id_empathy
                                ])
                            ]),
                            Flow(label="superego", flow=[
                                Parallel([
                                    superego_social_compliance,
                                    superego_reputiation_manager,
                                    superego_moral_judgement,
                                    superego_habit_reinforcement,
                                    superego_guid_self_correction,
                                    superego_self_image
                                ])
                            ]),
                            weighted_voting_scoring,
                            iterative_refinement,
                            meta_agent_meditation,
                            generate_gwt_prompt,
                        ])
                    ]))
                ]),
                ego_toolset_selection,
                Select([
                    Branch(lambda x: x["next_agent"] == "ego_metacongition", [
                        prepare_history_preconscious_third_person,
                        metacognitive_analysis,
                        decide_self_modification_necesary,
                        metacognition_act
                    ]),
                    Branch(lambda x: x["next_agent"] == "ego_acting", [
                        prepare_history_conscious_first_person,
                        decide_meta_acting_mode,
                        Select([
                            Branch(lambda x: x["next_agent"] == "ego_thought", [
                                create_counter_agent,
                                build_prompt_third_person,
                                generate_meta_analysis,
                                decide_optimization_necesary,
                                Select([
                                    Branch(lambda x: x["next_agent"] == "optimize_yes", [
                                        load_tool_set_metacognition,
                                        build_prompt_first_person,
                                        generate_tool_call
                                    ]),
                                    Branch(lambda x: x["next_agent"] == "optimize_no", [
                                        add_reason_no_optimize
                                    ])
                                ])
                            ]),
                            Branch(lambda x: x["next_agent"] == "ego_response", [
                                apply_conversational_agents,
                                build_prompt_third_person,
                                While(
                                    flow=[
                                        generate_response,
                                        evaluate_affect_compared_to_goal,
                                        evaluate_metacognitive
                                    ],
                                    break_condition=lambda x: x["response_ok"]
                                ),
                                send_response
                            ]),
                            Branch(lambda x: x["next_agent"] == "ego_tool_call", [
                                load_tool_set,
                                build_prompt_first_person,
                                generate_tool_call
                            ])
                        ])
                    ])
                ])
            ]))
        ])
    ])
    return graph


class Agent():
    def __init__(self):
        pass

    def set_role(self, param):
        self.role = param
        return self


class Bus():
    def __init__(self, level):
        pass


class AutomatedAgent:
    def __init__(self, **kwargs):
        pass


def make_graph2():
    bus_sensation = Bus(0)
    bus_m1 = Bus(-1)
    bus_m2 = Bus(-2)
    bus_m3 = Bus(-3)
    bus_1 = Bus(1)
    bus_2 = Bus(2)
    bus_3 = Bus(3)
    bus_action = Bus(4)
    bus_execution = Bus(5)

    l0_meta_main = AutomatedAgent(reactors_in=[bus_sensation], reactors_out=[bus_m1])
    l0_meta_main = AutomatedAgent(reactors_in=[bus_action], reactors_out=[bus_execution])

    l1_mission_meta = Agent()
    l1_mission_gen = Agent()
    l1_mission_pred = Agent()
    l1_mission_disc = Agent()

    l0_meta_main.add_subagents(l1_mission_meta)
    l1_mission_meta.add_subagents(l1_mission_gen, l1_mission_pred, l1_mission_disc)

    l2_strategy_meta = Agent()
    l2_strategy_gen = Agent()
    l2_strategy_pred = Agent()
    l2_strategy_disc = Agent()

    l1_mission_gen.add_subagents(l2_strategy_meta)
    l2_strategy_meta.add_subagents(l2_strategy_gen, l2_strategy_pred, l2_strategy_disc)

    l3_tactic_sensation_meta = Agent()
    l3_tactic_sensation_gen = Agent()
    l3_tactic_sensation_disc = Agent()
    l3_tactic_sensation_pred = Agent()

    l4_task_sensation_gather_metadata = Agent()
    l4_task_sensation_meta_check = Agent()
    l4_task_sensation_relevance_check = Agent()
    l4_task_sensation_urgency_check = Agent()
    l4_task_sensation_task_relevance_check = Agent()
    l4_task_sensation_redundancy_check = Agent()
    l4_task_sensation_focus_check = Agent()
    l4_task_sensation_tool_impact_chec = Agent()
    l4_task_sensation_decide_if_response_necessary = Agent()
    l4_task_generate_thought_response_unnecessary = Agent()
    l4_task_sensation_postprocess = Agent()

    l3_tactic_id_meta = Agent()
    l3_tactic_id_gen = Agent()
    l3_tactic_id_disc = Agent()
    l3_tactic_id_pred = Agent()

    l4_task_id_pleasure_seeking = Agent()
    l4_task_id_pain_avoidance = Agent()
    l4_task_id_drive_fulfilment = Agent()
    l4_task_id_impulse = Agent()
    l4_task_id_fantasy = Agent()
    l4_task_id_emotional_resonance = Agent()
    l4_task_id_empathy = Agent()

    l3_tactic_superego_meta = Agent()
    l3_tactic_superego_gen = Agent()
    l3_tactic_superego_disc = Agent()
    l3_tactic_superego_pred = Agent()

    l4_task_superego_social_compliance = Agent()
    l4_task_superego_reputation_manager = Agent()
    l4_task_superego_moral_judgement = Agent()
    l4_task_superego_habit_reinforcement = Agent()
    l4_task_superego_guid_self_correction = Agent()
    l4_task_superego_self_image = Agent()

    l3_tactic_ego_meta = Agent()
    l3_tactic_ego_gen = Agent()
    l3_tactic_ego_disc = Agent()
    l3_tactic_ego_pred = Agent()

    l4_task_ego_weighted_voting_scoring = Agent()
    l4_task_ego_iterative_refinement = Agent()
    l4_task_ego_meta_agent_meditation = Agent()
    l4_task_ego_generate_gwt_prompt = Agent()
    l4_task_ego_ego_toolset_selection = Agent()
    l4_task_ego_prepare_history_preconscious_third_person = Agent()
    l4_task_ego_metacognitive_analysis = Agent()
    l4_task_ego_decide_self_modification_necesary = Agent()
    l4_task_ego_metacognition_ac = Agent()
    l4_task_ego_prepare_history_conscious_first_person = Agent()
    l4_task_ego_decide_meta_acting_mode = Agent()
    l4_task_ego_create_counter_agent = Agent()
    l4_task_ego_build_prompt_third_person = Agent()
    l4_task_ego_generate_meta_analysis = Agent()
    l4_task_ego_decide_optimization_necesary = Agent()
    l4_task_ego_load_tool_set_metacognition = Agent()
    l4_task_ego_build_prompt_first_person = Agent()
    l4_task_ego_generate_tool_cal = Agent()
    l4_task_ego_add_reason_no_optimiz = Agent()
    l4_task_ego_apply_conversational_agents = Agent()
    l4_task_ego_build_prompt_third_person = Agent()
    l4_task_ego_generate_response = Agent()
    l4_task_ego_evaluate_affect_compared_to_goal = Agent()
    l4_task_ego_evaluate_metacognitiv = Agent()
    l4_task_ego_send_respons = Agent()
    l4_task_ego_load_tool_set = Agent()
    l4_task_ego_build_prompt_first_person = Agent()
    l4_task_ego_generate_tool_cal = Agent()


def make_graph3():
    # Define buses:
    #  - The numbering here is conceptual.
    #  - bus_sensation (layer 0) could carry raw sensations
    #  - negative buses might be metacognitive layers
    #  - positive buses might be subconscious/conscious layers
    #  - bus_action and bus_execution for action commands and execution feedback
    bus_sensation = Bus(id=0, name="bus_sensation")    # For raw sensory input
    bus_m1 = Bus(id=-1, name="bus_m1")                 # Metacognitive layer -1
    bus_m2 = Bus(id=-2, name="bus_m2")                 # Metacognitive layer -2
    bus_m3 = Bus(id=-3, name="bus_m3")                 # Metacognitive layer -3
    bus_1 = Bus(id=1, name="bus_1")                    # Subconscious layer 1
    bus_2 = Bus(id=2, name="bus_2")                    # Subconscious layer 2
    bus_3 = Bus(id=3, name="bus_3")                    # Subconscious layer 3
    bus_action = Bus(id=4, name="bus_action")          # Action commands
    bus_execution = Bus(id=5, name="bus_execution")    # Execution feedback and telemetry

    # The top-level meta agent (layer 0, possibly the "boss")
    # This agent oversees the main mission agents and decides when to finalize output.
    # It listens to sensations and execution feedback, and can communicate with meta-layers.
    l0_boss_agent = AutomatedAgent(
        name="l0_boss_agent",
        reactors_in=[bus_sensation, bus_action],   # Receives sensory input and action outcomes
        reactors_out=[bus_execution]               # Sends decisions downstream
    )
    l0_boss_agent.set_role("boss")  # Boss/GWT-like layer

    # Mission layer agents (layer 1): meta, generative, predictive, discriminatory
    l1_mission_meta = Agent(name="l1_mission_meta")
    l1_mission_meta.set_role("meta")
    # Generative (creates plans/ideas)
    l1_mission_gen = Agent(name="l1_mission_gen")
    l1_mission_gen.set_role("generative")
    # Predictive (for future reasoning or simulating outcomes)
    l1_mission_pred = Agent(name="l1_mission_pred")
    l1_mission_pred.set_role("predictive")
    # Discriminatory (evaluates mission feasibility, constraints)
    l1_mission_disc = Agent(name="l1_mission_disc")
    l1_mission_disc.set_role("discriminatory")

    # Connect mission agents to appropriate buses (for simplicity, assume mission uses bus_1)
    l1_mission_meta.connect_bus(bus_1)
    l1_mission_gen.connect_bus(bus_1)
    l1_mission_pred.connect_bus(bus_1)
    l1_mission_disc.connect_bus(bus_1)

    # Add subagents to the top boss agent
    # l0_boss_agent oversees the mission meta agent, which in turn manages the other mission agents
    l0_boss_agent.add_subagents(l1_mission_meta)
    l1_mission_meta.add_subagents(l1_mission_gen, l1_mission_pred, l1_mission_disc)

    # Strategy layer (layer 2): meta, gen, pred, disc
    l2_strategy_meta = Agent(name="l2_strategy_meta")
    l2_strategy_meta.set_role("meta")
    l2_strategy_gen = Agent(name="l2_strategy_gen")
    l2_strategy_gen.set_role("generative")
    l2_strategy_pred = Agent(name="l2_strategy_pred")
    l2_strategy_pred.set_role("predictive")
    l2_strategy_disc = Agent(name="l2_strategy_disc")
    l2_strategy_disc.set_role("discriminatory")

    # Connect strategy agents to bus_2
    l2_strategy_meta.connect_bus(bus_2)
    l2_strategy_gen.connect_bus(bus_2)
    l2_strategy_pred.connect_bus(bus_2)
    l2_strategy_disc.connect_bus(bus_2)

    # Mission generative agent can call strategy-level agents
    l1_mission_gen.add_subagents(l2_strategy_meta)
    l2_strategy_meta.add_subagents(l2_strategy_gen, l2_strategy_pred, l2_strategy_disc)

    # Tactic layer for sensation (layer 3): meta, gen, disc, pred
    # This layer deals with initial sensation filtering and checks
    l3_tactic_sensation_meta = Agent(name="l3_tactic_sensation_meta")
    l3_tactic_sensation_meta.set_role("meta")
    l3_tactic_sensation_gen = Agent(name="l3_tactic_sensation_gen")
    l3_tactic_sensation_gen.set_role("generative")
    l3_tactic_sensation_disc = Agent(name="l3_tactic_sensation_disc")
    l3_tactic_sensation_disc.set_role("discriminatory")
    l3_tactic_sensation_pred = Agent(name="l3_tactic_sensation_pred")
    l3_tactic_sensation_pred.set_role("predictive")

    # Connect tactic sensation agents to bus_3
    l3_tactic_sensation_meta.connect_bus(bus_3)
    l3_tactic_sensation_gen.connect_bus(bus_3)
    l3_tactic_sensation_disc.connect_bus(bus_3)
    l3_tactic_sensation_pred.connect_bus(bus_3)

    # Example of low-level tasks (layer 4) for sensation checks
    # These agents could be simple tools or API callers
    # Let's assume they connect to a specialized bus or reuse bus_3
    l4_task_sensation_gather_metadata = Agent(name="l4_task_sensation_gather_metadata").connect_bus(bus_3)
    l4_task_sensation_meta_check = Agent(name="l4_task_sensation_meta_check").connect_bus(bus_3)
    l4_task_sensation_relevance_check = Agent(name="l4_task_sensation_relevance_check").connect_bus(bus_3)
    l4_task_sensation_urgency_check = Agent(name="l4_task_sensation_urgency_check").connect_bus(bus_3)
    l4_task_sensation_task_relevance_check = Agent(name="l4_task_sensation_task_relevance_check").connect_bus(bus_3)
    l4_task_sensation_redundancy_check = Agent(name="l4_task_sensation_redundancy_check").connect_bus(bus_3)
    l4_task_sensation_focus_check = Agent(name="l4_task_sensation_focus_check").connect_bus(bus_3)
    l4_task_sensation_tool_impact_check = Agent(name="l4_task_sensation_tool_impact_check").connect_bus(bus_3)
    l4_task_sensation_decide_if_response_necessary = Agent(name="l4_task_sensation_decide_if_response_necessary").connect_bus(bus_3)
    l4_task_generate_thought_response_unnecessary = Agent(name="l4_task_generate_thought_response_unnecessary").connect_bus(bus_3)
    l4_task_sensation_postprocess = Agent(name="l4_task_sensation_postprocess").connect_bus(bus_3)

    # Add these low-level agents under the generative agent of sensation
    # This simulates the generative agent calling tools (subagents) for specific checks
    l3_tactic_sensation_gen.add_subagents(
        l4_task_sensation_gather_metadata,
        l4_task_sensation_meta_check,
        l4_task_sensation_relevance_check,
        l4_task_sensation_urgency_check,
        l4_task_sensation_task_relevance_check,
        l4_task_sensation_redundancy_check,
        l4_task_sensation_focus_check,
        l4_task_sensation_tool_impact_check,
        l4_task_sensation_decide_if_response_necessary,
        l4_task_generate_thought_response_unnecessary,
        l4_task_sensation_postprocess
    )

    # ID Layer (id impulses) - same approach
    l3_tactic_id_meta = Agent(name="l3_tactic_id_meta").set_role("meta").connect_bus(bus_3)
    l3_tactic_id_gen = Agent(name="l3_tactic_id_gen").set_role("generative").connect_bus(bus_3)
    l3_tactic_id_disc = Agent(name="l3_tactic_id_disc").set_role("discriminatory").connect_bus(bus_3)
    l3_tactic_id_pred = Agent(name="l3_tactic_id_pred").set_role("predictive").connect_bus(bus_3)

    l4_task_id_pleasure_seeking = Agent(name="l4_task_id_pleasure_seeking").connect_bus(bus_3)
    l4_task_id_pain_avoidance = Agent(name="l4_task_id_pain_avoidance").connect_bus(bus_3)
    l4_task_id_drive_fulfilment = Agent(name="l4_task_id_drive_fulfilment").connect_bus(bus_3)
    l4_task_id_impulse = Agent(name="l4_task_id_impulse").connect_bus(bus_3)
    l4_task_id_fantasy = Agent(name="l4_task_id_fantasy").connect_bus(bus_3)
    l4_task_id_emotional_resonance = Agent(name="l4_task_id_emotional_resonance").connect_bus(bus_3)
    l4_task_id_empathy = Agent(name="l4_task_id_empathy").connect_bus(bus_3)

    l3_tactic_id_gen.add_subagents(
        l4_task_id_pleasure_seeking,
        l4_task_id_pain_avoidance,
        l4_task_id_drive_fulfilment,
        l4_task_id_impulse,
        l4_task_id_fantasy,
        l4_task_id_emotional_resonance,
        l4_task_id_empathy
    )

    # Superego Layer
    l3_tactic_superego_meta = Agent(name="l3_tactic_superego_meta").set_role("meta").connect_bus(bus_3)
    l3_tactic_superego_gen = Agent(name="l3_tactic_superego_gen").set_role("generative").connect_bus(bus_3)
    l3_tactic_superego_disc = Agent(name="l3_tactic_superego_disc").set_role("discriminatory").connect_bus(bus_3)
    l3_tactic_superego_pred = Agent(name="l3_tactic_superego_pred").set_role("predictive").connect_bus(bus_3)

    l4_task_superego_social_compliance = Agent(name="l4_task_superego_social_compliance").connect_bus(bus_3)
    l4_task_superego_reputation_manager = Agent(name="l4_task_superego_reputation_manager").connect_bus(bus_3)
    l4_task_superego_moral_judgement = Agent(name="l4_task_superego_moral_judgement").connect_bus(bus_3)
    l4_task_superego_habit_reinforcement = Agent(name="l4_task_superego_habit_reinforcement").connect_bus(bus_3)
    l4_task_superego_guid_self_correction = Agent(name="l4_task_superego_guid_self_correction").connect_bus(bus_3)
    l4_task_superego_self_image = Agent(name="l4_task_superego_self_image").connect_bus(bus_3)

    l3_tactic_superego_gen.add_subagents(
        l4_task_superego_social_compliance,
        l4_task_superego_reputation_manager,
        l4_task_superego_moral_judgement,
        l4_task_superego_habit_reinforcement,
        l4_task_superego_guid_self_correction,
        l4_task_superego_self_image
    )

    # Ego Layer
    l3_tactic_ego_meta = Agent(name="l3_tactic_ego_meta").set_role("meta").connect_bus(bus_3)
    l3_tactic_ego_gen = Agent(name="l3_tactic_ego_gen").set_role("generative").connect_bus(bus_3)
    l3_tactic_ego_disc = Agent(name="l3_tactic_ego_disc").set_role("discriminatory").connect_bus(bus_3)
    l3_tactic_ego_pred = Agent(name="l3_tactic_ego_pred").set_role("predictive").connect_bus(bus_3)

    # Ego tasks
    l4_task_ego_weighted_voting_scoring = Agent("l4_task_ego_weighted_voting_scoring").connect_bus(bus_3)
    l4_task_ego_iterative_refinement = Agent("l4_task_ego_iterative_refinement").connect_bus(bus_3)
    l4_task_ego_meta_agent_meditation = Agent("l4_task_ego_meta_agent_meditation").connect_bus(bus_3)
    l4_task_ego_generate_gwt_prompt = Agent("l4_task_ego_generate_gwt_prompt").connect_bus(bus_3)
    l4_task_ego_ego_toolset_selection = Agent("l4_task_ego_ego_toolset_selection").connect_bus(bus_3)
    l4_task_ego_prepare_history_preconscious_third_person = Agent("l4_task_ego_prepare_history_preconscious_third_person").connect_bus(bus_3)
    l4_task_ego_metacognitive_analysis = Agent("l4_task_ego_metacognitive_analysis").connect_bus(bus_3)
    l4_task_ego_decide_self_modification_necesary = Agent("l4_task_ego_decide_self_modification_necesary").connect_bus(bus_3)
    l4_task_ego_metacognition_act = Agent("l4_task_ego_metacognition_act").connect_bus(bus_3)
    l4_task_ego_prepare_history_conscious_first_person = Agent("l4_task_ego_prepare_history_conscious_first_person").connect_bus(bus_3)
    l4_task_ego_decide_meta_acting_mode = Agent("l4_task_ego_decide_meta_acting_mode").connect_bus(bus_3)
    l4_task_ego_create_counter_agent = Agent("l4_task_ego_create_counter_agent").connect_bus(bus_3)
    l4_task_ego_build_prompt_third_person = Agent("l4_task_ego_build_prompt_third_person").connect_bus(bus_3)
    l4_task_ego_generate_meta_analysis = Agent("l4_task_ego_generate_meta_analysis").connect_bus(bus_3)
    l4_task_ego_decide_optimization_necesary = Agent("l4_task_ego_decide_optimization_necesary").connect_bus(bus_3)
    l4_task_ego_load_tool_set_metacognition = Agent("l4_task_ego_load_tool_set_metacognition").connect_bus(bus_3)
    l4_task_ego_build_prompt_first_person = Agent("l4_task_ego_build_prompt_first_person").connect_bus(bus_3)
    l4_task_ego_generate_tool_cal = Agent("l4_task_ego_generate_tool_cal").connect_bus(bus_3)
    l4_task_ego_add_reason_no_optimize = Agent("l4_task_ego_add_reason_no_optimize").connect_bus(bus_3)
    l4_task_ego_apply_conversational_agents = Agent("l4_task_ego_apply_conversational_agents").connect_bus(bus_3)
    l4_task_ego_generate_response = Agent("l4_task_ego_generate_response").connect_bus(bus_3)
    l4_task_ego_evaluate_affect_compared_to_goal = Agent("l4_task_ego_evaluate_affect_compared_to_goal").connect_bus(bus_3)
    l4_task_ego_evaluate_metacognitive = Agent("l4_task_ego_evaluate_metacognitive").connect_bus(bus_3)
    l4_task_ego_send_response = Agent("l4_task_ego_send_response").connect_bus(bus_3)
    l4_task_ego_load_tool_set = Agent("l4_task_ego_load_tool_set").connect_bus(bus_3)
    # (Some duplicates may need to be consolidated or renamed if necessary)

    l3_tactic_ego_gen.add_subagents(
        l4_task_ego_weighted_voting_scoring,
        l4_task_ego_iterative_refinement,
        l4_task_ego_meta_agent_meditation,
        l4_task_ego_generate_gwt_prompt,
        l4_task_ego_ego_toolset_selection,
        l4_task_ego_prepare_history_preconscious_third_person,
        l4_task_ego_metacognitive_analysis,
        l4_task_ego_decide_self_modification_necesary,
        l4_task_ego_metacognition_act,
        l4_task_ego_prepare_history_conscious_first_person,
        l4_task_ego_decide_meta_acting_mode,
        l4_task_ego_create_counter_agent,
        l4_task_ego_build_prompt_third_person,
        l4_task_ego_generate_meta_analysis,
        l4_task_ego_decide_optimization_necesary,
        l4_task_ego_load_tool_set_metacognition,
        l4_task_ego_build_prompt_first_person,
        l4_task_ego_generate_tool_cal,
        l4_task_ego_add_reason_no_optimize,
        l4_task_ego_apply_conversational_agents,
        l4_task_ego_generate_response,
        l4_task_ego_evaluate_affect_compared_to_goal,
        l4_task_ego_evaluate_metacognitive,
        l4_task_ego_send_response,
        l4_task_ego_load_tool_set
    )

    # Example: linking metacognitive layers
    # The top-level boss or meta agents at negative indices might oversee their counterpart positive layers
    # They could connect to the same buses or have special control lines
    # This is conceptual; actual code depends on how you've implemented your Agent/Bus classes.
    l0_boss_agent.connect_bus(bus_m1)
    l0_boss_agent.connect_bus(bus_m2)
    l0_boss_agent.connect_bus(bus_m3)

    # Placeholder: Add rating systems, memory references, conversation logs where needed
    # e.g., l1_mission_gen.conversation_log = ConversationLog()
    #       l1_mission_meta.rating_system = RatingSystem()

    # Return the top-level agent that encompasses everything
    return l0_boss_agent

    # what if i made several busses. static and dynamic?
    # the memory agent would subscribe to static and dynamic