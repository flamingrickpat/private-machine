def get_graph():
    graph = Supergraph([
        START,
        start_transaction,
        tasks_create,
        tasks_delegate,
        Selection([
            Key(lambda x: x["next_agent"] == "task_memory", Supergraph([memory_cluster, memory_extract_facts])),
            Key(lambda x: x["next_agent"] == "task_no_sensation", Supergraph([
                internal_thought_create_seed,
                internal_thought_contextualize_seed,
                internal_thought_prefilter,
                Selection(
                    Key(lambda x: x["next_agent"] == "thought_ok"), internal_thought_add_sensation, Supergraph(Goto(label="has_sensation")),
                    Key(lambda x: x["next_agent"] == "thought_not_ok"), Supergraph(END)
                )
            ])),
            Key(lambda x: x["next_agent"] == "task_idle", Supergraph(END)),
            Key(lambda x: x["next_agent"] == "task_sensation", Supergraph(label="has_sensation", flow=[
                sensation_filter_necessary,
                prepare_history_and_context,
                Selection([
                    Key(lambda x: x["next_agent"] == "sensation_filter_yes", Supergraph([
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
                        Selection(
                            Key(lambda x: x["next_agent"] == "response_necessary_yes", Supergraph([Goto(label="response_necessary")])),
                            Key(lambda x: x["next_agent"] == "response_necessary_no", Supergraph([generate_thought_response_unnecssary, sensation_postprocess, END))
                        )
                    ])),
                    Key(lambda x: x["next_agent"] == "sensation_filter_yes", Supergraph([END]))
                ]),
                id_superego_check,
                Selection([
                    Key(lambda x: x["next_agent"] == "id_superego_check_necessary", Supergraph([
                        Supgergraph(label="response_necessary", flow=[
                            prepare_gwt_workspace,
                            Supergraph(label="id", flow=[
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
                            Supergraph(label="superego", flow=[
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
                Selection([
                    Key(lambda x: x["next_agent"] == "ego_metacongition", Supergraph([
                        prepare_history_preconscious_third_person,
                        metacognitive_analysis,
                        decide_self_modification_necesary,
                        metacognition_act,
                    ])),
                    Key(lambda x: x["next_agent"] == "ego_acting", Supergraph([
                        prepare_history_conscious_first_person,
                        decide_meta_acting_mode,
                        Selection([
                            Key(lambda x: x["next_agent"] == "ego_thought", Supergraph([
                                create_counter_agent,
                                build_prompt_third_person,
                                generate_meta_analysis,
                                decide_optimization_necesary,
                                Selection([
                                    Key(lambda x: x["next_agent"] == "optimize_yes", Supergraph([
                                        load_tool_set_metacognition,
                                        build_prompt_first_person,
                                        generate_tool_call
                                    ])),
                                    Key(lambda x: x["next_agent"] == "optimize_no", Supergraph([
                                        add_reason_no_optimize
                                    ]))
                                ])
                            ])),
                            Key(lambda x: x["next_agent"] == "ego_response", Supergraph([
                                apply_conversational_agents,
                                build_prompt_third_person,
                                While([
                                        generate_response,
                                        evaluate_affect_compared_to_goal,
                                        evaluate_metacognitive,
                                    ],
                                    break_condition=lambda x: x["response_ok"]
                                ),
                                send_response
                            ])),
                            Key(lambda x: x["next_agent"] == "ego_tool_call", Supergraph([
                                load_tool_set,
                                build_prompt_first_person,
                                generate_tool_call
                            ]))
                        ])
                    ]))
                ])
            ]))
        ])
    ])
    return graph
