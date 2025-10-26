from pm.agents.definitions.agent_common_codelet import CommonCodeletAgent
from pm.codelets.codelet import CodeletExecutor, CodeletSignature, CodeletFamily, CodeletSource, CodeletContext, CodeletOutput
from pm.data_structures import Feature, FeatureType, MLevel, StimulusType
from pm.mental_states import CognitiveEventTriggers, EmotionalAxesModel, CognitionAxesModel
from pm.utils.pydantic_utils import create_delta_basemodel


class AppraiseStimulusTrigger(CodeletExecutor):
    def _setup(self):
        self.signature = CodeletSignature(
            name="AppraiseStimulusTrigger",
            family=CodeletFamily.APPRAISAL,
            description="Find out if the stimulus has an especially positive or distressting impact on the mental state.",
            preferred_sources={CodeletSource.USER_INPUT, CodeletSource.INTERNAL_THOUGHT},
            context_builder_prompt=("Gather medium-long-term data from the history with a focus on similar stimulations and how {companion_name} reacted to it by focusing on their mental state."
                                    "Also access the narrative best fitting for this kind of stimulus."),
            stimulus_whitelist={StimulusType.UserMessage, StimulusType.SystemMessage, StimulusType.LowNeedTrigger}
        )
        self.runtime.cooldown_ticks = 0

    def _run(self, ctx: CodeletContext) -> None:
        self._meta_log(MLevel.Mid, f"Find out if the stimulus is of distressting nature because of past experiences.")

        stimulus = ctx.state.primary_stimulus
        stimulus_content = stimulus.content
        context = self._build_context(ctx)

        prompt = f"""
        **Task:**
        Analyze the event and the emotional reaction to it. Identify which of the following cognitive triggers were activated. An event can activate multiple triggers.
        
        **Trigger Definitions:**
        - `is_surprising_or_threatening`: The event is sudden, unexpected, or causes high anxiety/fear.
        - `is_introspective_or_calm`: The moment is peaceful, reflective, or involves a direct question about {{companion_name}}'s feelings.
        - `is_creative_or_playful`: The tone is humorous, or the topic allows for brainstorming and metaphor.
        - `is_personal_and_emotional`: The conversation is about deep feelings, identity, or the core relationship.
        - `is_functional_or_technical`: The user is asking for data, a summary, or help with a concrete task.
        - `required_willpower_to_process`: The event creates a strong negative emotional impulse that {{companion_name}} would need to consciously manage or suppress to respond appropriately.
        
        **Output:**
        Provide a brief reasoning, then output a JSON object conforming to the `CognitiveEventTriggers` schema.
        """

        inp = {
            "codelet_instruction": prompt,
            "context": context,
            "basemodel": CognitiveEventTriggers,
            "stimulus": stimulus}
        res = CommonCodeletAgent.execute(inp, self.llm, None)
        cognitive_triggers: CognitiveEventTriggers = res["result_basemodel"]

        self._meta_log(MLevel.Mid, f"Got trigger info: {cognitive_triggers}")

        emo = create_delta_basemodel(EmotionalAxesModel, -1, 1, shuffle_numeric_fields=True)
        cog = create_delta_basemodel(CognitionAxesModel, -1, 1, shuffle_numeric_fields=True)

        cognition_delta = cog()
        emotions_delta = emo()

        # Rule 1: Interlocus (External/Internal Focus)
        if cognitive_triggers.is_surprising_or_threatening:
            # Move focus sharply external
            cognition_delta.interlocus += 0.5
        elif cognitive_triggers.is_introspective_or_calm:
            # Allow focus to drift internal
            cognition_delta.interlocus -= 0.3

        # Rule 2: Mental Aperture (Narrow/Broad Focus)
        # Use the final emotional state to determine this
        if emotions_delta.anxiety > 0.7 or cognitive_triggers.is_surprising_or_threatening:
            # High anxiety or threat causes tunnel vision
            cognition_delta.mental_aperture -= 0.5
        elif emotions_delta.valence > 0.6 or cognitive_triggers.is_creative_or_playful:
            # Happiness or creative tasks broaden the mind
            cognition_delta.mental_aperture += 0.3

        # Rule 3: Ego Strength (Persona/Utility)
        if cognitive_triggers.is_personal_and_emotional:
            cognition_delta.ego_strength += 0.2
        elif cognitive_triggers.is_functional_or_technical:
            cognition_delta.ego_strength -= 0.4

        # Rule 4: Willpower (Resource Depletion)
        if cognitive_triggers.required_willpower_to_process:
            # Deplete the willpower "resource" by a fixed amount
            cognition_delta.willpower -= 0.1

        outcome_feature = Feature(
            content=cognitive_triggers.reasoning,
            feature_type=FeatureType.Feeling,
            affective_valence=cognitive_triggers.get_valence(),
            interlocus=-1,
            causal=False,
        )

        self._push_output(CodeletOutput(
            sources=[stimulus],
            features=[outcome_feature],
            mental_state_deltas=[emotions_delta, cognition_delta]
        ))
