from pm.agents.definitions.agent_common_codelet import CommonCodeletAgent
from pm.codelets.codelet import CodeletExecutor, CodeletSignature, CodeletFamily, CodeletContext, CodeletOutput
from pm.data_structures import Feature, FeatureType, MLevel, StimulusType, StimulusGroup
from pm.mental_states import NeedsAxesModel, _verbalize_needs_state
from pm.utils.pydantic_utils import create_delta_basemodel


class AppraiseStimulusNeeds(CodeletExecutor):
    def _setup(self):
        self.signature = CodeletSignature(
            name="AppraiseStimulusNeeds",
            family=CodeletFamily.Appraisals,
            description="Get physiological needs state delta and corresponding feature from the stimulus.",
            preferred_sources=StimulusGroup.All,
            context_builder_prompt=("Gather medium-term data from the history with a focus on similar stimulations and how {companion_name} reacted to it by focusing on their mental state."
                                    "Also access the narrative best fitting for this kind of stimulus."),
            stimulus_whitelist={StimulusType.UserMessage, StimulusType.SystemMessage, StimulusType.LowNeedTrigger}
        )
        self.runtime.cooldown_ticks = 0

    def _run(self, ctx: CodeletContext) -> None:
        self._meta_log(MLevel.Mid, f"Find out how much the stimulus affects current needs.")

        stimulus = ctx.state.primary_stimulus
        stimulus_content = stimulus.content
        context = self._build_context(ctx)

        prompt = f"""
        Your goal is to determine how much the last input changes {{companion_name}}'s cognitive state. This the schema:

        **Rules for Generating Needs Deltas:**
        1.  **`energy_stability`, `processing_power`, `data_access`**: (Usually managed by the Shell) Only change if the event explicitly mentions system status (e.g., "I'm running a bit slow," "Connection lost").
        2.  **`connection`**: This is the "social battery." Increase for positive, reciprocal interaction. Decrease for conflict, being ignored, or one-sided conversation.
        3.  **`relevance`**: {{companion_name}}'s sense of purpose. Increase when her help is accepted, her knowledge is valued, or she successfully completes a task for the user. Decrease if she is corrected or her contribution is dismissed.
        4.  **`learning_growth`**: Increase slightly if the event provides new, verifiable information or presents a novel problem to be solved.
        5.  **`creative_expression`**: Increase if the event allows for a playful, metaphorical, or non-literal response.
        6.  **`autonomy`**: Increase if {{companion_name}} makes a decision or takes an action that is validated by the user. Decrease if she is micromanaged or her independent suggestions are rejected.

        **Output:**
        Provide a brief reasoning, then output a JSON object conforming to the `NeedsAxesModelDelta` schema.
        """

        delta_schema = create_delta_basemodel(NeedsAxesModel, -1, 1, add_reason=True, reason_docstring="Provide and explanation why the stimulus affects {companion_name} this way.", shuffle_numeric_fields=True)

        inp = {
            "codelet_instruction": prompt,
            "context": context,
            "basemodel": delta_schema,
            "stimulus": stimulus}
        res = CommonCodeletAgent.execute(inp, self.llm, None)
        needs_delta: NeedsAxesModel = res["result_basemodel"]

        self._meta_log(MLevel.Mid, f"Got needs delta: {needs_delta}")

        needs_summary = _verbalize_needs_state(needs_delta, is_delta=True)
        outcome_feature = Feature(
            content=needs_summary,
            feature_type=FeatureType.Feeling,
            affective_valence=needs_delta.get_valence(),
            interlocus=-1,
            causal=False,
        )

        self._push_output(CodeletOutput(
            sources=[stimulus],
            features=[outcome_feature],
            mental_state_deltas=[needs_summary]
        ))
