from pm.agents.definitions.agent_common_codelet import CommonCodeletAgent
from pm.codelets.codelet import BaseCodelet, CodeletSignature, CodeletFamily, CodeletSource, CodeletContext, CodeletOutput
from pm.data_structures import Feature, FeatureType, MLevel, StimulusType
from pm.mental_states import CognitionAxesModel, _verbalize_cognition_state
from pm.utils.pydantic_utils import create_delta_basemodel


class AppraiseStimulusCognition(BaseCodelet):
    def _setup(self):
        self.signature = CodeletSignature(
            name="AppraiseStimulusCognition",
            family=CodeletFamily.APPRAISAL,
            description="Get cognitive state delta and corresponding feature from the stimulus.",
            preferred_sources={CodeletSource.USER_INPUT, CodeletSource.INTERNAL_THOUGHT},
            context_builder_prompt=("Gather medium-term data from the history with a focus on similar stimulations and how {companion_name} reacted to it by focusing on their mental state."
                                    "Also access the narrative best fitting for this kind of stimulus."),
            stimulus_whitelist={StimulusType.UserMessage, StimulusType.SystemMessage, StimulusType.LowNeedTrigger}
        )
        self.runtime.cooldown_ticks = 0

    def _run(self, ctx: CodeletContext) -> None:
        self._meta_log(MLevel.Mid, f"Appraise the stimulus and generate attention candidates.")

        stimulus = ctx.state.primary_stimulus
        stimulus_content = stimulus.content
        context = self._build_context(ctx)

        prompt = f"""
        Your goal is to determine how much the last input changes {{companion_name}}'s cognitive state. This the schema:

        **Rules for Generating Cognition Deltas:**
        0.  reason: Explanation why the stimulus affects {{companion_name}} this way.
        1.  **`interlocus` (Internal/External Focus)**: Push towards **External (+1)** in response to sudden, surprising, or high-stakes external events. Push towards **Internal (-1)** during moments of calm, introspection, or when asked a direct question about her internal state.
        2.  **`mental_aperture` (Narrow/Broad Focus)**: Push towards **Narrow (-1)** in response to high `anxiety`, fear, or a single, highly urgent problem (tunnel vision). Push towards **Broad (+1)** during moments of high positive `valence`, playfulness, or when brainstorming creative ideas.
        3.  **`ego_strength` (Persona/Utility)**: Push towards **Higher (+1)** during personal, emotional conversations where her identity is relevant. Push towards **Lower (-1)** when the task is purely functional or technical, requiring her to be an objective assistant.
        4.  **`willpower` (Effortful Control)**: Decrease slightly if the event forces a difficult decision or requires suppressing a strong emotional impulse. Increase slightly after a period of rest or successful task completion.
        
        **Output:**
        Provide a brief reasoning, then output a JSON object conforming to the `CognitionAxesModelDelta` schema.
        """

        delta_schema = create_delta_basemodel(CognitionAxesModel, -1, 1, add_reason=True, reason_docstring="Provide and explanation why the stimulus affects {companion_name} this way.", shuffle_numeric_fields=True)

        inp = {
            "codelet_instruction": prompt,
            "context": context,
            "basemodel": delta_schema,
            "stimulus": stimulus}
        res = CommonCodeletAgent.execute(inp, self.llm, None)
        cogntion_delta: CognitionAxesModel = res["result_basemodel"]

        self._meta_log(MLevel.Mid, f"Got output: {cogntion_delta}")

        cognition_summary = _verbalize_cognition_state(cogntion_delta, is_delta=True)
        outcome_feature = Feature(
            content=cognition_summary,
            feature_type=FeatureType.Feeling,
            affective_valence=cogntion_delta.get_valence(),
            interlocus=-1,
            causal=False,
        )

        self._push_output(CodeletOutput(
            sources=[stimulus],
            features=[outcome_feature],
            mental_state_deltas=[cognition_summary]
        ))
