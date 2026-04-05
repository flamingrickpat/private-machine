import json
import random
from enum import StrEnum
from typing import List, Type, Union

from pydantic import BaseModel, Field

from pm.agents.agent_base import CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System, Message, BaseAgent, Assistant, ExampleBegin
from pm.agents.agent_manager import AgentManager
from pm.subsystems.codelet.codelet_percepts import sample_percept_types, model_for_percept
from pm.system.llm.llm_proxy import LlmManagerProxy
from pm.utils.pydantic_utils import create_basemodel, generate_pydantic_markdown_str


class AgentComplexCodelet(BaseAgent):
    """
    Universal executor for grounded non-dialogue codelets.

    Design goals:
    - stable prefix for cache reuse
    - structured missing-memory detection
    - retrieval only when needed
    - no literary analysis step
    - explicit grounding self-check
    - final result is usable by CSM / GWT / action selection
    """

    name = "AgentComplexCodelet"

    # system prompts
    def __init__(self, llm: LlmManagerProxy, manager: AgentManager):
        super().__init__(llm, manager)
        self.codelet_name = None
        self.llm = llm
        self.memory_router = manager.memory_router
        self.salience_estimator = manager.salience_estimator
        self.codelet = None

    def get_stable_system_prompt(self) -> str:
        return """
You are a cognitive codelet executor.

You do not write dialogue.
You do not roleplay.
You do not write literary inner monologue.
You do not invent bodily sensations, hardware states, cinematic imagery, or hidden motives.

Your task is to transform a situation into grounded provisional percepts for a cognitive architecture.

Rules:
- Use only supplied evidence.
- Mark uncertainty explicitly.
- Prefer reversible interpretations over dramatic ones.
- Request more memory when needed.
- Produce compact structured outputs.
- Treat all outputs as provisional unless explicitly marked otherwise.
"""

    def build_plan(self) -> list[PromptOp]:
        codelet = self.input["codelet_spec"]
        self.codelet = codelet

        ops: list[PromptOp] = []

        # ----------------------------------------------------
        # 0. Stable prefix
        # ----------------------------------------------------
        ops.append(StableSystem(self.get_stable_system_prompt()))
        ops.append(Checkpoint("stable_prefix"))

        # ----------------------------------------------------
        # 1. Inject permanent codelet identity
        # ----------------------------------------------------
        ops.append(InjectTaskPacket(
            situation_key="situation_packet",
            raw_inputs_key="raw_inputs",
            workspace_key="workspace_snapshot",
            prior_codelet_key="prior_provisional_codelets",
            memory_bundle_key=None,
        ))

        ops.append(FewShot("codelet_examples__clean_grounded"))
        ops.append(InjectSchemaGuide(MissingMemoryAssessment, with_examples=True))

        # ----------------------------------------------------
        # 2. Ask only for structured missing-memory assessment
        # ----------------------------------------------------
        ops.append(GenerateJSON(
            model_type=MissingMemoryAssessment,
            target_key="missing_memory_assessment"
        ))

        # ----------------------------------------------------
        # 3. Optional retrieval branch
        # ----------------------------------------------------
        ops.append(
            BranchIf(
                predicate_key="missing_memory_assessment.needs_more_memory",
                if_true=[
                    MemoryLookup(
                        requests_key="missing_memory_assessment.requests",
                        target_key="retrieved_memory_raw",
                        max_items=10,
                        allow_types=codelet.allowed_memory_types,
                    ),
                    EvidencePrune(
                        source_key="retrieved_memory_raw",
                        target_key="retrieved_memory_pruned",
                        max_items=6,
                    ),
                ],
                if_false=[]
            )
        )

        # ----------------------------------------------------
        # 4. Rewind to stable prefix and rebuild clean context
        # ----------------------------------------------------
        ops.append(Restore("stable_prefix"))

        ops.append(InjectTaskPacket(
            situation_key="situation_packet",
            raw_inputs_key="raw_inputs",
            workspace_key="workspace_snapshot",
            prior_codelet_key="prior_provisional_codelets",
            memory_bundle_key="retrieved_memory_pruned",
        ))

        ops.append(FewShot("codelet_examples__clean_grounded"))
        ops.append(InjectSchemaGuide(CodeletRunResult, with_examples=True))

        # ----------------------------------------------------
        # 5. First draft
        # ----------------------------------------------------
        ops.append(GenerateJSON(
            model_type=CodeletRunResult,
            target_key="draft_result"
        ))

        # ----------------------------------------------------
        # 6. Grounding self-check
        # ----------------------------------------------------
        ops.append(SelfCheckJSON(
            model_type=GroundingSelfCheck,
            against_keys=[
                "raw_inputs",
                "workspace_snapshot",
                "prior_provisional_codelets",
                "retrieved_memory_pruned",
                "draft_result"
            ],
            target_key="draft_selfcheck"
        ))

        # ----------------------------------------------------
        # 7. Repair if needed
        # ----------------------------------------------------
        ops.append(
            RetryIfInvalid(
                source_key="draft_result",
                feedback_key="draft_selfcheck",
                max_retries=1,
                retry_ops=[
                    Restore("stable_prefix"),
                    InjectTaskPacket(
                        situation_key="situation_packet",
                        raw_inputs_key="raw_inputs",
                        workspace_key="workspace_snapshot",
                        prior_codelet_key="prior_provisional_codelets",
                        memory_bundle_key="retrieved_memory_pruned",
                    ),
                    FewShot("codelet_examples__clean_grounded"),
                    InjectSchemaGuide(CodeletRunResult, with_examples=True),
                    # imaginary op: inject the repair feedback directly as structured constraint
                    InjectTaskPacket(
                        situation_key="selfcheck_feedback_packet",
                        raw_inputs_key="draft_selfcheck",
                        workspace_key=None,
                        prior_codelet_key=None,
                        memory_bundle_key=None,
                    ),
                    GenerateJSON(
                        model_type=CodeletRunResult,
                        target_key="final_result"
                    ),
                ]
            )
        )

        # ----------------------------------------------------
        # 8. If no repair happened, promote draft -> final
        # ----------------------------------------------------
        ops.append(MergeResult(
            source_keys=["draft_result"],
            target_key="final_result"
        ))

        # ----------------------------------------------------
        # 9. Deterministic or hybrid salience pass
        # ----------------------------------------------------
        ops.append(ComputeSalience(
            source_key="final_result",
            target_key="final_salience"
        ))

        ops.append(Finish())
        return ops