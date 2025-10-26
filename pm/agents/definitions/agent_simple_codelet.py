import json
import random
from enum import StrEnum
from typing import List, Type, Union

from pydantic import BaseModel, Field

from pm.agents.agent_base import CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System, Message, BaseAgent, Assistant, ExampleBegin
from pm.agents.agent_manager import AgentManager
from pm.codelets.codelet_outputs import sample_percept_types, model_for_percept
from pm.config_loader import companion_name
from pm.llm.llm_proxy import LlmManagerProxy
from pm.utils.pydantic_utils import create_basemodel, flatten_pydantic_model, generate_pydantic_json_schema_llm


class AgentSimpleCodelet(BaseAgent):
    name = "AgentSimpleCodelet"

    # system prompts
    def __init__(self, llm: LlmManagerProxy, manager: AgentManager):
        super().__init__(llm, manager)
        self.codelet_name = None

    def get_system_prompts(self) -> List[str]:
        return ["You are an expert in storywriting. You can write stories with vivid, immersive and complex characters and you can analyze stories to the same extent."]

    def get_rating_system_prompt(self) -> str:
        return ""

    def get_rating_probability(self) -> float | None:
        return 0

    def get_default_few_shots(self) -> List[Message]:
        return []

    def get_log_name(self) -> str:
        return f"{self.name}_{self.codelet_name}"

    def build_plan(self) -> List[PromptOp]:
        context = self.input.get("context", "")
        codelet = self.input["codelet"]
        story = self.input["story"]
        story_new = self.input["story_new"]
        csm_snapshot = self.input["csm_snapshot"]
        ops: List[PromptOp] = []
        percept_type_count = 3

        self.codelet_name = codelet.name

        ops.append(System("\n".join(self.get_system_prompts())))

        user1 = "I'm writing a story about a next-level AI companion and a user. I'm having trouble continuing and keeping consistent personas. Here is the story, I only selected the most important parts of the story: " \
                 f"\n### BEGIN STORY\n{story}\n### END STORY\n" \
                 f"Did you get all of that?"

        ass1 = "Yes, I perfectly understand the story and its characters. What is your task?"

        user2 =  "Perfect. This is what I'm going to write next:" \
                 f"\n### BEGIN NEW\n{story_new}\n### END NEW\n"

        ops.append(User(user1))
        ops.append(Assistant(ass1))
        ops.append(User(user2))

        ops.append(ExampleBegin())
        if context and context != "":
            ops.append(User(
                f"Here is some extra information. Unassorted notes and world lore. "
                f"\n### BEGIN CONTEXT\n"
                f"{context}"
                f"\n### END CONTEXT\n"
            ))

        if csm_snapshot and csm_snapshot != "":
            ops.append(User(
                f"Here is the analysis of the new input I already did, with a different focus as you have."
                f"You can build on them, or go into a different direction if you think it fits the story better."
                f"\n### BEGIN MENTAL FEATUERES\n"
                f"{csm_snapshot}"
                f"\n### END MENTAL FEATUERES\n"
            ))

        ops.append(User(
            "Your task is to pretend to be a mental mechanism of {companion_name} and to simulate what happens in their mind."
            f"Your goal is act like this mental mechanism:\n"
            f"Name: {codelet.name}\n"
            f"Description: {codelet.description}\n"
            f"Instructions: {codelet.prompt}\n"
        ))

        ass_seed = f"""Okay, I understand. I will act as the "{codelet.name}" mental mechanism for {companion_name}, crafting 3-5 sentences of third-person narrative. Let's begin!

Here's my response, simulating {companion_name}'s thought process:

"""

        ops.append(Assistant(ass_seed))
        ops.append(CompletionText(target_key="first_pass"))

        possible_percept_types = sample_percept_types(codelet.families, percept_type_count, temperature=0.2)
        defs = []
        for p in possible_percept_types:
            model: Type[BaseModel] = model_for_percept(p)
            flat = flatten_pydantic_model(model)
            d = {
                "name": model.__name__,
                "description": model.__doc__,
                "type": flat
            }
            defs.append(d)
        FeatureContainer = create_basemodel(defs, randomnize_entries=True)

        ops.append(User(
            "Very nice, but you could make it more 'narrative', as in, being able to plug it into a story."
            "I want to keep track of all possible story paths. Could you put the result of your analysis into this JSON schema?"
            "I defined a few possible output types for easy categorization. Use as many as you like."
            f"Please use this JSON schema to generate a answer in valid JSON: {generate_pydantic_json_schema_llm(FeatureContainer)}"
            f"Please only output valid JSON, nothing else."
        ))

        ops.append(CompletionJSON(schema=FeatureContainer, target_key="output_feature"))
        ops.append(ExampleEnd())

        return ops