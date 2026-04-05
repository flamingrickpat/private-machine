import json
import random
from enum import StrEnum
from typing import List, Type, Union

from pydantic import BaseModel, Field

from pm.agents.agent_base import CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System, Message, BaseAgent, Assistant, ExampleBegin, FunctionCall
from pm.agents.agent_manager import AgentManager
from pm.data.prompts.codelt_instruction import universal_codelet_agent_system_prompt
from pm.subsystems.codelet.codelet_percepts import sample_percept_types, model_for_percept
from pm.system.llm.llm_proxy import LlmManagerProxy
from pm.utils.pydantic_utils import create_basemodel, generate_pydantic_markdown_str


class AgentSimpleCodelet(BaseAgent):
    name = "AgentSimpleCodelet"

    # system prompts
    def __init__(self, llm: LlmManagerProxy, manager: AgentManager):
        super().__init__(llm, manager)
        self.codelet_name = None

    def get_system_prompts(self) -> List[str]:
        return [universal_codelet_agent_system_prompt]

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

        ops.append(User("This is the most recent context of the AI companion {companion_name}'s mental state. It includes dialogue with user as well as previous mental features (memories, feelings, previous codelet output)."
                f"\n```context\n{story}\n```\n"
                "The AI companion's system just received this new input:" 
                f"\n```input\n{story_new}\n```\n"
                ""
                "Your goal is act like this mental mechanism of {companion_name}:\n"
                f"Name: {codelet.name}\n"
                f"Description: {codelet.description}\n"
                f"Instructions: {codelet.prompt}\n"
                f"To give a detailed analysis, you will need more information. Generate a list of questions and answers will be provided. About background, characters, world. Anything not already in the context."))

        def memquery(questions: str) -> str:
            lines = questions.splitlines(keepends=False)
            answers = []
            for i, q in enumerate(lines):
                answers.append(f"Question {i}: {q}")
                answer = self.agent_manager.ghost.memory_manager.answer_question(q)
                answers.append(f"Asnwer {i}: {answer}")
            res = "\n".join(answers)
            print(res)
            return res

        ops.append(Assistant("Questions:\n-"))
        ops.append(CompletionText(target_key="questions"))
        ops.append(FunctionCall(source_key="questions", target_key="answers", delegate=memquery))
        ops.append(User(source_key="answers"))

        ops.append(Assistant(f"""I will now act as the "{codelet.name}" mental mechanism for {{companion_name}}, and analyse the situation and new input from the codelets point of view.

Here's my response, simulating {{companion_name}}'s thought process from the angle of {codelet.name}:

"""))

        ops.append(CompletionText(target_key="first_pass"))

        possible_percept_types = sample_percept_types(codelet.families, percept_type_count, temperature=0.2)
        defs = []
        for p in possible_percept_types:
            model: Type[BaseModel] = model_for_percept(p)
            d = {
                "name": model.__name__,
                "description": model.__doc__,
                "type": model
            }
            defs.append(d)
        FeatureContainer = create_basemodel(defs, randomnize_entries=True)

        ops.append(User(
            f"Now create a JSON object based on your analysis. Here is a schema you must follow when creating the JSON: {generate_pydantic_markdown_str(FeatureContainer)}"
            f"Only output valid JSON, nothing else."
        ))

        ops.append(CompletionJSON(schema=FeatureContainer, target_key="output_feature"))
        ops.append(ExampleEnd())

        return ops