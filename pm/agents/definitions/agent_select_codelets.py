import random
from typing import List, Type

from pydantic import BaseModel

from pm.agents.agent_base import CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System, Message, BaseAgent, Assistant, ExampleBegin
from pm.codelets.codelet import CodeletExecutor
from pm.codelets.codelet_definitions import SimpleCodelet
from pm.utils.pydantic_utils import create_basemodel


class AgentSelectCodelets(BaseAgent):
    name = "AgentSelectCodelets"

    # system prompts
    def get_system_prompts(self) -> List[str]:
        return []

    def get_rating_system_prompt(self) -> str:
        return ""

    def get_rating_probability(self) -> float:
        return 0

    def get_default_few_shots(self) -> List[Message]:
        return []

    def build_plan(self) -> List[PromptOp]:
        content = """I have made a list with possible mental functions. I want to psychologically simulate what {companion_name} would say based on her psyche."
"But the list is long. You now act as a selector for possible mental functions. Here is the JSON schema with all the descriptions, and names. "
"Please give each codelet a rating between 0 and 1, for how likely it is I should invoke it for this conversation."""

        context = self.input.get("context", "")
        full_prompt = self.input.get("full_prompt", None)
        codelets = self.input["codelets"]
        ops: List[PromptOp] = []

        if full_prompt:
            for tmp in full_prompt:
                role, text = tmp[0], tmp[1]
                if role == "user": ops.append(User(text))
                elif role == "assistant": ops.append(Assistant(text))
                elif role == "system": ops.append(System(text))

        ops.append(ExampleBegin())
        if context and context != "":
            ops.append(User(
                f"Context: {context}\n"
            ))
        ops.append(User(
            f"{content}"
        ))

        defs = []
        for c in codelets:
            codelet: CodeletExecutor = c
            d = {
                "name": codelet.signature.name,
                "description": codelet.signature.description,
                "type": float
            }
            defs.append(d)

        AllCodeletsWithDesc = create_basemodel(defs)

        ops.append(CompletionJSON(schema=AllCodeletsWithDesc, target_key="codelets"))
        ops.append(ExampleEnd())

        return ops