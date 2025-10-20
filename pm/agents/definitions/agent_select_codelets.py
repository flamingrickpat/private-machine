import random
from typing import List, Type

from pydantic import BaseModel

from pm.agents.agent_base import CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System, Message, BaseAgent, Assistant, ExampleBegin
from pm.codelets.codelet_definitions import ALL_CODELETS, SimpleCodelet
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
        content = self.input["content"]
        context = self.input.get("context", "")
        full_prompt = self.input.get("full_prompt", None)
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

        tmp = random.sample(ALL_CODELETS, 20)
        defs = []
        for c in tmp:
            codelet: SimpleCodelet = c
            d = {
                "name": codelet.name,
                "description": codelet.description,
                "type": float
            }
            defs.append(d)

        AllCodeletsWithDesc = create_basemodel(defs)

        ops.append(CompletionJSON(schema=AllCodeletsWithDesc, target_key="codelets"))
        ops.append(ExampleEnd())

        return ops