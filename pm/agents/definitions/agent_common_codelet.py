from typing import List, Type

from pydantic import BaseModel

from pm.agents.agent_base import CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System, Message, BaseAgent, Assistant, ExampleBegin

class CommonCodeletAgent(BaseAgent):
    name = "CommonCodelet"

    # system prompts
    def get_system_prompts(self) -> List[str]:
        return []

    def get_rating_system_prompt(self) -> str:
        return ""

    def get_rating_probability(self) -> float:
        return 0

    def get_default_few_shots(self) -> List[Message]:
        # ───── Cluster 13: Hierarchical Summarization ─────
        text_cluster_5 = """
        The team discussed their cloud migration strategy. They decided on AWS as the primary provider, agreed on a phased rollout starting with internal tools, and highlighted security concerns around identity management and data encryption. A timeline was drafted, aiming for completion within six months, with regular check-ins every two weeks.
        The product design group reviewed user feedback from the latest prototype. They agreed to simplify navigation menus, improve accessibility for visually impaired users, and refine the color palette for better contrast. Testing with a broader user base was scheduled for the next sprint, and final design approval was targeted for the end of the quarter.
        """

        text_summary_5 = """
        Both groups are aligning strategy and design toward long-term product readiness. The cloud migration team is focused on infrastructure and security, while the product design team is concentrating on usability and accessibility. Taken together, the organization is simultaneously addressing the technical foundation and the user-facing experience. The combined summary shows a coordinated effort: secure and scalable systems being developed in parallel with accessible, user-friendly design, each progressing along defined timelines toward a cohesive launch.
        """

        return [
            ("user", text_cluster_5),
            ("assistant", text_summary_5),
        ]

    def build_plan(self) -> List[PromptOp]:
        dialog = self.input["content"]
        context = self.input["context"]
        ops: List[PromptOp] = []

        for s in self.get_system_prompts():
            ops.append(System(s))
        for role, text in self.get_default_few_shots():
            if role == "user": ops.append(User(text))
            elif role == "assistant": ops.append(Assistant(text))

        ops.append(ExampleBegin())
        ops.append(User(
            f"Context: {context}\n"
            f"{dialog}"
        ))
        ops.append(CompletionText(target_key="summary"))
        ops.append(ExampleEnd())

        return ops