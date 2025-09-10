from typing import List, Type

from pydantic import BaseModel

from pm.agents.agent_base import CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System, Message, BaseAgent, Assistant, ExampleBegin


class SummaryCategoryBaseModel(BaseModel):
    topic: str
    confidence: float
    rationale: str

# ──────────────────────────────────────────────────────────────────────────────
# Concrete agent
# ──────────────────────────────────────────────────────────────────────────────

class SummarizeTopicConversation(BaseAgent):
    name = "SummarizeTopicConversation"

    # system prompts
    def get_system_prompts(self) -> List[str]:
        return [
            "You are an expert in literate and literal analysis; you help with stories and book reports.",
            "Your task: summarize a piece of dialogue. Be neutral, descriptive, concise. Do not invent facts.",
        ]

    def get_rating_system_prompt(self) -> str:
        return ("You rate summaries of dialogue. Criteria: (1) shorter than original; "
                "(2) neutral & descriptive; (3) no fabrication; (4) well-categorized topic. "
                "Return JSON: {rating:int 0..5, reason:str}")

    def get_rating_probability(self) -> float:
        return 0.03

    def get_default_few_shots(self) -> List[Message]:
        # minimal seed; expand as you mine examples
        return [
            ("user", "D: 'Where were you?' — E: 'At the library.' They discuss late arrival."),
            ("assistant", "They briefly clarify whereabouts; tone is calm, mildly apologetic."),
            ("user", "A: 'You never listen.' B: 'I do, but I'm stressed.'"),
            ("assistant", "They argue about attentiveness; frustration and defensiveness surface."),
        ]

    def build_plan(self) -> List[PromptOp]:
        dialog = self.input["content"]
        schema: Type[BaseModel] = self.input.get("SummaryCategoryBaseModel", SummaryCategoryBaseModel)

        ops: List[PromptOp] = []

        # Static preamble (kept in KV)
        for s in self.get_system_prompts():
            ops.append(System(s))
        for role, text in self.get_default_few_shots():
            if role == "user": ops.append(User(text))
            elif role == "assistant": ops.append(Assistant(text))

        # Begin mined example recording
        ops.append(ExampleBegin())

        # Instruction and user content
        ops.append(User(
            "Here is the dialogue block:\n"
            f"{dialog}\n"
            "### END\n"
            "Please summarize the dialogue (neutral, concise, no invention). The summary should be usable in place of the raw dialogue."
        ))

        # First completion: natural-language summary
        ops.append(CompletionText(target_key="summary"))

        # Second step: JSON topic categorization (no re-eval; same session)
        ops.append(User(
            "Now categorize that summary into a 'Topic' and explain briefly. "
            "Return strictly valid JSON that matches the provided schema."
        ))
        # Provide the schema verbatim to the model (your manager can enforce tools too)
        schema_json = schema.model_json_schema()
        ops.append(User("JSON schema:\n" + str(schema_json)))

        ops.append(CompletionJSON(schema=schema, target_key="category"))

        # End example recording
        ops.append(ExampleEnd())

        return ops