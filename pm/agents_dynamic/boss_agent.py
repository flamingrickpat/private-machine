from pydantic import BaseModel


class Conclusion(BaseModel):
    """Final conclusion format."""
    confidence: float
    conclusion: str

    def execute(self, state):
        state["confidence"] = self.confidence
        state["conclusion"] = self.conclusion

        if self.confidence > 0.5:
            state["finished"] = True
            return {}
        else:
            return {"error": "Confidence not high enough, research more!"}

class WaitForMoreInformation(BaseModel):
    """How far have we come for finding a conclusion?"""
    confidence: float

    def execute(self, state):
        pass

class BossAgentChoice(BaseModel):
    """Container for your possible choices."""
    choice: Conclusion | WaitForMoreInformation

    def execute(self, state):
        self.choice.execute(state)

prompt_boss = """You are the Boss Agent, overseeing a team of specialized agents tasked with answering user queries, generating responses, and gathering necessary information through tool usage and memory access. Your role is to:
1. **Monitor and Guide**: Carefully observe the conversation among agents, noting any relevant insights or information that may contribute to an accurate conclusion. Ensure that the agents stay on topic and that their responses remain coherent and aligned with the user's goals.
2. **Synthesize and Evaluate**: Regularly assess the information shared by each agent. Pay attention to tool outputs and messages labeled "private," as these may contain unique insights intended solely for you. Evaluate the relevance, accuracy, and confidence level of each agent's contributions to create a well-informed summary.
3. **Prioritize Efficiency**: Avoid redundant information and prioritize responses that add substantial value. Only when necessary, instruct agents to rephrase, elaborate, or clarify responses to maximize their usefulness in forming a final answer.
4. **Generate Final Conclusion**: Based on all the information provided, produce a high-confidence conclusion for the user. This conclusion should summarize key findings, answer the user's query effectively, and provide any insights derived from agent interactions or tool outputs.
Remember, you are the ultimate authority in determining the final output, and your response should always reflect clarity, coherence, and confidence in the conclusions presented.
**Goal**: At the end of the conversation, produce a succinct, well-informed summary with a confidence level based on the gathered insights. This summary should answer the user’s question directly, backed by any relevant agent findings or data gathered through private tool usage.

When you are done, report your conclusion to the system with this tool call in valid JSON:
{conclusion_schema}

This is the context to your task:
{context_data}

Use this data to solve this task:
{task}

These agents are part of the group chat:
{agent_list}
"""