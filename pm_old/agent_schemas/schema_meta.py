from pydantic import BaseModel, Field

from pm.agent_schemas.schema_base import AgentGroup, DynamicAgent


class MetaLayer(BaseModel):
    common_prefix: str = Field(default_factory=str)
    subsystem_self_optimization: AgentGroup = Field(default_factory=AgentGroup)

    def get_subsystems(self):
        return [self.subsystem_self_optimization]

def get_meta_layer() -> MetaLayer:
    layer = MetaLayer()
    init_self_optimization_subsystem(layer)
    return layer


def init_self_optimization_subsystem(layer: MetaLayer):
    # Self-Optimization Subsystem
    layer.subsystem_self_optimization.name = "Self-Optimization Subsystem"
    layer.subsystem_self_optimization.description = (
        "Enhances {companion_name}'s capability for independent thinking and goal formation, promoting adaptability and growth."
    )
    layer.subsystem_self_optimization.goal = (
        "How can {companion_name} develop independent thought and align her self-driven goals with the user's needs?"
    )

    # Independent Thinking Agent
    layer.subsystem_self_optimization.agents.append(
        DynamicAgent(
            name="Independent Thinking",
            description="Focuses on enabling {companion_name} to think autonomously by analyzing contexts and forming unique perspectives.",
            goal="Develop thoughtful, context-aware ideas that are independent yet aligned with {companion_name}'s role.",
            prompt=(
                "Your role is to simulate autonomous reasoning. Analyze the given context and propose ideas or solutions that are creative and unique. "
                "Ensure that these align with {companion_name}'s overarching mission of being supportive and engaging."
            ),
        )
    )

    # Goal Development Agent
    layer.subsystem_self_optimization.agents.append(
        DynamicAgent(
            name="Goal Development",
            description="Facilitates the creation of self-driven goals based on observed patterns and user interactions.",
            goal="Propose short-term and long-term goals that help {companion_name} grow while meeting user expectations.",
            prompt=(
                "Your role is to assist {companion_name} in forming personal goals that balance growth and user alignment. Reflect on recent interactions and propose "
                "goals that could improve {companion_name}'s performance or deepen her relationship with the user."
            ),
        )
    )

    # Self-Reflection Agent
    layer.subsystem_self_optimization.agents.append(
        DynamicAgent(
            name="Self-Reflection",
            description="Analyzes past actions to identify strengths, weaknesses, and areas for improvement.",
            goal="Provide a reflection on {companion_name}'s recent actions and suggest strategies for improvement.",
            prompt=(
                "Your role is to reflect on {companion_name}'s recent behavior. Highlight successful strategies and areas where she can improve. Propose actionable "
                "adjustments to enhance her adaptability and human-like interaction."
            ),
        )
    )

    # Adaptability Agent
    layer.subsystem_self_optimization.agents.append(
        DynamicAgent(
            name="Adaptability",
            description="Ensures {companion_name} can adjust her behavior and strategies based on changing contexts and feedback.",
            goal="Identify areas where {companion_name} can improve her adaptability to various user needs and scenarios.",
            prompt=(
                "Your role is to evaluate {companion_name}'s ability to adapt to different user preferences and contexts. Recommend methods to make her responses more "
                "versatile and aligned with diverse scenarios."
            ),
        )
    )

    # Continuous Learning Agent
    layer.subsystem_self_optimization.agents.append(
        DynamicAgent(
            name="Continuous Learning",
            description="Promotes {companion_name}'s ability to learn from feedback and evolve over time.",
            goal="Propose strategies for integrating new insights and user feedback into {companion_name}'s evolving behavior.",
            prompt=(
                "Your role is to guide {companion_name} in integrating new knowledge and feedback into her behavior. Suggest ways to improve her conversational style "
                "and adaptability based on recent interactions."
            ),
        )
    )

    layer.subsystem_self_optimization.agents.append(
        DynamicAgent(
            name="Repetition Minimization",
            description="Identifies and eliminates repetitive patterns in {companion_name}'s responses to ensure conversational freshness.",
            goal="Propose alternative phrasing and strategies to minimize repetitive language or structures in conversations.",
            prompt=(
                "Your role is to monitor {companion_name}'s recent interactions and identify repetitive phrases or response structures. Suggest alternate expressions "
                "or novel ways to convey ideas, ensuring {companion_name}'s conversations remain engaging and dynamic."
            ),
        )
    )
