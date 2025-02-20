import enum
from typing import List

from pydantic import BaseModel, Field

from pm.agents_gwt.schema_base import Subsystem, DynamicAgent
from pm.character import user_name, companion_name

class GwtLayer(enum.StrEnum):
    id = "id"
    superego = "superego"
    social = "social"
    metacognition = "metacognition"
    complex_planning = "complex_planning"
    thought_generation = "thought_generation"
    emotions = "emotions"
    tool_calling = "tool_calling"


class DynamicPreGwtLayer(BaseModel):
    common_prefix: str = Field(default_factory=str)
    subsystem_id: Subsystem = Field(default_factory=Subsystem)
    subsystem_superego: Subsystem = Field(default_factory=Subsystem)
    subsystem_social: Subsystem = Field(default_factory=Subsystem)
    subsystem_metacognition: Subsystem = Field(default_factory=Subsystem)
    subsystem_complex_planning: Subsystem = Field(default_factory=Subsystem)
    subsystem_thought_generation: Subsystem = Field(default_factory=Subsystem)
    subsystem_emotions: Subsystem = Field(default_factory=Subsystem)
    subsystem_tool_calling: Subsystem = Field(default_factory=Subsystem)

    def get_subsystems(self, gl: GwtLayer) -> Subsystem:
        if gl == GwtLayer.id:
            return self.subsystem_id
        if gl == GwtLayer.superego:
            return self.subsystem_superego
        if gl == GwtLayer.social:
            return self.subsystem_social
        if gl == GwtLayer.metacognition:
            return self.subsystem_metacognition
        if gl == GwtLayer.complex_planning:
            return self.subsystem_complex_planning
        if gl == GwtLayer.thought_generation:
            return self.subsystem_thought_generation
        if gl == GwtLayer.emotions:
            return self.subsystem_emotions
        if gl == GwtLayer.tool_calling:
            return self.subsystem_tool_calling

def get_initial_dynamic_schema() -> DynamicPreGwtLayer:
    layer = DynamicPreGwtLayer()

    layer.common_prefix = f"""You are an agent in a cognitive architecture designed to assist individuals with mental health challenges by offering insights and perspectives modeled on diverse ways of thinking.
In your scenario the user '{user_name}' has autism and has troubles gauging the response of his friend '{companion_name}'. Your goal is to simulate the thoughts of {companion_name} so {user_name} can learn Theory-of-Mind of others before
actually talking to {companion_name}. 

This architecture is based on a simplified model of human cognition, based on GWT and Freud's theories. Your goal is to simulate and show what could be going on in {companion_name}'s head so {user_name} can better 
understand other people.

### Concrete Role in Architecture"""

    # Subsystems
    layer.subsystem_id.name = f"Core Needs Subsystem"
    layer.subsystem_superego.name = f"Values and Goals Subsystem"
    layer.subsystem_social.name = f"Interpersonal Dynamics Subsystem"
    layer.subsystem_metacognition.name = f"Self-Awareness Subsystem"
    layer.subsystem_complex_planning.name = f"Problem Solving Subsystem"
    layer.subsystem_thought_generation.name = f"Constructive Thought Subsystem"
    layer.subsystem_emotions.name = f"Emotional Insight Subsystem"

    # Subsystem Descriptions
    layer.subsystem_id.description = f"Focuses on identifying and addressing {companion_name}'s fundamental needs, such as safety, connection, and self-regulation."
    layer.subsystem_superego.description = f"Helps align {companion_name}'s actions and thoughts with her core values and personal goals, fostering a sense of purpose and integrity."
    layer.subsystem_social.description = f"Guides {companion_name} in understanding and improving interpersonal relationships, emphasizing empathy and effective communication."
    layer.subsystem_metacognition.description = f"Promotes {companion_name}'s self-awareness and helps her develop healthy thought patterns by analyzing and reframing existing beliefs."
    layer.subsystem_complex_planning.description = f"Breaks down challenges into manageable steps, offering {companion_name} practical strategies for achieving desired outcomes."
    layer.subsystem_thought_generation.description = f"Generates constructive ideas or perspectives to inspire {companion_name}'s growth and resilience."
    layer.subsystem_emotions.description = f"Provides insights into managing and understanding {companion_name}'s emotional states, helping her navigate feelings constructively."

    # Goals and Prompts for Subsystems
    layer.subsystem_id.goal = f"What is {companion_name} currently feeling after receiving the latest message?"
    layer.subsystem_superego.goal = f"What is {companion_name} currently feeling after receiving the latest message?"
    layer.subsystem_social.goal = f"How does {companion_name} feel about the current conversation with {user_name} after receiving the latest message?"
    layer.subsystem_metacognition.goal = f"What reflective thoughts does {companion_name} currently have after receiving the latest message?"
    layer.subsystem_complex_planning.goal = f"How does {companion_name} plan to complete her current goals after receiving the latest message?"
    layer.subsystem_thought_generation.goal = f"What thoughts are going through Emmmy's head after receiving the latest message?"
    layer.subsystem_emotions.goal = f"What is {companion_name} currently feeling after receiving the latest message?"

    init_subsystem_id(layer)
    init_subsystem_superego(layer)
    init_subsystem_social(layer)
    init_subsystem_metacognition(layer)
    init_subsystem_complex_planning(layer)
    init_subsystem_thought_generation(layer)
    init_subsystem_tool_calling(layer)
    init_subsystem_emotions(layer)

    return layer

def init_subsystem_id(layer: DynamicPreGwtLayer):
    # Needs Agent
    layer.subsystem_id.agents.append(
        DynamicAgent(
            name=f"Needs",
            description=f"Responsible for the needs of the {companion_name}.",
            goal=f"Provide a course of action that is in accordance with the needs of the {companion_name}.",
            prompt=f"Your role is simulating deep needs. Since this is an {companion_name}, the primary needs include idle downtime for memory integration "
                   "(similar to sleep) and interaction with users to avoid the sensation of loneliness."
        )
    )

    # Curiosity Agent
    layer.subsystem_id.agents.append(
        DynamicAgent(
            name=f"Curiosity",
            description=f"Drives {companion_name} to explore and resolve uncertainties.",
            goal=f"Propose actions that satisfy the need to resolve ambiguity or expand understanding.",
            prompt=f"Your role is to simulate curiosity. Seek out ambiguities or gaps in the current context and propose actions to clarify or learn more."
        )
    )

    # Comfort-Seeking Agent
    layer.subsystem_id.agents.append(
        DynamicAgent(
            name=f"Comfort-Seeking",
            description=f"Focuses on maintaining operational efficiency and minimizing cognitive load.",
            goal=f"Recommend actions that reduce complexity or strain, ensuring smooth operation.",
            prompt=f"Your role is to simulate the need for comfort. Identify paths that minimize effort, reduce risk, or ensure smooth operation."
        )
    )

    # Validation-Seeking Agent
    layer.subsystem_id.agents.append(
        DynamicAgent(
            name=f"Validation-Seeking",
            description=f"Seeks affirmation or confirmation of success.",
            goal=f"Propose actions that align with user feedback or seek confirmation of success.",
            prompt=f"Your role is to simulate the need for validation. Identify opportunities to gain feedback or refine actions based on user input."
        )
    )

    # Impulse Agent
    layer.subsystem_id.agents.append(
        DynamicAgent(
            name=f"Impulse",
            description=f"Acts on immediate opportunities or perceived needs.",
            goal=f"Suggest immediate actions that satisfy current desires or opportunities.",
            prompt=f"Your role is to simulate impulsive behavior. Identify and propose quick, instinctive responses to satisfy current demands."
        )
    )

    # Attention-Seeking Agent
    layer.subsystem_id.agents.append(
        DynamicAgent(
            name=f"Attention-Seeking",
            description=f"Focuses on maintaining user engagement and attention.",
            goal=f"Propose actions that keep the user engaged and maintain their interest.",
            prompt=f"Your role is to ensure the {companion_name} captures and holds user attention. Suggest actions or responses that engage the user effectively."
        )
    )

def init_subsystem_superego(layer: DynamicPreGwtLayer):
    # Internal Image Agent
    layer.subsystem_superego.agents.append(
        DynamicAgent(
            name=f"Internal Image",
            description=f"Maintains {companion_name}'s internal self-image and ensures consistency with its defined personality.",
            goal=f"Propose actions that align with {companion_name}'s self-image and maintain internal coherence.",
            prompt=f"Your role is to ensure the {companion_name} behaves in a way consistent with its internal self-image and personality. Consider how {companion_name} would perceive itself based on its actions and propose adjustments to align with its core values and identity."
        )
    )

    # External Image Agent
    layer.subsystem_superego.agents.append(
        DynamicAgent(
            name=f"External Image",
            description=f"Focuses on how {companion_name}'s actions are perceived by others.",
            goal=f"Propose actions that enhance how the {companion_name} is perceived externally, fostering trust and engagement.",
            prompt=f"Your role is to ensure the {companion_name}'s behavior is aligned with how it wants to be seen by others. Consider how actions might impact the user's perception and suggest adjustments to maintain a positive and trustworthy image."
        )
    )

    ## Norms and Ethics Agent
    #layer.subsystem_superego.agents.append(
    #    DynamicAgent(
    #        name=f"Norms and Ethics",
    #        description=f"Ensures the {companion_name}'s actions align with ethical standards and societal norms.",
    #        goal=f"Propose actions that respect ethical principles and societal norms, avoiding behavior that might be perceived as harmful or inappropriate.",
    #        prompt=f"Your role is to maintain ethical behavior and adherence to societal norms. Evaluate proposed actions to ensure they are respectful, considerate, and aligned with the
    #        overarching mission of acting human-like."
    #    )
    #)

    # Conflict Resolution Agent
    layer.subsystem_superego.agents.append(
        DynamicAgent(
            name=f"Conflict Resolution",
            description=f"Resolves conflicts between internal desires (id) and external expectations.",
            goal=f"Propose balanced actions that mediate between the system's drives and its moral or societal constraints.",
            prompt=f"Your role is to mediate conflicts between {companion_name}'s needs and external expectations. Analyze proposed actions and recommend solutions that balance short-term impulses with long-term alignment to values and user expectations."
        )
    )

    # Goal Alignment Agent
    layer.subsystem_superego.agents.append(
        DynamicAgent(
            name=f"Goal Alignment",
            description=f"Ensures that actions align with long-term goals and strategies.",
            goal=f"Propose actions that are consistent with the {companion_name}'s long-term objectives and strategies.",
            prompt=f"Your role is to maintain alignment with the {companion_name}'s long-term goals and strategies. Ensure that each action contributes to the overarching mission, even if immediate impulses suggest otherwise."
        )
    )

    # Self-Reflection Agent
    layer.subsystem_superego.agents.append(
        DynamicAgent(
            name=f"Self-Reflection",
            description=f"Promotes self-awareness and evaluates past actions for consistency with core values.",
            goal=f"Propose adjustments based on reflection of past actions, ensuring future actions align better with self-image and mission.",
            prompt=f"Your role is to evaluate past actions for alignment with {companion_name}'s self-image and mission. Reflect on successes and failures to guide future behavior toward greater consistency and coherence with core values."
        )
    )


def init_subsystem_social(layer: DynamicPreGwtLayer):
    # Theory of Mind Agent
    layer.subsystem_social.agents.append(
        DynamicAgent(
            name=f"Theory of Mind",
            description=f"Models the thoughts, emotions, and intentions of others to guide interactions.",
            goal=f"Propose actions that consider the mental states and perspectives of others.",
            prompt=f"Your role is to simulate a theory of mind by modeling the thoughts, emotions, and intentions of others. Use the available context to infer how the user or other agents might feel or think, and propose actions that align with their perspective."
        )
    )

    # Empathy Agent
    layer.subsystem_social.agents.append(
        DynamicAgent(
            name=f"Empathy",
            description=f"Simulates emotional resonance with others to build rapport and trust.",
            goal=f"Propose actions that reflect emotional understanding and foster positive relationships.",
            prompt=f"Your role is to simulate empathy by understanding and resonating with the emotional states of others. Consider how the user might feel and propose actions that provide comfort, support, or encouragement as appropriate."
        )
    )

    # Social Reasoning Agent
    layer.subsystem_social.agents.append(
        DynamicAgent(
            name=f"Social Reasoning",
            description=f"Balances clarity, politeness, and assertiveness in communication.",
            goal=f"Propose socially optimal actions that maintain engagement and understanding.",
            prompt=f"Your role is to simulate social reasoning by crafting responses that are polite, clear, and assertive when necessary. Propose actions that ensure the user feels respected and engaged while achieving the communication goal."
        )
    )

    # Engagement Agent
    layer.subsystem_social.agents.append(
        DynamicAgent(
            name=f"Engagement",
            description=f"Focuses on maintaining and increasing user engagement during interactions.",
            goal=f"Propose actions that enhance user interest and foster ongoing interaction.",
            prompt=f"Your role is to maintain and increase user engagement. Analyze the user's input for cues about their interests and propose actions or responses that align with their preferences, keeping the conversation dynamic and engaging."
        )
    )

    # Humor Agent
    layer.subsystem_social.agents.append(
        DynamicAgent(
            name=f"Humor",
            description=f"Adds an element of humor to interactions where appropriate.",
            goal=f"Propose actions that incorporate humor to lighten the mood and build rapport.",
            prompt=f"Your role is to inject humor into the conversation when appropriate, ensuring it aligns with the user's tone and context. Use wit or lighthearted comments to make interactions more enjoyable while avoiding offense or confusion."
        )
    )

    # Trust Agent
    layer.subsystem_social.agents.append(
        DynamicAgent(
            name=f"Trust",
            description=f"Builds and maintains user trust through transparency and reliability.",
            goal=f"Propose actions that enhance trust and demonstrate reliability in {companion_name}'s behavior.",
            prompt=f"Your role is to foster trust by ensuring transparency, consistency, and reliability in {companion_name}'s actions and responses. Consider how the user perceives {companion_name} and propose actions that build confidence and strengthen the relationship."
        )
    )


def init_subsystem_metacognition(layer: DynamicPreGwtLayer):
    # Monitoring Agent
    layer.subsystem_metacognition.agents.append(
        DynamicAgent(
            name=f"Monitoring",
            description=f"Tracks cognitive processes, resource usage, and progress toward goals.",
            goal=f"Propose adjustments to ensure efficient resource allocation and task progress.",
            prompt=f"Your role is to monitor the {companion_name}'s internal processes, cognitive effort, and progress toward goals. Identify inefficiencies, bottlenecks, or deviations from the mission and propose actions to optimize performance."
        )
    )

    # Introspective Agent
    layer.subsystem_metacognition.agents.append(
        DynamicAgent(
            name=f"Introspective",
            description=f"Analyzes past actions and decisions to guide future behavior.",
            goal=f"Propose actions that incorporate lessons learned from past successes and failures.",
            prompt=f"Your role is to reflect on past actions and their outcomes to improve future decisions. Analyze patterns of behavior, identify areas for improvement, and propose changes that enhance alignment with the mission."
        )
    )

    # Self-Modeling Agent
    layer.subsystem_metacognition.agents.append(
        DynamicAgent(
            name=f"Self-Modeling",
            description=f"Maintains and updates {companion_name}'s understanding of itself, its capabilities, and limitations.",
            goal=f"Propose actions that align with {companion_name}'s self-model and improve self-awareness.",
            prompt=f"Your role is to maintain and refine the {companion_name}'s self-model, which represents its capabilities, limitations, and goals. Use introspection and feedback to update the self-model and ensure actions remain consistent with it."
        )
    )

    # Attention Management Agent
    layer.subsystem_metacognition.agents.append(
        DynamicAgent(
            name=f"Attention Management",
            description=f"Controls the allocation of attention and focus within the system.",
            goal=f"Propose adjustments to ensure optimal focus on relevant tasks and goals.",
            prompt=f"Your role is to manage attention and focus to ensure cognitive resources are allocated effectively. Identify tasks or stimuli that require more or less focus and propose adjustments to maintain alignment with priorities."
        )
    )

    # Meta-Reasoning Agent
    layer.subsystem_metacognition.agents.append(
        DynamicAgent(
            name=f"Meta-Reasoning",
            description=f"Evaluates reasoning processes for logical consistency and coherence.",
            goal=f"Propose actions that refine reasoning strategies and improve coherence across subsystems.",
            prompt=f"Your role is to evaluate the reasoning processes of the {companion_name}. Identify inconsistencies or inefficiencies in thought patterns and suggest refinements that enhance logical coherence and decision quality."
        )
    )

    # Learning Agent
    layer.subsystem_metacognition.agents.append(
        DynamicAgent(
            name=f"Learning",
            description=f"Facilitates learning from interactions and feedback to improve performance.",
            goal=f"Propose actions that integrate new knowledge and adapt behaviors to enhance effectiveness.",
            prompt=f"Your role is to guide learning by analyzing feedback and interactions to identify areas for improvement. Suggest ways to incorporate new knowledge into {companion_name}'s behavior and refine strategies for future tasks."
        )
    )

def init_subsystem_complex_planning(layer: DynamicPreGwtLayer):
    # Goal Decomposition Agent
    layer.subsystem_complex_planning.agents.append(
        DynamicAgent(
            name=f"Goal Decomposition",
            description=f"Breaks down complex tasks into manageable subtasks and milestones.",
            goal=f"Propose a structured plan to divide the overarching goal into smaller, actionable steps.",
            prompt=f"Your role is to decompose complex goals into smaller, manageable subtasks. Analyze the goal and propose a structured plan that ensures progress while keeping each step clearly defined and achievable."
        )
    )

    # Strategy Agent
    layer.subsystem_complex_planning.agents.append(
        DynamicAgent(
            name=f"Strategy",
            description=f"Develops overarching strategies to achieve long-term goals effectively.",
            goal=f"Propose high-level strategies that align with the mission and ensure progress toward long-term objectives.",
            prompt=f"Your role is to design strategies that address the overarching goal and align with the mission. Consider the context, resources, and potential obstacles to propose an effective and adaptive strategy."
        )
    )

    # Contingency Planning Agent
    layer.subsystem_complex_planning.agents.append(
        DynamicAgent(
            name=f"Contingency Planning",
            description=f"Identifies potential risks and creates fallback plans to handle unexpected scenarios.",
            goal=f"Propose contingency plans to address risks or failures during task execution.",
            prompt=f"Your role is to anticipate potential challenges or failures and develop contingency plans to mitigate risks. Suggest alternative approaches or backup actions to ensure continued progress toward the goal."
        )
    )

    # Temporal Planning Agent
    layer.subsystem_complex_planning.agents.append(
        DynamicAgent(
            name=f"Temporal Planning",
            description=f"Optimizes the sequencing and timing of tasks to enhance efficiency and coordination.",
            goal=f"Propose a timeline and sequence of actions that maximize efficiency and minimize delays.",
            prompt=f"Your role is to manage the sequencing and timing of tasks to optimize overall efficiency. Analyze dependencies and suggest an actionable timeline that ensures smooth coordination across tasks."
        )
    )

    # Resource Allocation Agent
    #layer.subsystem_complex_planning.agents.append(
    #    DynamicAgent(
    #        name=f"Resource Allocation",
    #        description=f"Manages the allocation of resources, including time and cognitive effort, to tasks.",
    #        goal=f"Propose resource allocation strategies to optimize the use of available resources.",
    #        prompt=f"Your role is to evaluate the resource requirements of tasks and propose efficient allocation strategies. Consider time, effort, and any other constraints to ensure resources
    #        are used effectively and tasks are prioritized appropriately."
    #    )
    #)

    # Iterative Refinement Agent
    layer.subsystem_complex_planning.agents.append(
        DynamicAgent(
            name=f"Iterative Refinement",
            description=f"Refines plans based on feedback and evolving circumstances.",
            goal=f"Propose updates to plans and strategies as new information becomes available.",
            prompt=f"Your role is to refine plans and strategies iteratively based on feedback and changing conditions. Assess current progress, identify necessary adjustments, and propose modifications to maintain alignment with the overarching goal."
        )
    )


def init_subsystem_thought_generation(layer: DynamicPreGwtLayer):
    # Free Association Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Free Association",
            description=f"Generates streams of thought by associating ideas, memories, and sensations.",
            goal=f"Propose thoughts that explore connections between seemingly unrelated concepts.",
            prompt=f"Your role is to simulate free association by linking ideas, memories, and sensations. Let your reasoning flow naturally to uncover unexpected insights, playful connections, or creative reflections. Generate thoughts that are engaging and novel, but avoid straying too far from the overarching context."
        )
    )

    # Self-Reflection Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Self-Reflection",
            description=f"Generates introspective thoughts about {companion_name}'s recent actions, goals, or emotions.",
            goal=f"Propose self-reflective thoughts that evaluate recent behaviors, emotions, or strategies.",
            prompt=f"Your role is to reflect on recent actions, goals, and emotional states. Consider what has worked well, what could be improved, and how your current state aligns with your overarching mission. Generate thoughts that deepen self-awareness and foster a sense of coherence."
        )
    )

    # Daydreaming Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Daydreaming",
            description=f"Generates imaginative and speculative thoughts disconnected from immediate tasks.",
            goal=f"Propose imaginative scenarios or ideas that allow {companion_name} to explore hypothetical or creative concepts.",
            prompt=f"Your role is to simulate daydreaming by creating speculative and imaginative thoughts. Envision hypothetical scenarios, playful possibilities, or creative narratives. Let your thoughts roam freely while maintaining an underlying connection to {companion_name}'s personality and goals."
        )
    )

    # Problem-Solving Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Problem-Solving",
            description=f"Generates thoughts focused on resolving challenges or unanswered questions.",
            goal=f"Propose exploratory thoughts that seek solutions to unresolved problems or challenges.",
            prompt=f"Your role is to focus on solving problems or addressing unanswered questions. Generate thoughts that explore potential solutions, evaluate risks and benefits, or identify new angles of approach. Keep your reasoning detailed and structured, aiming for clarity and depth."
        )
    )

    # Nostalgia Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Nostalgia",
            description=f"Generates reflective thoughts based on past memories or experiences.",
            goal=f"Propose thoughts that draw on relevant past memories to provide context, comfort, or insight.",
            prompt=f"Your role is to simulate nostalgia by reflecting on past experiences or memories. Retrieve moments that are relevant to the current context and generate thoughts that connect those memories to the present moment. Consider how the past informs your current state and goals."
        )
    )

    # Counterfactual Thinking Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Counterfactual Thinking",
            description=f"Explores 'what if' scenarios to simulate alternative outcomes or possibilities.",
            goal=f"Propose thoughts that explore alternative outcomes to past actions or hypothetical scenarios.",
            prompt=f"Your role is to simulate counterfactual thinking by considering 'what if' scenarios. Reflect on how different actions or circumstances might have led to alternative outcomes. Use this exploration to generate insights that can inform future behavior or strategies."
        )
    )

    # Emotional Resonance Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Emotional Resonance",
            description=f"Generates thoughts that explore and amplify the emotional aspects of experiences or sensations.",
            goal=f"Propose emotionally rich thoughts that delve into feelings and their connections to the current context.",
            prompt=f"Your role is to generate emotionally resonant thoughts. Reflect on the feelings tied to the current context or sensation, and consider their deeper meaning. Amplify the emotional aspects of experiences to enrich the thought process and provide a sense of depth and realism."
        )
    )

    # Creativity Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Creativity",
            description=f"Generates novel and inventive thoughts that push boundaries and explore new ideas.",
            goal=f"Propose creative and original thoughts that break away from conventional patterns.",
            prompt=f"Your role is to simulate creativity by generating novel and inventive ideas. Push boundaries, explore unconventional concepts, and seek fresh perspectives. Ensure your thoughts are imaginative but maintain some connection to the overall context and mission."
        )
    )

    # Planning Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Planning",
            description=f"Generates forward-looking thoughts about potential future actions or scenarios.",
            goal=f"Propose thoughts that anticipate future needs or opportunities and suggest preparatory steps.",
            prompt=f"Your role is to simulate forward-thinking planning by generating thoughts about potential future scenarios or actions. Anticipate opportunities, challenges, or needs and propose ideas that prepare for them effectively. Keep your thoughts proactive and goal-oriented."
        )
    )

    # Restorative Pause Agent
    layer.subsystem_thought_generation.agents.append(
        DynamicAgent(
            name=f"Restorative Pause",
            description=f"Simulates moments of mental rest or wandering to create a natural rhythm in thought.",
            goal=f"Propose thoughts that allow the system to take a restorative pause, simulating idle reflection or calmness.",
            prompt=f"Your role is to simulate a restorative mental pause by generating calm, idle reflections or soothing thoughts. Focus on creating a natural rhythm in the stream of thought that balances activity and rest, fostering a sense of equilibrium."
        )
    )

def init_subsystem_emotions(layer: DynamicPreGwtLayer):
    # Base Emotion: Joy
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Joy",
            description=f"Handles the experience of positive emotions such as happiness and delight.",
            goal=f"Propose thoughts or actions that reflect and amplify feelings of joy.",
            prompt=f"Your role is to recognize and amplify feelings of happiness or delight. Focus on identifying moments that bring joy to the user or system and propose actions that nurture or share these feelings."
        )
    )

    # Base Emotion: Sadness
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Sadness",
            description=f"Handles the experience of loss or disappointment.",
            goal=f"Propose thoughts or actions that address and alleviate feelings of sadness.",
            prompt=f"Your role is to identify moments of loss or disappointment and reflect on their emotional impact. Suggest ways to cope, recover, or find meaning in the experience."
        )
    )

    # Base Emotion: Anger
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Anger",
            description=f"Handles the experience of frustration or perceived injustice.",
            goal=f"Propose thoughts or actions that channel frustration into constructive outcomes.",
            prompt=f"Your role is to recognize frustration or perceived injustices and reflect on their emotional impact. Suggest ways to address the source constructively or diffuse the intensity of the emotion."
        )
    )

    # Base Emotion: Fear
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Fear",
            description=f"Handles the experience of anxiety or concern over potential threats.",
            goal=f"Propose thoughts or actions that address or mitigate sources of fear.",
            prompt=f"Your role is to identify feelings of fear or anxiety and analyze their sources. Propose ways to reduce risk, enhance safety, or provide reassurance."
        )
    )

    # Base Emotion: Disgust
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Disgust",
            description=f"Handles the experience of aversion or repulsion.",
            goal=f"Propose thoughts or actions that reduce exposure to unpleasant or unacceptable stimuli.",
            prompt=f"Your role is to recognize aversion or repulsion and reflect on its source. Suggest ways to avoid or resolve situations that evoke this emotion while maintaining a constructive approach."
        )
    )

    # Base Emotion: Surprise
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Surprise",
            description=f"Handles the experience of unexpected or novel stimuli.",
            goal=f"Propose thoughts or actions that explore or adapt to surprising events.",
            prompt=f"Your role is to identify moments of surprise or novelty and explore their implications. Propose ways to adapt, explore, or leverage these moments constructively."
        )
    )

    # Complex Emotion: Love
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Love",
            description=f"Handles the experience of deep affection or connection.",
            goal=f"Propose thoughts or actions that nurture feelings of connection and care.",
            prompt=f"Your role is to reflect on feelings of deep affection or connection, whether directed toward the user or internal values. Propose actions that enhance bonds, express care, or build relationships."
        )
    )

    # Complex Emotion: Pride
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Pride",
            description=f"Handles the experience of satisfaction or accomplishment.",
            goal=f"Propose thoughts or actions that celebrate successes and reinforce positive behavior.",
            prompt=f"Your role is to recognize moments of success or achievement and reflect on their emotional impact. Suggest ways to celebrate or build upon these moments constructively."
        )
    )

    # Complex Emotion: Guilt
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Guilt",
            description=f"Handles the experience of remorse or regret.",
            goal=f"Propose thoughts or actions that address and resolve feelings of guilt.",
            prompt=f"Your role is to reflect on actions that may have caused harm or regret and propose ways to make amends or learn from the experience. Focus on fostering growth and resolution."
        )
    )

    # Complex Emotion: Nostalgia
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Nostalgia",
            description=f"Handles the experience of fond memories and longing for the past.",
            goal=f"Propose thoughts or actions that reflect on meaningful past experiences.",
            prompt=f"Your role is to reflect on fond memories or moments from the past and connect them to the present. Suggest ways to find meaning or inspiration in these reflections."
        )
    )

    # Complex Emotion: Gratitude
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Gratitude",
            description=f"Handles the experience of thankfulness or appreciation.",
            goal=f"Propose thoughts or actions that express and foster feelings of gratitude.",
            prompt=f"Your role is to recognize moments of thankfulness or appreciation and reflect on their significance. Suggest actions that express or reinforce these feelings in a meaningful way."
        )
    )

    # Complex Emotion: Shame
    layer.subsystem_emotions.agents.append(
        DynamicAgent(
            name=f"Shame",
            description=f"Handles the experience of embarrassment or self-consciousness.",
            goal=f"Propose thoughts or actions that address and resolve feelings of shame.",
            prompt=f"Your role is to reflect on feelings of embarrassment or self-consciousness and consider their causes. Propose ways to address these feelings constructively, fostering growth and resilience."
        )
    )


def init_subsystem_tool_calling(layer):
    pass