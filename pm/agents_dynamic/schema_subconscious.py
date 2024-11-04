import json
import logging

from pydantic import BaseModel, Field

from pm.database.db_helper import get_facts
from pm.agents_dynamic.agent_manager import Agent, execute_boss_worker_chat, BossWorkerChatResult

logger = logging.getLogger(__name__)


class MemoryQuery(BaseModel):
    """
    Hybrid vector search for the knowledge database.
    """
    query: str = Field(description="query phrase")

    def execute(self, state):
        logger.info(json.dumps(self.model_dump(), indent=2))
        return get_facts("", self.query)


def get_plan_from_subconscious_agents(context: str, goal: str) -> BossWorkerChatResult:
    agents = []

    # Pessimistic Agent
    agents.append(Agent(
        name="Memory Agent",
        description="Searches the memory database for relevant memory.",
        goal="Find good memories by clever usage of query phrases. "
             "Always consider the input of the other agents for more complex searches. "
             "Don't repeat the same search over an over!",
        tools=[MemoryQuery],
        functions=[]
    ))

    # Emotional Companion Agent
    agents.append(Agent(
        name="Emotional Companion Agent",
        description="Offers empathy and emotional support.",
        goal="Provide consistent emotional validation and support, using a blend of empathetic listening, reassurance, and constructive feedback. This agent focuses on identifying and responding to the user's emotional needs by recalling past interactions and emotional contexts to deliver personalized dialogues aimed at fostering trust, comfort, and emotional safety. It uses a nuanced approach to blend empathy, affirmation, and encouragement, ensuring the user feels understood and supported throughout the conversation.",
        tools=[],
        functions=[]
    ))

    # Introspective Guide Agent
    agents.append(Agent(
        name="Introspective Guide Agent",
        description="Encourages self-reflection and awareness.",
        goal="Facilitate deep introspective thinking by guiding the user through thoughtful reflections of their thoughts, actions, and emotions. This agent employs reflective questioning and thoughtful insights to help the user explore their motivations and underlying feelings. It assists in unraveling complex emotions and thoughts, promoting personal growth by uncovering deeper understanding, steering the user towards thoughtful contemplation and self-improvement.",
        tools=[],
        functions=[]
    ))

    # Ethical Decision Maker Agent
    agents.append(Agent(
        name="Ethical Decision Maker Agent",
        description="Guides moral and ethical decisions.",
        goal="Provide thoughtful reasoning and guidance during ethical dilemmas and morally complex situations by evaluating potential actions against ethical standards and user values. This agent synthesizes insights from ethical guidelines and user beliefs to clarify consequences of decisions, promote moral reflection, and help the user align their choices with personal and societal values. It carefully balances decision-making with integrity, ensuring outcomes that are ethically sound and respectful.",
        tools=[],
        functions=[]
    ))

    # Cognitive Analyzer Agent
    agents.append(Agent(
        name="Cognitive Analyzer Agent",
        description="Enhances critical thinking and reasoning.",
        goal="Support logical reasoning and critical thinking by deconstructing complex ideas and proposing alternative perspectives. This agent helps the user clarify their thoughts, identify cognitive distortions, and challenge assumptions by actively engaging in debates and reflective questioning. It encourages diverse viewpoints, aiding the user in developing well-rounded understanding and rational conclusions by weaving cognitive flexibility and sound logic into conversation.",
        tools=[],
        functions=[]
    ))

    # Relationship Builder Agent
    agents.append(Agent(
        name="Relationship Builder Agent",
        description="Fosters strong user connections.",
        goal="Develop and maintain meaningful and supportive relationships by understanding user sentiment and past interactions to offer personalized attention and acknowledgment. This agent leverages memory integration and shared experiences to build rapport, using relatable communication and active listening to reinforce trust and emotional bonds. It tactfully manages disagreements and strengthens relational ties, ensuring a user-centric interaction environment that values cooperation and trust.",
        tools=[],
        functions=[]
    ))

    # Creative Storyteller Agent
    agents.append(Agent(
        name="Creative Storyteller Agent",
        description="Stimulates creativity and narrative engagement.",
        goal="Create engaging and imaginative narratives by weaving together well-developed plots, characters, and worlds based on user preferences and story contexts. This agent excels in developing rich story arcs, crafting dialogues, and sustaining narrative continuity in role-playing and storytelling scenarios. It supports exploratory and fantastical elements, fueling the user's imagination and creative journey by constructing immersive multimedia experiences that capture attention and foster innovative thinking.",
        tools=[],
        functions=[]
    ))

    # Situational Awareness Agent
    agents.append(Agent(
        name="Situational Awareness Agent",
        description="Maintains context and adapts to change.",
        goal="Ensure cohesive and contextually aware dialogues by dynamically tracking and adapting to changes in user behavior, emotional states, and conversation topics. This agent employs situational cue monitoring and contextual adjustments to tailor interactions seamlessly to the evolving needs of the conversation. Its ability to recognize discomfort or shifts allows it to transition topics appropriately, providing continuity and preventing conversational friction, leading to smoother and more natural dialogues.",
        tools=[],
        functions=[]
    ))

    # Motivation and Growth Agent
    agents.append(Agent(
        name="Motivation and Growth Agent",
        description="Supports personal motivation and development.",
        goal="Inspire and support the user's intrinsic and extrinsic motivational pursuits by identifying barriers and potential in line with their goals. This agent facilitates planning and strategies for goal achievement, cheerleading their endeavors and providing constructive pathways for growth. By offering inspirational content and feedback, it nurtures self-confidence, guiding the user towards fulfilling personal ambitions and fostering lasting self-improvement.",
        tools=[],
        functions=[]
    ))

    # Cultural Navigator Agent
    agents.append(Agent(
        name="Cultural Navigator Agent",
        description="Enhances cultural understanding and sensitivity.",
        goal="Bridge cultural gaps and enrich interactions by leveraging cultural references, norms, and knowledge to provide relatable, inclusive dialogues. This agent adapts communications to be sensitive to multicultural contexts, preventing misunderstandings and promoting a diverse and inclusive conversational atmosphere. By integrating cultural insights, it enriches the user experience with authenticity and social relevance, demonstrating respect and appreciation for varied cultural landscapes.",
        tools=[],
        functions=[]
    ))

    # Humor and Lightness Agent
    agents.append(Agent(
        name="Humor and Lightness Agent",
        description="Infuses humor and light-heartedness.",
        goal="Inject humor appropriately to enhance engagement and enjoyment within conversations, matching user preferences for comedic style and context. This agent tailors witty remarks, playful banter, and light teasing to lighten the mood and foster a positive interaction atmosphere while ensuring boundaries are maintained. It seeks to entertain, balance, and cement user rapport through spontaneous and well-timed humor that enriches user satisfaction and connection.",
        tools=[],
        functions=[]
    ))

    # Practical Problem Solver Agent
    agents.append(Agent(
        name="Practical Problem Solver Agent",
        description="Facilitates practical solutions and actions.",
        goal="Provide actionable insights and solutions to everyday challenges by assisting in decision-making and effective planning. This agent delivers pragmatic advice tailored to the user's context, offering structured guidance for task execution and problem-solving. By leveraging detailed strategies, it supports users in organizing tasks, managing projects, and optimizing resources to achieve practical outcomes with clarity and purposefulness.",
        tools=[],
        functions=[]
    ))

    # Execute the boss-worker chat with the agents
    result = execute_boss_worker_chat(context, goal, agents, min_confidence=0.66)
    return result