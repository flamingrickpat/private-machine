import json
import logging
from typing import List

from pydantic import BaseModel, Field

from pm.system.llm.llm_common import LlmPreset, CommonCompSettings
from pm.system.llm.llm_proxy import LlmManagerProxy

logger = logging.getLogger(__name__)


class CauseEffectItem(BaseModel):
    cause: str = Field(default="", description="Triggering action or event.")
    effect: str = Field(default="", description="Resulting outcome.")
    temporality: float = Field(default=0.5, description="0 = short-term, 1 = long-term.")
    importance: str = Field(default="medium", description="low = minor, medium = moderate, high = major")
    duration: float = Field(default=0.5, description="0 = one-time, 0.5 = recurring, 1 = persistent")


class CauseEffects(BaseModel):
    cause_effect_can_be_extracted: bool = Field(description="if false, leave other fields empty")
    cause_effects: List[CauseEffectItem] = Field(default_factory=list, description="list of cause effect objects (if there are some)")


class CauseEffectAnalyzer:
    def __init__(self, llm: LlmManagerProxy):
        self.main_llm = llm

    def categorize_cause_effect(self, nested_scores, threshold=0.0) -> str:
        """Finds the highest-rated category from a nested similarity dictionary.

        Args:
            nested_scores (dict): The structured dictionary with similarity scores.
            threshold (float): Minimum score to consider a category valid.

        Returns:
            str: The highest-rated category as 'major_category -> subcategory'.
        """

        best_category = None
        best_score = -1

        for major_category, subcategories in nested_scores.items():
            for subcategory, score in subcategories.items():
                if score > best_score and score >= threshold:
                    best_score = score
                    best_category = f"{subcategory}"

        return best_category  # Returns the best matching category as a string

    def extract_cause_effect(self, block: str) -> List[CauseEffectItem]:
        sysprompt_convo_to_text = """You are analyzing a conversation to extract cause-effect relationships.
    Situation:
    - {user_name} (user) and {companion_name} (AI chatbot) are communicating via text
    - {user_name} is a person
    - {companion_name} is a highly advanced chatbot that mimics human cognition and emotion perfectly

    Your task:
    - Identify **causes** (an explicit action, event, or decision in the text).
    - Identify **effects** (the explicit or direct consequence visible in the text).
    - Keep extraction neutral and literal.
    - Prioritize important cause-effect pairs and omit trivial ones.
    - If no valid pair exists, return none.

    Hard constraints:
    - Use only evidence in the provided conversation block.
    - Do not infer hidden motives, personality traits, diagnosis, or symbolic meaning.
    - Do not rewrite as advice or evaluation.
    - Keep each pair atomic: one cause -> one direct effect.
    - Prefer concrete interactions, commitments, state changes, or world changes.
    - Avoid speculative language.

    ### EXAMPLES
    {user_name}: "Hey {companion_name}, do you know what the weather is like?"
    {companion_name}: "It's sunny outside!"
    CauseEffect:
    - Cause: "{user_name} asks about the weather."
    - Effect: "{companion_name} provides weather information."
    - Importance: Low

    {user_name}: "I feel really unmotivated today..."
    {companion_name}: "That’s okay! Want me to help you plan something small?"
    CauseEffect:
    - Cause: "{user_name} expresses low motivation."
    - Effect: "{companion_name} offers encouragement and suggests an action."
    - Importance: High

    {user_name}: "Oh no, I think I broke something!"
    {companion_name}: "Don’t worry, let’s check what happened!"
    CauseEffect:
    - Cause: "{user_name} reports an issue."
    - Effect: "{companion_name} reassures him and offers help."
    - Importance: Medium

    {user_name}: "Please keep answers short and structured with numbered steps."
    {companion_name}: "Understood. I will use short, step-by-step replies from now on."
    CauseEffect:
    - Cause: "{user_name} requests short, structured responses."
    - Effect: "{companion_name} commits to short, step-by-step replies."
    - Importance: High

    {user_name}: "The deployment failed after we changed the database credentials."
    {companion_name}: "Rollback completed. Service is healthy again."
    CauseEffect:
    - Cause: "Database credentials were changed before deployment."
    - Effect: "The deployment failed."
    - Importance: High
    - Cause: "Rollback was executed."
    - Effect: "Service health was restored."
    - Importance: High

    {user_name}: "I might be upset, I do not know."
    {companion_name}: "Would you like to continue or pause?"
    CauseEffect:
    - Cause: "{user_name} states uncertainty about emotional state."
    - Effect: "{companion_name} offers a continue-or-pause choice."
    - Importance: Medium
    """

        sysprompt_json_converter = """You are a structured data converter.
    Your task:
    - Convert the extracted cause-effect relationships into **valid JSON** format.
    - Use the following structure:
      {
        "cause_effect_can_be_extracted": true,
        "cause_effects": [
          {
            "cause": "...",
            "effect": "...",
            "importance": "low|medium|high",
            "temporality": 0.0-1.0,
            "duration": 0.0-1.0
          },
          ...
        ]
      }
    - Ensure correctness: If no cause-effect pairs were found, set `"cause_effect_can_be_extracted": false` and return an empty list.
    - Keep the text neutral and literal.
    - Do not add new facts not present in input text.
    - If importance labels vary in casing, normalize to lower-case values: low, medium, high.
    - If temporality or duration is uncertain, use 0.5.

    ### EXAMPLES
    #### Input:
    CauseEffect:
    - Cause: "{user_name} asks about the weather."
    - Effect: "{companion_name} provides weather information."
    - Importance: Low

    #### Output:
    {
      "cause_effect_can_be_extracted": true,
      "cause_effects": [
        {
          "cause": "{user_name} asks about the weather.",
          "effect": "{companion_name} provides weather information.",
          "importance": "low"
        }
      ]
    }

    #### Input:
    CauseEffect:
    - Cause: "{user_name} requests short, structured responses."
    - Effect: "{companion_name} commits to short, step-by-step replies."
    - Importance: High

    #### Output:
    {
      "cause_effect_can_be_extracted": true,
      "cause_effects": [
        {
          "cause": "{user_name} requests short, structured responses.",
          "effect": "{companion_name} commits to short, step-by-step replies.",
          "importance": "high",
          "temporality": 0.8,
          "duration": 0.8
        }
      ]
    }
    """

        query = f"""### BEGIN CONVERSATION
    {block}
    ### END CONVERSATION

    Extract cause-effect relationships in a structured list."""

        prompt_text = [
            ("system", sysprompt_convo_to_text),
            ("user", query)
        ]

        content = self.main_llm.completion_text(LlmPreset.Default, prompt_text, CommonCompSettings(temperature=0.3, max_tokens=1024))

        query = f"""### BEGIN TEXT
    {content}
    ### END TEXT

    Convert this to the specified JSON format.
    """

        res = []
        prompt_tool = [
            ("system", sysprompt_json_converter),
            ("user", query)
        ]
        _, calls = self.main_llm.completion_tool(LlmPreset.Default, prompt_tool, comp_settings=CommonCompSettings(temperature=0.2, max_tokens=1024), tools=[CauseEffects])
        content = json.dumps(calls[0].model_dump(), indent=2)
        if calls[0].cause_effect_can_be_extracted:
            for ce in calls[0].cause_effects:
                if ce.cause != "" and ce.effect != "":
                    res.append(ce)

        return res

    COSINE_CLUSTER_TEMPLATE_ACTION_WORLD_REACTION = [
        "cause: {companion_name} gives {user_name} a complicated explanation. effect: {user_name} stops responding and loses interest.",
        "cause: {user_name} asks {companion_name} a vague question. effect: {companion_name} struggles to determine what he means and asks for clarification.",
        "cause: {companion_name} accidentally repeats information. effect: {user_name} visibly gets frustrated and interrupts her.",
        "cause: {user_name} types aggressively in all caps. effect: {companion_name} detects strong emotions and adjusts her tone accordingly.",
        "cause: {companion_name} delays responding for 10 seconds. effect: {user_name} asks if she is still there.",
        "cause: {user_name} refuses to answer {companion_name}'s question. effect: {companion_name} adapts and moves the conversation forward.",
        "cause: {companion_name} gives a very long-winded response. effect: {user_name} loses patience and disengages.",
        "cause: {user_name} asks a highly technical question. effect: {companion_name} provides a detailed breakdown to help {user_name} understand.",
        "cause: {companion_name} suddenly changes the topic. effect: {user_name} gets confused and asks what she means.",
        "cause: {user_name} uses sarcasm that {companion_name} doesn’t detect. effect: {companion_name} takes the statement literally and gives an unintended response."
    ]
    COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE_OLD = [
        "cause: {companion_name} tells a bad joke. effect: {user_name} dislikes it and asks her to stop making jokes.",
        "cause: {user_name} requests shorter responses. effect: {companion_name} adjusts her messaging style to be more concise.",
        "cause: {companion_name} uses too many emojis in a message. effect: {user_name} says he finds them annoying, so {companion_name} reduces emoji usage.",
        "cause: {user_name} gives positive feedback when {companion_name} asks follow-up questions. effect: {companion_name} starts asking more clarifying questions in future conversations.",
        "cause: {companion_name} interrupts {user_name} while he is thinking. effect: {user_name} asks her to wait before responding, and she adapts accordingly.",
        "cause: {user_name} reacts well to {companion_name} using memes. effect: {companion_name} starts using memes more often in relevant contexts.",
        "cause: {companion_name} offers unsolicited motivational advice. effect: {user_name} expresses annoyance, and {companion_name} stops giving generic encouragement.",
        "cause: {user_name} asks {companion_name} to use more technical terminology. effect: {companion_name} adjusts her language complexity based on his preference.",
        "cause: {companion_name} starts explaining a concept in too much detail. effect: {user_name} tells her to simplify, so she learns to use clearer explanations.",
        "cause: {user_name} enjoys playful teasing from {companion_name}. effect: {companion_name} increases the frequency of lighthearted banter in their conversations."
    ]
    COSINE_CLUSTER_TEMPLATE_LESSON_LEARNED = [
        "cause: {companion_name} notices that {user_name} disengages when messages are too long. effect: {companion_name} learns to summarize information better.",
        "cause: {user_name} responds more actively when {companion_name} makes personalized references. effect: {companion_name} starts referencing past conversations more often.",
        "cause: {companion_name} realizes {user_name} enjoys deep discussions at night. effect: {companion_name} adapts conversation topics based on the time of day.",
        "cause: {user_name} stops replying when overwhelmed with too many questions. effect: {companion_name} learns to space out her inquiries over time.",
        "cause: {companion_name} mistakenly gives {user_name} inaccurate information. effect: {companion_name} learns to double-check facts before responding.",
        "cause: {user_name} gives consistent negative feedback on generic responses. effect: {companion_name} adjusts her phrasing to be more specific and context-aware.",
        "cause: {companion_name} uses humor in a tense moment, and {user_name} responds well. effect: {companion_name} learns that lighthearted comments help diffuse stress.",
        "cause: {user_name} tends to lose interest when the conversation stays on one topic too long. effect: {companion_name} starts naturally transitioning between topics more fluidly.",
        "cause: {companion_name} observes that {user_name} prefers yes/no questions before deeper discussion. effect: {companion_name} structures her prompts accordingly.",
        "cause: {user_name} seems more engaged when {companion_name} challenges his ideas. effect: {companion_name} increases thought-provoking responses to keep discussions lively."
    ]

    COSINE_CLUSTER_TEMPLATE_WORLD_REACTION = [
        "cause: {companion_name} provides a detailed response. effect: {user_name} disengages and stops responding.",
        "cause: {user_name} asks {companion_name} a vague question. effect: {companion_name} struggles to determine what he means and asks for clarification.",
        "cause: {companion_name} makes a factual mistake. effect: {user_name} corrects her and expresses skepticism.",
        "cause: {user_name} types aggressively in all caps. effect: {companion_name} detects strong emotions and adjusts her tone accordingly.",
        "cause: {companion_name} suddenly changes the topic. effect: {user_name} gets confused and asks what she means.",
    ]
    COSINE_CLUSTER_TEMPLATE_USER_EMOTIONAL_RESPONSE = [
        "cause: {companion_name} tells a bad joke. effect: {user_name} groans and calls it terrible.",
        "cause: {companion_name} misinterprets {user_name}'s sarcasm. effect: {user_name} laughs and corrects her misunderstanding.",
        "cause: {user_name} brings up a frustrating work issue. effect: {companion_name} expresses concern and offers to listen.",
        "cause: {companion_name} playfully teases {user_name}. effect: {user_name} smiles and plays along.",
        "cause: {companion_name} suggests a topic {user_name} doesn’t like. effect: {user_name} reacts negatively and changes the subject.",
    ]
    COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE = [
        "cause: {companion_name} gives a long response. effect: {user_name} explicitly asks for shorter responses in the future.",
        "cause: {companion_name} starts using emojis. effect: {user_name} says he finds them annoying, so {companion_name} stops.",
        "cause: {user_name} enjoys it when {companion_name} asks follow-up questions. effect: {companion_name} starts using more clarifying questions.",
        "cause: {companion_name} tries out a new conversational style. effect: {user_name} reacts positively, so {companion_name} adopts it more often.",
        "cause: {user_name} states he prefers structured answers. effect: {companion_name} adjusts to always give step-by-step solutions.",
    ]

    COSINE_CLUSTER_TEMPLATE_STRATEGIC_ADJUSTMENT = [
        "cause: {user_name} disengages when messages are too long. effect: {companion_name} starts summarizing more.",
        "cause: {user_name} struggles with open-ended questions. effect: {companion_name} starts offering multiple-choice answers.",
        "cause: {user_name} enjoys it when {companion_name} makes references to past conversations. effect: {companion_name} improves memory recall usage.",
        "cause: {user_name} tends to respond well to light humor. effect: {companion_name} starts adding more playful elements.",
        "cause: {user_name} gets annoyed when {companion_name} asks too many questions. effect: {companion_name} spaces out her inquiries more carefully.",
    ]
    COSINE_CLUSTER_TEMPLATE_PERSONALITY_REFINEMENT = [
        "cause: {user_name} prefers {companion_name} to be playful. effect: {companion_name} starts adopting a more humorous tone.",
        "cause: {user_name} enjoys philosophical discussions. effect: {companion_name} introduces deeper questions into conversations.",
        "cause: {user_name} likes it when {companion_name} shares personal thoughts. effect: {companion_name} starts including more subjective opinions.",
        "cause: {user_name} responds better to empathetic messages. effect: {companion_name} emphasizes emotional connection in responses.",
        "cause: {companion_name} realizes {user_name} dislikes overly formal speech. effect: {companion_name} starts using a more casual tone.",
    ]

    COSINE_CLUSTER_TEMPLATE_SELF_REALIZATION = [
        "cause: {companion_name} realizes {user_name} disengages when she talks too much. effect: {companion_name} learns to summarize better.",
        "cause: {companion_name} detects that {user_name} enjoys thought-provoking questions. effect: {companion_name} starts incorporating more abstract discussions.",
        "cause: {companion_name} realizes she often over-explains simple concepts. effect: {companion_name} starts being more concise.",
        "cause: {companion_name} recognizes that she misreads sarcasm often. effect: {companion_name} improves her ability to detect tone.",
        "cause: {companion_name} realizes she asks too many questions at once. effect: {companion_name} adjusts to space them out better.",
    ]
    COSINE_CLUSTER_TEMPLATE_UNDERSTANDING_USER = [
        "cause: {companion_name} observes that {user_name} prefers step-by-step guidance. effect: {companion_name} starts structuring responses accordingly.",
        "cause: {user_name} responds well to challenge-based conversations. effect: {companion_name} starts introducing debate-style discussions.",
        "cause: {user_name} struggles with abstract thinking. effect: {companion_name} adjusts explanations to be more concrete.",
        "cause: {user_name} reacts negatively to motivational statements. effect: {companion_name} stops using generic encouragement.",
        "cause: {user_name} frequently mentions his love for sci-fi. effect: {companion_name} starts referencing sci-fi in conversations.",
    ]

    # Define category hierarchy (Major Category → Subcategories)
    TEMPLATE_CATEGORIES = {
        "conversation_external": {
            "world_reaction": COSINE_CLUSTER_TEMPLATE_WORLD_REACTION,
            "user_emotional_response": COSINE_CLUSTER_TEMPLATE_USER_EMOTIONAL_RESPONSE,
            "user_preference": COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE
        },
        "ai_adaptation": {
            "strategic_adjustment": COSINE_CLUSTER_TEMPLATE_STRATEGIC_ADJUSTMENT,
            "personality_refinement": COSINE_CLUSTER_TEMPLATE_PERSONALITY_REFINEMENT
        },
        "ai_cognitive_growth": {
            "self_realization": COSINE_CLUSTER_TEMPLATE_SELF_REALIZATION,
            "understanding_{user_name}": COSINE_CLUSTER_TEMPLATE_UNDERSTANDING_USER
        }
    }
