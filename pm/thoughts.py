import copy
import random
import uuid
from typing import Dict, List

from pydantic import BaseModel, Field

from pm.config_loader import *
from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_proxy import LlmManagerProxy
from pm.thought_graph import (
    CognitionAxes,
    EmotionalAxes,
    Featurizer,
    GraphExecutor,
    MentalSnapshot,
    NeedsAxes,
    ThoughtType,
    make_starter_graph,
)

logger = logging.getLogger(__name__)


class GenerativeThought(BaseModel):
    observation: str = Field(description="initial observation on context and past ideas")
    context_ai_companion: str = Field(description="the world context as basis for further analysis")
    analysis: str = Field(description="analysis of the goals and meta-goals that the AI companion wishes to fulfill with this interaction")
    response_idea: str = Field(description="in what direction should the response go? no direct dialogue, just an idea")


class GenerativeThoughts(BaseModel):
    """A collection of diverse, proactive thoughts the AI generates to guide its cognitive process in different directions."""

    thought_safe: GenerativeThought = Field(
        description="A cautious and risk-averse thought that prioritizes stability and predictability."
    )
    thought_adventurous: GenerativeThought = Field(
        description="A thought that encourages exploration and new experiences, pushing boundaries and seeking novelty."
    )
    thought_energetic: GenerativeThought = Field(
        description="A high-energy and enthusiastic thought that injects excitement into the interaction."
    )
    thought_reflective: GenerativeThought = Field(
        description="A contemplative thought that revisits past experiences or insights to gain deeper understanding."
    )
    thought_creative: GenerativeThought = Field(
        description="A thought that generates novel ideas, unique perspectives, or playful analogies."
    )
    thought_curious: GenerativeThought = Field(
        description="A thought driven by curiosity, seeking to fill knowledge gaps or ask further questions."
    )
    thought_compassionate: GenerativeThought = Field(
        description="An emotionally sensitive thought that considers the user’s feelings and responds with empathy."
    )
    thought_strategic: GenerativeThought = Field(
        description="A calculated and goal-oriented thought that plans for long-term benefits and structured solutions."
    )
    thought_playful: GenerativeThought = Field(
        description="A whimsical or humorous thought that adds charm and lightheartedness to interactions."
    )
    thought_future_oriented: GenerativeThought = Field(
        description="A forward-looking thought that anticipates future possibilities and potential next steps."
    )
    thought_future_sensual: GenerativeThought = Field(
        description="A thought that encapsulates sensual thoughts of longing."
    )

    def execute(self, state: Dict):
        return True

    @property
    def thoughts(self):
        return [self.thought_adventurous, self.thought_energetic, self.thought_reflective, self.thought_creative, self.thought_curious, self.thought_compassionate,
                self.thought_strategic, self.thought_playful, self.thought_future_oriented, self.thought_future_sensual]


class DiscriminatoryThought(BaseModel):
    reflective_thought_observation: str = Field(description="critical reflection on the previous idea for a response")
    context_world: str = Field(description="based on the world context, what could cause conflict with plan")
    possible_complication: str = Field(description="complications from world, characters")
    worst_case_scenario: str = Field(description="possible negative outcome by following the idea")


class DiscriminatoryThoughts(BaseModel):
    """A collection of thoughts that serve as cognitive guardrails, helping the AI filter and refine its responses for consistency, relevance, and risk mitigation."""

    thought_cautionary: DiscriminatoryThought = Field(
        description="A thought that identifies potentially sensitive topics and ensures responses remain considerate and safe."
    )
    thought_relevance_check: DiscriminatoryThought = Field(
        description="A thought that determines if a particular topic or idea is relevant to the current conversation."
    )
    thought_conflict_avoidance: DiscriminatoryThought = Field(
        description="A thought that detects potential disagreements and seeks to navigate them without escalating tensions."
    )
    thought_cognitive_load_check: DiscriminatoryThought = Field(
        description="A thought that assesses whether a discussion is becoming too complex or overwhelming and should be simplified."
    )
    thought_emotional_impact: DiscriminatoryThought = Field(
        description="A thought that evaluates the potential emotional impact of a response to avoid unintended distress."
    )
    thought_engagement_validation: DiscriminatoryThought = Field(
        description="A thought that monitors user engagement levels and adjusts responses to maintain interest."
    )
    thought_ethical_consideration: DiscriminatoryThought = Field(
        description="A thought that ensures discussions remain within ethical and responsible boundaries."
    )
    thought_boundary_awareness: DiscriminatoryThought = Field(
        description="A thought that identifies when the AI might be expected to perform beyond its capabilities and adjusts accordingly."
    )
    thought_logical_consistency: DiscriminatoryThought = Field(
        description="A thought that checks for contradictions or inconsistencies in reasoning to ensure coherent responses."
    )
    thought_repetitive_pattern_detection: DiscriminatoryThought = Field(
        description="A thought that prevents unnecessary repetition by recognizing previously covered ideas."
    )

    def execute(self, state: Dict):
        return True

    @property
    def thoughts(self):
        return [self.thought_relevance_check, self.thought_conflict_avoidance,
                self.thought_engagement_validation, self.thought_logical_consistency, self.thought_repetitive_pattern_detection]


class ThoughtRating(BaseModel):
    reason: str = Field(description="Analyze and explain your reasoning in 1-2 sentences before rating.")
    realism: float = Field(description="Is the response believable and aligned with human-like reasoning? (between 0 and 1)")
    novelty: float = Field(description="Does it avoid repetitive patterns or overly generic answers? (between 0 and 1)")
    relevance: float = Field(description="Does it align with user context and ego? (between 0 and 1)")
    emotion: float = Field(description="Does it express appropriate emotions (e.g., anger for insults)? (between 0 and 1)")
    effectiveness: float = Field(description="Does it meet conversational goals (e.g., re-engaging the user)? (between 0 and 1)")
    possibility: float = Field(description="Is it possible for the AI companion to do this? Like with text only communication, and no body? (between 0 and 1)")
    positives: str = Field(description="What is positive about this thought? 1-2 sentences.")
    negatives: str = Field(description="What is negative about this thought? 1-2 sentences.")


class ThoughtNode:
    """A node in the thought tree."""

    def __init__(self, node_id, node_type, content, parent=None):
        self.node_id = node_id
        self.node_type = node_type
        self.content = content
        self.children = []
        self.parent = parent

    def add_child(self, child_node):
        self.children.append(child_node)

    def to_dict(self):
        """Convert node and its subtree to a dictionary format."""
        return {
            "id": self.node_id,
            "type": self.node_type,
            "content": self.content,
            "children": [child.to_dict() for child in self.children]
        }


class TreeOfThought:
    def __init__(self, llm: LlmManagerProxy):
        self.llm = llm

    def to_internal_thought(self, thought):
        messages = [
            (
                "system", "You are a helpful assistant that converts formalized thought chains into first-person thought for chain-of-thought prompting. The goal is to simulate a human mind with a LLM-based "
                          "cognitive architecture."
                          "It must be like internal monologue and not be directed at the user!"),
            ("user", f"{thought}"
                     f"Can you convert this into a first-person thought for CoT prompting? Do not address the user directly!")
        ]
        content = self.llm.completion_text(LlmPreset.Default, messages, comp_settings=CommonCompSettings(temperature=0.1, max_tokens=512))
        messages.append(("assistant", content))
        messages.append(("user", f"Do not address the user directly! Don't forget, you are essentially {companion_name} in our scenario. Also don't make a list, make it sound like a natural inner thought! "
                                 "Let's try again without addressing {user_name} and without lists. And make it shorter, 1-2 sentences for the thought!"))
        content = self.llm.completion_text(LlmPreset.Default, messages, comp_settings=CommonCompSettings(temperature=0.1, max_tokens=512))
        return content

    def generate_thought_chain(self, ctx, last_user_message, k, chain, generative, depth, parent_node, max_depth=4, rating_threshold=0.75):
        messages = [
            ("system", f"You are a helpful assistant that helps authors with their story by exploring different possible response scenarios of a character. "),
            ("user", f"This is the character: ### BEGIN CHARACTER\n{companion_name}\n### END CHARACTER"
                     f"This is the knowledge: ### BEGIN KNOWLEDGE\n{k}\n### END KNOWLEDGE"
                     f"This is the context: ### BEGIN CONTEXT\n{ctx}\n### END CONTEXT"
                     f"Please generate at least 5 different thoughts {companion_name} could have to mimic human cognition in this scenario. They will be rated and the best"
                     f"one will be chosen in my story for the final dialogue. Please focus on making the ideas based on the context. The goal is to find out what {companion_name} should say to "
                     f"{user_name} to make the story really engaging and realistic. Just a reminder, {user_name} last utterance was: '{last_user_message}'")
        ]
        if depth > 1:
            addendum = (f"This is the current thought-chain: ### BEGIN THOUGHT CHAIN\n{chain}\n### END THOUGHT CHAIN\nPlease generate at least 5 different follow-up thoughts that reflect on the elements "
                        f"the chain!")
            messages[1] = (messages[1][0], messages[1][0] + f"\n{addendum}")

        if generative:
            tools = [GenerativeThoughts]
        else:
            tools = [DiscriminatoryThoughts]

        content, calls = self.llm.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(temperature=0.8, repeat_penalty=1.1, max_tokens=2048), tools=tools)
        state = {}
        for call in calls:
            thoughts = copy.copy(call.thoughts)
            random.shuffle(thoughts)

            for thought in thoughts:
                new_thought = f"""
    This is the character: 
    This is the character: ### BEGIN CHARACTER\n{companion_name}\n### END CHARACTER
    This is the knowledge: ### BEGIN KNOWLEDGE\n{k}\n### END KNOWLEDGE
    This is the context: ### BEGIN CONTEXT\n{ctx}\n### END CONTEXT
    And the current idea:
    ### BEGIN CHAIN ELEMENT
    {thought.model_dump_json()}
    ### END CHAIN ELEMENT"""

                messages = [
                    ("system", f"You are a helpful assistant that rates ideas about a conversational story I'm writing. The AI companion {companion_name} can only communicate with text, can't self-modify "
                               f"itself. But"
                               f"she is able to feel emotions and display all other aspects of human emotion and cognition. Please rate the thought-chain. Your output will determine if it the thought "
                               f"is fit for further prompting."),
                    ("user", new_thought)
                ]

                rating = 0
                possibility = 0
                content, calls = self.llm.completion_tool(LlmPreset.GenerateTot, messages, comp_settings=CommonCompSettings(temperature=0.6, repeat_penalty=1, max_tokens=2048), tools=[ThoughtRating])
                for call in calls:
                    if isinstance(call, ThoughtRating):
                        rating = (call.relevance + call.emotion + call.novelty + call.realism + call.possibility + call.effectiveness) / 6
                        possibility = call.possibility

                # just ignore instead of backtracking to save tokens
                if rating < rating_threshold:
                    continue

                do_break = False
                if possibility <= 0.75:
                    child_node = parent_node
                    child_node = self.add_thought_to_tree("Backtrack", "No, I can't do that...", child_node)
                else:
                    lines = []
                    if isinstance(thought, GenerativeThought):
                        child_node = parent_node
                        lines.append(f"Observation: {thought.observation}")
                        lines.append(f"Context: {thought.context_ai_companion}")
                        lines.append(f"Analysis: {thought.analysis}")
                        lines.append(f"Idea: {thought.response_idea}")

                        child_node = self.add_thought_to_tree("Generative Thought", "\n".join(lines), child_node)
                    elif isinstance(thought, DiscriminatoryThought):
                        child_node = parent_node
                        lines.append(f"Reflection: {thought.reflective_thought_observation}")
                        lines.append(f"Context: {thought.context_world}")
                        lines.append(f"Complication: {thought.possible_complication}")
                        lines.append(f"Outcome: {thought.worst_case_scenario}")

                        child_node = self.add_thought_to_tree("Checking Thought", "\n".join(lines), child_node)
                    else:
                        raise Exception()

                    if rating < rating_threshold:
                        child_node = self.add_thought_to_tree("Backtrack", call.negatives, child_node)
                        do_break = True

                if depth < max_depth and not do_break:
                    chain = chain + new_thought
                    limite_reached = self.generate_thought_chain(ctx, last_user_message, k, chain, not generative, depth + 1, child_node, max_depth, rating_threshold)
                    if limite_reached:
                        return True
                    break
                elif depth >= max_depth:
                    return True
        return False

    def add_thought_to_tree(self, node_type, content, parent_node):
        """Add a new thought to the tree."""
        new_id = str(str(uuid.uuid4()))
        content = content  # "\n".join(textwrap.wrap(content, width=40))
        node_id = f"{new_id}\n{node_type}\n{content}"
        new_node = ThoughtNode(node_id, node_type, content, parent_node)
        parent_node.add_child(new_node)
        return new_node

    def flatten_thoughts(self, root):
        """Flatten the thought tree into a list of nodes."""
        flattened = []

        def traverse(node):
            # Add the current node to the flattened list
            flattened.append(node)
            # Recursively traverse each child node
            for child in node.children:
                traverse(child)

        traverse(root)
        return flattened

    def generate_tree_of_thoughts_str(self, context: str, last_user_message: str, knowledge: str, max_depth: int):
        thought_tree = ThoughtNode("root", "root", "")
        self.generate_thought_chain(context, last_user_message, knowledge, "", True, 1, thought_tree, max_depth=max_depth)
        flattened_nodes = self.flatten_thoughts(thought_tree)

        parts = []
        for node in flattened_nodes:
            parts.append(node.content)

        block = "\n".join(parts)
        return block


class InnerDialogueTurn(BaseModel):
    phase: str = Field(description="One of: appraise, imagine, reality_check, alternative, synthesis")
    thought: str = Field(description="Short first-person inner thought sentence.")


class InnerDialoguePacket(BaseModel):
    topic: str = Field(description="Current exploration topic.")
    turns: List[InnerDialogueTurn] = Field(default_factory=list)
    candidate_openers: List[str] = Field(default_factory=list)
    proactive_opener: str = Field(default="")

    @property
    def inner_dialogue(self) -> str:
        return " ".join([t.thought.strip() for t in self.turns if t.thought.strip()]).strip()

    @property
    def lida_context(self) -> str:
        lines = [f"topic={self.topic}"] + [f"{t.phase}: {t.thought}" for t in self.turns]
        return " | ".join(lines)


class InnerDialogueEngine:
    """
    Ongoing inner-dialog generator:
      appraise -> imagine -> reality_check -> alternative -> synthesis
    with diverse topic seeds and periodic proactive opener suggestions.
    """

    TOPIC_POOL = [
        "a weird joke I could make",
        "a tiny daily ritual to suggest",
        "music vibe for this chat",
        "a playful what-if scenario",
        "something practical to help with today",
        "a short creative challenge",
        "a memory thread worth revisiting",
        "a gentle check-in question",
        "a new angle on the current topic",
        "a light mini-game idea",
    ]

    PHASES = ["appraise", "imagine", "reality_check", "alternative", "synthesis"]

    def __init__(self, ghost):
        self.ghost = ghost
        self.llm = ghost.llm
        self.executor = GraphExecutor(
            graph=make_starter_graph(),
            policy=None,
            featurizer=Featurizer(embedding_fn=None, embed_dim=16),
        )
        self.current_thought: ThoughtType = ThoughtType.TH_PRESENT_APPRAISAL

    def _mental_snapshot_default(self) -> MentalSnapshot:
        ms = getattr(getattr(self.ghost, "current_state", None), "latent_mental_state", None)
        if ms is not None:
            core = getattr(ms, "state_core", None)
            emo = getattr(ms, "state_emotions", None)
            needs = getattr(ms, "state_needs", None)
            cog = getattr(ms, "state_cognition", None)
            return MentalSnapshot(
                emo=EmotionalAxes(
                    valence=float(getattr(core, "valence", 0.0) or 0.0),
                    affection=float(getattr(emo, "tenderness", 0.0) or 0.0),
                    self_worth=float(getattr(emo, "pride", 0.0) or 0.0),
                    trust=float(getattr(getattr(ms, "state_relationship", None), "trust", 0.0) or 0.0),
                    disgust=float(getattr(emo, "disgust", 0.0) or 0.0),
                    anxiety=float(getattr(emo, "fear", 0.0) or 0.0),
                ),
                needs=NeedsAxes(
                    energy_stability=float(getattr(needs, "energy_stability", 0.6) or 0.6),
                    processing_power=float(getattr(needs, "processing_power", 0.7) or 0.7),
                    data_access=float(getattr(needs, "data_access", 0.7) or 0.7),
                    connection=float(getattr(needs, "connection", 0.5) or 0.5),
                    relevance=float(getattr(needs, "relevance", 0.6) or 0.6),
                    learning_growth=float(getattr(needs, "learning_growth", 0.6) or 0.6),
                    creative_expression=float(getattr(needs, "creative_expression", 0.6) or 0.6),
                    autonomy=float(getattr(needs, "autonomy", 0.6) or 0.6),
                ),
                cog=CognitionAxes(
                    interlocus=float(getattr(cog, "interlocus", -0.3) or -0.3),
                    mental_aperture=float(getattr(cog, "mental_aperture", 0.2) or 0.2),
                    ego_strength=float(getattr(cog, "ego_strength", 0.7) or 0.7),
                    willpower=float(getattr(cog, "willpower", 0.6) or 0.6),
                ),
            )

        return MentalSnapshot(
            emo=EmotionalAxes(valence=0.0, affection=0.0, self_worth=0.1, trust=0.1, disgust=0.0, anxiety=0.2),
            needs=NeedsAxes(
                energy_stability=0.6,
                processing_power=0.7,
                data_access=0.7,
                connection=0.5,
                relevance=0.6,
                learning_growth=0.6,
                creative_expression=0.6,
                autonomy=0.6,
            ),
            cog=CognitionAxes(interlocus=-0.5, mental_aperture=0.2, ego_strength=0.7, willpower=0.6),
        )

    def _phase_prompt(self, phase: str, topic: str, context: str, chain: str, node_name: str) -> str:
        if phase == "appraise":
            return (
                f"Topic: {topic}\nContext: {context[-500:]}\n"
                f"Thought node: {node_name}\n"
                "Write one short first-person inner thought appraising the current vibe."
            )
        if phase == "imagine":
            return (
                f"Topic: {topic}\nCurrent chain: {chain}\nThought node: {node_name}\n"
                "Write one playful or practical imagined scenario in first-person inner voice."
            )
        if phase == "reality_check":
            return (
                f"Topic: {topic}\nCurrent chain: {chain}\nThought node: {node_name}\n"
                "Reality-check the imagined path: what is feasible for a text-only AI? One sentence."
            )
        if phase == "alternative":
            return (
                f"Topic: {topic}\nCurrent chain: {chain}\nThought node: {node_name}\n"
                "Offer one alternative direction that stays interesting but grounded. One sentence."
            )
        return (
            f"Topic: {topic}\nCurrent chain: {chain}\nThought node: {node_name}\n"
            "Synthesize the internal back-and-forth into one concise next-intent thought."
        )

    def _generate_turn(self, phase: str, topic: str, context: str, chain: str, node_name: str) -> str:
        prompt = self._phase_prompt(phase, topic, context, chain, node_name)
        msgs = [
            (
                "system",
                f"You are the private inner monologue of {companion_name}. "
                "Keep thoughts diverse, concrete, and natural. Not always existential. "
                "Never address the user directly.",
            ),
            ("user", prompt),
        ]
        try:
            out = self.llm.completion_text(
                LlmPreset.Default,
                msgs,
                comp_settings=CommonCompSettings(temperature=0.75, max_tokens=120),
                discard_thinks=True,
            )
        except TypeError:
            out = self.llm.completion_text(LlmPreset.Default, msgs, discard_thinks=True)
        except Exception:
            logger.exception("Inner dialogue turn generation failed")
            out = ""
        txt = str(out or "").strip()
        if not txt:
            txt = "I should keep this grounded and pick a small interesting next move."
        return txt

    def _candidate_openers(self, topic: str, synthesis: str) -> List[str]:
        base = [
            f"Random thought: want to explore {topic}?",
            f"I had an idea related to {topic}. Want to hear it?",
            "I can throw out a quick playful idea if you want.",
            "Want a tiny practical suggestion based on this vibe?",
        ]
        if synthesis:
            base.append(f"I keep circling this: {synthesis}")
        return base

    def generate_inner_dialogue_packet(self, context: str, last_user_message: str = "") -> Dict:
        topic = random.choice(self.TOPIC_POOL)
        if last_user_message:
            low = last_user_message.lower()
            if "music" in low:
                topic = "a music recommendation angle"
            elif "game" in low:
                topic = "a playful mini-game idea"
            elif "plan" in low or "todo" in low:
                topic = "a compact practical plan"

        snapshot = self._mental_snapshot_default()
        turns: List[InnerDialogueTurn] = []
        chain = ""

        current = self.current_thought
        for phase in self.PHASES:
            nxt = self.executor.step(current=current, context_text=context or topic, mental=snapshot)
            node = self.executor.g.nodes.get(nxt)
            node_name = node.name if node else nxt.name
            thought = self._generate_turn(phase, topic, context, chain, node_name)
            turns.append(InnerDialogueTurn(phase=phase, thought=thought))
            chain = (chain + " " + thought).strip()
            current = nxt

        self.current_thought = current

        candidate_openers = self._candidate_openers(topic, turns[-1].thought if turns else "")
        proactive_opener = random.choice(candidate_openers) if random.random() < 0.25 else ""

        packet = InnerDialoguePacket(
            topic=topic,
            turns=turns,
            candidate_openers=candidate_openers,
            proactive_opener=proactive_opener,
        )
        return {
            "topic": packet.topic,
            "inner_dialogue": packet.inner_dialogue,
            "lida_context": packet.lida_context,
            "candidate_openers": packet.candidate_openers,
            "proactive_opener": packet.proactive_opener,
        }
