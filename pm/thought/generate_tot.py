import json
import string
import textwrap
import uuid
from copy import copy
from enum import StrEnum
from pprint import pprint
import random
from typing import Dict, List

from pydantic import BaseModel, Field
import networkx as nx
import json

from pm.character import companion_name
from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

class Valence(StrEnum):
    Positive = "positive"
    Negative = "negative"


class GenerativeThought(BaseModel):
    observation: str = Field()
    context_ai_companion: str = Field()
    analysis: str = Field()
    action: str = Field()

class GenerativeThoughts(BaseModel):
    thoughts: List[GenerativeThought] = Field(default_factory=list)

    def execute(self, state: Dict):
        return True

class DiscrimminatoryThought(BaseModel):
    reflective_thought_observation: str = Field()
    context_world: str = Field()
    possible_complication: str = Field()
    worst_case_scenario: str = Field()

class DiscrimminatoryThoughts(BaseModel):
    thoughts: List[DiscrimminatoryThought] = Field(default_factory=list)

    def execute(self, state: Dict):
        return True


def to_internal_thought(thought):
    return thought

    #messages = [
    #    (
    #    "system", "You are a helpful assistant that converts formalized thought chains into first-person thought for chain-of-thought prompting. The goal is to simulate a human mind with a
    #    LLM-based "
    #              "cognitive architecture."
    #              "It must be like internal monologue and not be directed at the user!"),
    #    ("user", f"{thought}"
    #             f"Can you convert this into a first-person thought for CoT prompting? Do not address the user directly!")
    #]
    #content = controller.completion_text(LlmPreset.ConvertToInternalThought, messages, comp_settings=CommonCompSettings(temperature=0.1, max_tokens=512))
    #messages.append(("assistant", content))
    #messages.append(("user", "Do not address the user directly! Don't forget, you are essentially {companion_name} in our scenario. Also don't make a list, make it sound like a natural inner
    # thought! "
    #                         "Let's try again without addressing {user_name} and without lists. And make it shorter, 1-2 sentences for the thought!"))
    #content = controller.completion_text(LlmPreset.ConvertToInternalThought, messages, comp_settings=CommonCompSettings(temperature=0.1, max_tokens=512))
    #return content


class ThoughtRating(BaseModel):
    reason: str = Field(description="Analyze and explain your reasoning in 1-2 sentences before rating.")
    realism: float = Field(description="Is the response believable and aligned with human-like reasoning? (between 0 and 1)")
    novelty: float = Field(description="Does it avoid repetitive patterns or overly generic answers? (between 0 and 1)")
    relevance: float = Field(description="Does it align with user context and ego? (between 0 and 1)")
    emotion: float = Field(description="Does it express appropriate emotions (e.g., anger for insults)? (between 0 and 1)")
    effectiveness: float = Field(description="Does it meet conversational goals (e.g., re-engaging the user)? (between 0 and 1)")
    possibility: float = Field(description="Is it possible for the AI companion to do this? (between 0 and 1)")
    positives: str = Field(description="What is positive about this thought? 1-2 sentences.")
    negatives: str = Field(description="What is negative about this thought? 1-2 sentences.")

def generate_thought_chain(ctx, k, chain, generative, depth, parent_node, max_depth=4, rating_threshold=0.75):
    if depth == 0:
        messages = [
            ("system", f"You are the thought-generating agent in a cognitive architecture. Your goal is to generate at least 5 different possible thoughts or follow-up thoughts in a sequential "
                       f"thought-chain."),
            ("user", f"This is the context: {ctx}"
                     f"And the current thought chain: {chain}"
                     f"Please generate at least 5 different thoughts {companion_name} could have to mimic human cognition. They will be rated and the best"
                     f"one executed.")
        ]
    else:
        messages = [
            ("system", f"You are the thought-generating agent in a cognitive architecture. Your goal is to generate at least 5 different follow-up thoughts that are an extension of the latest "
                       f"thoughts. This is the knowledge: {k}"),
            ("user",
                     f"This is the current thought-chain: {chain}"
                     f"Please generate at least 5 different follow-up thoughts that reflect on the elements the chain!")
        ]

    if generative:
        tools = [GenerativeThoughts]
    else:
        tools = [DiscrimminatoryThoughts]

    content, calls = controller.completion_tool(LlmPreset.GenerateTot, messages, comp_settings=CommonCompSettings(temperature=0.8, repeat_penalty=1.1, max_tokens=2048), tools=tools)
    state = {}
    for call in calls:
        thoughts = copy(call.thoughts)
        thoughts.reverse()
        for thought in thoughts:
            new_thought = f"""
### BEGIN CHAIN ELEMENT
{thought.model_dump_json()}
### END CHAIN ELEMENT"""

            messages = [
                ("system", f"You are a helpful assistant that rates internal thoughts of an AI companion system. The AI companion can only communicate with text, can't self-modify itself. But"
                           f"she is able to feel emotions and display all other aspects of human emotion and cognition. Please rate the thought-chain. Your output will determine if it the thought "
                           f"is fit for further prompting."),
                ("user", new_thought)
            ]

            rating = 0
            possibility = 0
            content, calls = controller.completion_tool(LlmPreset.GenerateTot, messages, comp_settings=CommonCompSettings(temperature=0.6, repeat_penalty=1, max_tokens=2048), tools=[ThoughtRating])
            for call in calls:
                if isinstance(call, ThoughtRating):
                    rating = (call.relevance + call.emotion + call.novelty + call.realism + call.possibility + call.effectiveness) / 6
                    possibility = call.possibility

            do_break = False
            if possibility <= 0.75:
                child_node = parent_node
                child_node = add_thought_to_tree("Backtrack", "No, I can't do that...", child_node)
            else:
                lines = []
                if isinstance(thought, GenerativeThought):
                    child_node = parent_node
                    lines.append(f"Observation: {to_internal_thought(thought.observation)}")
                    lines.append(f"Context: {to_internal_thought(thought.context_ai_companion)}")
                    lines.append(f"Analysis: {to_internal_thought(thought.analysis)}")
                    lines.append(f"Idea: {to_internal_thought(thought.action)}")

                    child_node = add_thought_to_tree("Generative Thought", "\n".join(lines), child_node)
                elif isinstance(thought, DiscrimminatoryThought):
                    child_node = parent_node
                    lines.append(f"Reflection: {to_internal_thought(thought.reflective_thought_observation)}")
                    lines.append(f"Context: {to_internal_thought(thought.context_world)}")
                    lines.append(f"Complication: {to_internal_thought(thought.possible_complication)}")
                    lines.append(f"Outcome: {to_internal_thought(thought.worst_case_scenario)}")

                    child_node = add_thought_to_tree("Checking Thought", "\n".join(lines), child_node)
                else:
                    raise Exception()

                if rating < rating_threshold:
                    child_node = add_thought_to_tree("Backtrack", call.negatives, child_node)
                    do_break = True

            if depth < max_depth and not do_break:
                chain = chain + new_thought
                limite_reached = generate_thought_chain(ctx, k, chain, not generative, depth + 1, child_node, max_depth, rating_threshold)
                if limite_reached:
                    return True
                break
            elif depth >= max_depth:
                return True
    return False


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

def add_thought_to_tree(node_type, content, parent_node):
    """Add a new thought to the tree."""
    new_id = str(str(uuid.uuid4()))
    content = content #"\n".join(textwrap.wrap(content, width=40))
    node_id = f"{new_id}\n{node_type}\n{content}"
    new_node = ThoughtNode(node_id, node_type, content, parent_node)
    parent_node.add_child(new_node)
    return new_node

def export_to_networkx(node):
    """Export the tree to a NetworkX graph."""
    graph = nx.DiGraph()

    def add_edges(parent):
        for child in parent.children:
            graph.add_edge(parent.node_id, child.node_id, type=child.node_type)
            graph.nodes[child.node_id]["content"] = child.content
            graph.nodes[child.node_id]["type"] = child.node_type
            add_edges(child)

    add_edges(node)
    return graph


def flatten_thoughts(root):
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


def generate_tot_v1(context: str, knowledge: str, max_depth: int):
    if controller.test_mode:
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(int(max_depth * 50)))

    thought_tree = ThoughtNode("root", "root", "")
    generate_thought_chain(context, knowledge, "", True, 0, thought_tree, max_depth=max_depth)
    flattened_nodes = flatten_thoughts(thought_tree)

    parts = []
    for node in flattened_nodes:
        parts.append(node.content)

    # Export to NetworkX
    graph = export_to_networkx(thought_tree)
    nx.write_gml(graph, f"./logs/thought_tree_{uuid.uuid4().hex}.gml")

    return "\n".join(parts)



