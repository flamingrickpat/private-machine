
import time
from datetime import timedelta
from textwrap import indent
from typing import List, Type

from pydantic import BaseModel
from sqlalchemy import text, func
from sqlmodel import select, col

from pm.character import cluster_split, companion_name, user_name
from pm.cluster.cluster_temporal import MiniContextCluster, temporally_cluster_context, TemporalCluster, temp_cluster_to_level
from pm.cluster.cluster_utils import get_optimal_clusters
from pm.common_prompts.extract_facts import extract_facts
from pm.common_prompts.summary_perspective import summarize_messages
from pm.controller import controller
from pm.cosine_clustering.cosine_cluster import get_best_category, cluster_text, TEMPLATE_CATEGORIES
from pm.database.db_utils import update_database_item
from pm.database.tables import Event, EventCluster, Cluster, ClusterType, Fact, CauseEffectDbEntry, FactCategory
from pm.embedding.token import get_token
from pm.fact_learners.extract_cause_effect_agent import extract_cas
from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.utils.pydantic_utils import create_basemodel

memory_schema = [
    {
        f"id": 1,
        f"path": f"belong_people_{companion_name.lower()}_personality",
        f"description": f"Contains facts and observations about {companion_name}'s core personality traits, such as her empathy, curiosity, playfulness, and reflective nature. This entry helps capture how {companion_name}’s character is expressed and evolves during interactions."
    },
    {
        f"id": 2,
        f"path": f"belong_people_{companion_name.lower()}_needs",
        f"description": f"Records information about {companion_name}’s intrinsic needs—such as her need for downtime, connection through interaction, and other cognitive or emotional necessities—to maintain a balanced state."
    },
    {
        f"id": 3,
        f"path": f"belong_people_{companion_name.lower()}_goals",
        f"description": f"Contains facts regarding {companion_name}’s objectives and aspirations, including her mission to support {user_name} and align with her internal values and long-term purpose as derived from her simulated thought processes."
    },
    {
        f"id": 4,
        f"path": f"belong_people_{companion_name.lower()}_preferences",
        f"description": f"Documents {companion_name}’s likes, dislikes, and decision-making biases, capturing her preferences in conversation, topic choices, and interaction styles that reflect her unique AI character."
    },
    {
        f"id": 5,
        f"path": f"belong_people_{companion_name.lower()}_interactions",
        f"description": f"Holds details about {companion_name}’s interactions with others—especially her communication with {user_name}—including tone, engagement patterns, and the dynamics of their exchanges."
    },
    {
        f"id": 6,
        f"path": f"belong_people_{user_name.lower()}_personality",
        f"description": f"Contains facts about {user_name}’s personality traits and characteristics, as observed through his communication style, behavior, and emotional responses during his interactions with {companion_name}."
    },
    {
        f"id": 7,
        f"path": f"belong_people_{user_name.lower()}_needs",
        f"description": f"Records information regarding {user_name}’s intrinsic needs, including his need for emotional support, clarity in communication, and other personal requirements as perceived during interactions."
    },
    {
        f"id": 8,
        f"path": f"belong_people_{user_name.lower()}_goals",
        f"description": f"Documents {user_name}’s objectives and aspirations, reflecting his short-term desires and long-term ambitions that emerge from his dialogue with {companion_name}."
    },
    {
        f"id": 9,
        f"path": f"belong_people_{user_name.lower()}_preferences",
        f"description": f"Captures {user_name}’s preferences, including his interests, topics of conversation, and decision-making biases that influence his interactions with {companion_name}."
    },
    {
        f"id": 10,
        f"path": f"belong_people_{user_name.lower()}_interactions",
        f"description": f"Holds details about {user_name}’s interactions with {companion_name}, including his communication style, feedback, and the evolving dynamics of their relationship."
    },
    {
        f"id": 11,
        f"path": f"belong_world_places",
        f"description": f"Contains information about specific locations and physical settings mentioned in the narrative, providing context for where interactions take place."
    },
    {
        f"id": 12,
        f"path": f"belong_world_environment",
        f"description": f"Records broader environmental details such as atmosphere, weather, and contextual surroundings that influence the narrative’s mood and situational context."
    },
    {
        f"id": 13,
        f"path": f"belong_world_world_building",
        f"description": f"Documents facts related to the constructed world, including societal structures, cultural elements, and background lore that enrich the narrative's setting."
    },
    {
        f"id": 14,
        f"path": f"belong_relationships_{companion_name.lower()}_{user_name.lower()}_good",
        f"description": f"Captures positive aspects of the relationship between {companion_name} and {user_name}, including instances of cooperation, mutual understanding, and supportive interactions."
    },
    {
        f"id": 15,
        f"path": f"belong_relationships_{companion_name.lower()}_{user_name.lower()}_bad",
        f"description": f"Records negative aspects or conflicts in the relationship between {companion_name} and {user_name}, highlighting moments of tension, misunderstanding, or disagreement."
    }
]

path_to_id = {item["path"]: item["id"] for item in memory_schema}

def init_basemodel() -> Type[BaseModel]:
    model_definitions = []
    for node in memory_schema:
        model_definitions.append({
            f"name": node["path"],
            f"type": float,
            f"description": node["description"]
        })

    DynamicModel = create_basemodel(model_definitions, randomnize_entries=True)
    DynamicModel.__doc__ = f"Use this JSON schema to rate the belonging of facts to different categories. Use 0 for no belonging and 1 for maximum belonging."
    return DynamicModel


sysprompt = f"""You are a helpful assistant that helps organize facts from a story about {companion_name}, an AI and her User, {user_name}, into categories. These categories will help the author not forget stuff.
You do this by creating a JSON where you rate the belongingness to different categories with a float between 0 and 1.    
"""

def categorize_fact(fact: Fact) -> List[FactCategory]:
    res = []
    bm = init_basemodel()
    messages = [
        ("system", sysprompt),
        ("user", f"Can you put this fact into the different categories? ### BEGIN FACT\n{fact.content}\n### END FACT")
    ]

    _, calls = controller.completion_tool(LlmPreset.Conscious, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.1), tools=[bm])

    mb = calls[0]
    max_weight = 0
    for f in mb.model_fields_set:
        max_weight = max(max_weight, getattr(mb, f))

    for f in mb.model_fields_set:
        weight = getattr(mb, f)
        cat_id = path_to_id.get(f, 0)
        fc = FactCategory(
            fact_id=fact.id,
            category=f,
            category_id=cat_id,
            weight=weight,
            max_weight=max_weight)
        update_database_item(fc)
        res.append(fc)
    return res