import json
from enum import StrEnum
from typing import List, Union

from pydantic import BaseModel, Field

from pm.character import sysprompt
from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

import json
from enum import StrEnum, auto
from typing import List, Union
from pydantic import BaseModel, Field


# 🍃 Cozy World Fact Categories
class FactCategory(StrEnum):
    UNSET = "Unset"
    PLACE = "Place"  # Locations in the world
    HOUSEHOLD = "Household & Decor"  # Furniture, decorations, appliances
    NATURE = "Plants & Nature"  # Trees, flowers, gardens
    ACTIVITY = "Activities & Events"  # Hobbies, festivals, weather


# 🏡 Cozy World Fact Subcategories
class FactSubcategory(StrEnum):
    # Place types
    CREATURE = "Creature"
    VILLAGE = "Village"
    HOME = "Home"
    SHOP = "Shop"
    OUTDOOR_AREA = "Outdoor Area"

    # Household items & decor
    FURNITURE = "Furniture"
    APPLIANCE = "Appliance"
    DECORATION = "Decoration"
    OUTDOOR_DECOR = "Outdoor Decor"

    # Nature elements
    FLOWER = "Flower"
    TREE = "Tree"
    SHRUB = "Shrub & Bush"
    GARDEN = "Crops & Gardens"

    # Activities & events
    HOBBY = "Hobby"
    FESTIVAL = "Festival"
    WEATHER = "Weather Event"

class CharacterState(BaseModel):
    """Keeps track of the AI companion's current state in the virtual world."""
    location: str = Field(default="Cozy Cottage", description="Where the character is currently located.")
    inventory: List[str] = Field(default_factory=list, description="Items the character is carrying.")
    mood: str = Field(default="Content", description="Current emotional state of the character.")
    last_action: str = Field(default="", description="Most recent action taken by the character.")
    last_observation: str = Field(default="", description="Most recent thing observed in the world.")

    def update_location(self, new_location: str):
        """Change the AI's current location."""
        self.location = new_location

    def add_to_inventory(self, item: str):
        """Adds an item to the AI's inventory."""
        self.inventory.append(item)

    def remove_from_inventory(self, item: str):
        """Removes an item from the AI's inventory, if present."""
        if item in self.inventory:
            self.inventory.remove(item)

    def update_mood(self, new_mood: str):
        """Updates the AI's emotional state."""
        self.mood = new_mood

    def update_last_action(self, action: str):
        """Stores the most recent action taken."""
        self.last_action = action

    def update_last_observation(self, observation: str):
        """Stores the most recent observation."""
        self.last_observation = observation

# 🌸 Cozy World Fact Model
class Fact(BaseModel):
    id: int = Field(default=-1)
    content: str = Field(default="", description="Information about the world, locations, or objects")
    subject: str = Field(default="", description="What this fact is about")

    category: FactCategory = Field(default=FactCategory.UNSET, description="General category, like Place or Household & Decor")
    subcategory: Union[FactSubcategory, None] = Field(default=None, description="More specific type (e.g., Flower, Furniture)")

    concepts: List[str] = Field(default_factory=list, description="Extra details, like color, material, or emotional feeling")
    overwrite_id: List[int] = Field(default_factory=list, description="If this fact updates another, it overwrites it")

    describes_world: bool = Field(default=False, description="does the fact describe the world?")
    describes_action: bool = Field(default=False, description="does the fact describe an action of the character?")


# 📦 Fact Storage System
class FactContainer(BaseModel):
    facts: List[Fact] = Field(default_factory=list, description="new world facts. please limit yourself to 2-4 most important facts of new information")

    def add_fact(self, fact: Fact):
        self.facts.append(fact)

    def get_facts_by_category(self, category: FactCategory):
        return [fact for fact in self.facts if fact.category == category]


import random

def initialize_cozy_world():
    """Creates a peaceful, cozy world setting in a meadow near a forest with a warm home and friendly critters."""

    world = FactContainer()

    # 🏡 Cozy Home
    world.add_fact(Fact(
        id=1,
        content="A small wooden cottage with a thatched roof, nestled in a peaceful meadow. The windows let in plenty of warm sunlight, and the front porch has a cozy swing.",
        subject="Cozy Cottage",
        category=FactCategory.PLACE,
        subcategory=FactSubcategory.HOME,
        concepts=["Warm and inviting", "Smells like fresh-baked bread", "Has a tiny chimney with a wisp of smoke"]
    ))

    # 🌿 Nature Around the Cottage
    world.add_fact(Fact(
        id=2,
        content="A vibrant meadow filled with soft grass and colorful wildflowers. Butterflies dance in the sunlight, and a gentle breeze carries the scent of lavender.",
        subject="Meadow",
        category=FactCategory.PLACE,
        subcategory=FactSubcategory.OUTDOOR_AREA,
        concepts=["Serene", "Full of wildflowers", "Perfect for afternoon naps"]
    ))

    world.add_fact(Fact(
        id=3,
        content="A small bubbling stream flows near the cottage, its water sparkling in the sunlight. Tiny fish dart beneath the surface, and smooth pebbles line the riverbed.",
        subject="Little Stream",
        category=FactCategory.PLACE,
        subcategory=FactSubcategory.OUTDOOR_AREA,
        concepts=["Crystal-clear water", "Gentle bubbling sound", "Cool to the touch"]
    ))

    world.add_fact(Fact(
        id=4,
        content="A dense forest stands nearby, filled with towering trees that provide shade and shelter to birds, rabbits, and other small creatures.",
        subject="Whispering Woods",
        category=FactCategory.PLACE,
        subcategory=FactSubcategory.OUTDOOR_AREA,
        concepts=["Tall trees", "Dappled sunlight", "Home to playful critters"]
    ))

    # 🌸 Flowers & Plants
    flowers = ["Sunflowers", "Bluebells", "Daisies", "Lavender", "Wild roses"]
    for i, flower in enumerate(flowers, start=5):
        world.add_fact(Fact(
            id=i,
            content=f"A patch of {flower.lower()} grows near the cottage, adding bright colors to the landscape.",
            subject=flower,
            category=FactCategory.NATURE,
            subcategory=FactSubcategory.FLOWER,
            concepts=["Vibrant", "Gently sways in the breeze"]
        ))

    # 🪑 Cozy Furniture Inside the Home
    world.add_fact(Fact(
        id=10,
        content="A plush armchair sits by the fireplace, perfect for curling up with a good book and a warm cup of tea.",
        subject="Armchair",
        category=FactCategory.HOUSEHOLD,
        subcategory=FactSubcategory.FURNITURE,
        concepts=["Soft cushions", "Smells like vanilla", "Worn but well-loved"]
    ))

    world.add_fact(Fact(
        id=11,
        content="A sturdy wooden table in the kitchen, covered with a checkered tablecloth and a bowl of fresh fruit.",
        subject="Kitchen Table",
        category=FactCategory.HOUSEHOLD,
        subcategory=FactSubcategory.FURNITURE,
        concepts=["Made of oak", "Always has fresh fruit", "Perfect for morning tea"]
    ))

    world.add_fact(Fact(
        id=12,
        content="A small bookshelf filled with well-loved books, from adventure stories to cozy mysteries.",
        subject="Bookshelf",
        category=FactCategory.HOUSEHOLD,
        subcategory=FactSubcategory.FURNITURE,
        concepts=["Smells like old parchment", "Creaks slightly when touched"]
    ))

    # 🐦 Friendly Critters Outside
    critters = [
        ("Sparrow", "A tiny sparrow flits between tree branches, chirping a cheerful tune."),
        ("Rabbit", "A fluffy white rabbit nibbles on some clover near the edge of the meadow."),
        ("Butterfly", "A bright blue butterfly flutters lazily through the flowers, enjoying the sun."),
        ("Squirrel", "A mischievous squirrel scampers up a tree, looking for acorns."),
    ]

    for i, (name, description) in enumerate(critters, start=20):
        world.add_fact(Fact(
            id=i,
            content=description,
            subject=name,
            category=FactCategory.NATURE,
            subcategory=FactSubcategory.CREATURE,
            concepts=["Harmless", "Lively", "Part of the peaceful ecosystem"]
        ))

    # ☀️ Weather & Atmosphere
    world.add_fact(Fact(
        id=30,
        content="The sky is a soft pastel blue with a few wispy clouds floating by. The air smells fresh and clean.",
        subject="Sky",
        category=FactCategory.ACTIVITY,
        subcategory=FactSubcategory.WEATHER,
        concepts=["Peaceful", "Bright but gentle sunlight"]
    ))

    world.add_fact(Fact(
        id=31,
        content="A gentle breeze rustles the leaves, making them whisper softly.",
        subject="Breeze",
        category=FactCategory.ACTIVITY,
        subcategory=FactSubcategory.WEATHER,
        concepts=["Calming", "Feels like a hug from the wind"]
    ))

    return world


# 🎨 Markdown Formatter for System Prompt
class WorldFormatter:
    @staticmethod
    def format_facts(fact_container: FactContainer) -> str:
        """Turns stored facts into a structured Markdown world description."""
        sections = []

        def add_section(title, facts):
            if facts:
                sections.append(f"## {title}")
                for fact in facts:
                    subcat = f" ({fact.subcategory})" if fact.subcategory else ""
                    concept_tags = f" | *Details*: {', '.join(fact.concepts)}" if fact.concepts else ""
                    sections.append(f"- **{fact.subject}{subcat}**: {fact.content}{concept_tags}")

        # Organize by category
        add_section("Places", fact_container.get_facts_by_category(FactCategory.PLACE))
        add_section("Household & Decor", fact_container.get_facts_by_category(FactCategory.HOUSEHOLD))
        add_section("Plants & Nature", fact_container.get_facts_by_category(FactCategory.NATURE))
        add_section("Activities & Events", fact_container.get_facts_by_category(FactCategory.ACTIVITY))

        return "\n\n".join(sections) if sections else "No world data yet! Let’s add some details. ☕"

class ActionType(StrEnum):
    Interact = "Interact"
    Observe = "Observe"

class Action(BaseModel):
    action_type: ActionType = Field(description="do something or observer something")
    target_instruction: str = Field(description="describe the action. what you observe or what you do")

class ResponseAction(BaseModel):
    action_valid: bool = Field(description="is the action valid from a gm perspective?")
    action_result: Union[None, str] = Field(default="", description="the result of action on the world")

class ResponseObservation(BaseModel):
    observation_result: Union[None, str] = Field(default="", description="the result of the observation")

sysprompt_gm = "You are a game master for a fantasy text RPG like Animal Crossing. You goal is to provide a virtual world as enrichtment activity for an AI companion.. \n## World Information:\n"
sysprompt_player = sysprompt + "\nYou are in virtual world mode right now. \n## World Information:\n"
sysprompt_fe = """You extract facts from a text RPG for enhanced memory storage.
You will be given existing facts, use the ID of those to check if some need extension or must be overridden with new information!
## Existing facts: 
"""

fact_id_cnt = 1
existing_facts = initialize_cozy_world()
character_state = CharacterState()

prompt_gm = [("system", sysprompt_gm)]
prompt_player = [("system", sysprompt_player)]

player_turn = True
current_observation = True
while True:

    world_block = WorldFormatter.format_facts(fact_container=existing_facts)
    prompt_gm[0] = ("system", sysprompt_gm + world_block)
    prompt_player[0] = ("system", sysprompt_player) # + world_block)

    if player_turn:
        print(f"### BEGIN PLAYER")
        _, calls = controller.completion_tool(LlmPreset.Default, prompt_player, CommonCompSettings(max_tokens=512), tools=[Action])
        content = json.dumps(calls[0].model_dump(), indent=2)
        action: Action = calls[0]
        if action.action_type == ActionType.Observe:
            current_observation = True
        else:
            current_observation = False

        # Update character state
        character_state.update_last_action(action.target_instruction)

        prompt_gm.append(("user", content))
        prompt_player.append(("assistant", content))
    else:
        print(f"### BEGIN GM")
        _, calls = controller.completion_tool(LlmPreset.Default, prompt_gm, CommonCompSettings(max_tokens=512), tools=[ResponseObservation if current_observation else ResponseAction])
        content = json.dumps(calls[0].model_dump(), indent=2)

        # Update character state with observation result
        if current_observation:
            character_state.update_last_observation(calls[0].observation_result)

        prompt_gm.append(("assistant", content))
        prompt_player.append(("user", content))

    print(content)
    sysprompt_fe_with_facts = sysprompt_fe + json.dumps(existing_facts.model_dump(), indent=2)
    prompt_fe = [("system", sysprompt_fe_with_facts),
                 ("user", content + "\nPlease extract new facts!")]
    _, calls = controller.completion_tool(LlmPreset.Default, prompt_player, CommonCompSettings(max_tokens=512), tools=[FactContainer])
    for fact in calls[0].facts:
        fact.id = fact_id_cnt
        fact_id_cnt += 1

        if fact.describes_world and not fact.describes_action:
            existing_facts.facts.append(fact)

    #print(json.dumps(existing_facts.model_dump(), indent=2))
    player_turn = not player_turn
