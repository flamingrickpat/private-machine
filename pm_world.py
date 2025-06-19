# virtual_world_manager.py

import json
import math
import random
import threading
import queue
import time
import os
import sqlite3
from enum import StrEnum
from typing import List, Optional, Tuple, Dict, Any, Set
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

# Assuming pm_lida.py is in the same project and we can import from it.
# In a real project, these would be in a shared 'utils' or 'llm' module.
from pm_lida import main_llm, LlmPreset, CommonCompSettings, llama_worker


# --- 1. Core Data Models for the Virtual World (Extended) ---

class Vector3D(BaseModel):
    """Represents a point or vector in 3D space."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class RelationshipType(StrEnum):
    """Defines the type of spatial or logical connection between two objects."""
    ON_TOP_OF = "on top of"
    INSIDE = "inside"
    ATTACHED_TO = "attached to"
    NEXT_TO = "next to"
    UNDER = "under"


class Relationship(BaseModel):
    """Describes a directional link from this object to another."""
    target_id: int = Field(description="The unique ID of the object this one is related to.")
    type: RelationshipType = Field(description="The nature of the relationship.")


class WorldObject(BaseModel):
    """Represents any single entity in the virtual world."""
    id: int
    name: str = Field(description="The common name of the object (e.g., 'Oak Tree', 'Wooden Desk').")
    description: str = Field(description="A brief descriptive text about the object.")
    owner_id: Optional[int] = Field(None, description="The ID of the character who owns this object.")

    position: Vector3D = Field(default_factory=Vector3D, description="The object's center point in world coordinates.")
    size: Vector3D = Field(default_factory=lambda: Vector3D(x=1.0, y=1.0, z=1.0),
                           description="The object's bounding box dimensions.")

    relationships: List[Relationship] = Field(default_factory=list,
                                              description="List of relationships to other objects.")
    properties: Dict[str, Any] = Field(default_factory=dict,
                                       description="Arbitrary properties like 'color', 'is_openable', 'current_state'.")


class Character(BaseModel):
    """Represents an AI's state within the virtual world."""
    id: int
    name: str
    position: Vector3D = Field(default_factory=Vector3D)
    inventory: List[WorldObject] = Field(default_factory=list)
    conversation_partners: Set[int] = Field(default_factory=set,
                                            description="IDs of characters this one is talking to.")


# --- 2. Procedural World Generation (Heavily Extended) ---

class WorldGenerator:
    """Creates the initial state of the virtual world with configurable parameters."""

    def __init__(self, seed: int = 1337):
        self.rng = random.Random(seed)
        self.current_id = 1

    def _get_next_id(self) -> int:
        id_val = self.current_id
        self.current_id += 1
        return id_val

    def generate_world(self, characters_to_create: List[str], world_size_x: int = 100, world_size_y: int = 100,
                       num_houses: int = 3) -> Tuple[List[WorldObject], List[Character]]:
        """
        Generates a small village scene with multiple houses and characters.
        Each character is assigned their own house.
        """
        objects: List[WorldObject] = []
        characters: List[Character] = []
        character_map: Dict[str, Character] = {}

        # Ground plane
        ground_size = Vector3D(x=float(world_size_x), y=float(world_size_y), z=0.1)
        objects.append(WorldObject(
            id=self._get_next_id(), name="Grassy Ground",
            description="A vast, flat expanse of soft green grass, forming the foundation of this small settlement.",
            position=Vector3D(x=0, y=0, z=-0.1), size=ground_size
        ))

        # Create characters first, so we can assign ownership
        character_id_counter = 0
        for char_name in characters_to_create:
            char = Character(id=character_id_counter, name=char_name)
            characters.append(char)
            character_map[char_name] = char
            character_id_counter += 1

        if num_houses < len(characters):
            print(
                f"Warning: Not enough houses ({num_houses}) for all characters ({len(characters)}). Some will be homeless.")

        # Generate houses and assign them to characters
        house_positions = []
        for i in range(num_houses):
            while True:
                house_pos = Vector3D(x=self.rng.uniform(-world_size_x / 4, world_size_x / 4),
                                     y=self.rng.uniform(-world_size_y / 4, world_size_y / 4), z=0)
                # Ensure houses are not too close
                if all(math.dist((house_pos.x, house_pos.y), (p.x, p.y)) > 20.0 for p in house_positions):
                    house_positions.append(house_pos)
                    break

            owner_char = characters[i] if i < len(characters) else None
            objects.extend(self._generate_house(house_pos, owner_char))

        # Place characters in their houses
        for char in characters:
            owned_house = next(
                (obj for obj in objects if obj.owner_id == char.id and obj.properties.get("type") == "building"), None)
            if owned_house:
                char.position = owned_house.position

        # Generate a forest area
        forest_center_x = world_size_x * 0.35
        forest_center_y = 0
        for _ in range(int(world_size_x * world_size_y / 100)):  # Scale tree count with world size
            tree_pos = Vector3D(
                x=self.rng.uniform(forest_center_x - world_size_x / 5, forest_center_x + world_size_x / 5),
                y=self.rng.uniform(forest_center_y - world_size_y / 2, forest_center_y + world_size_y / 2), z=0)
            if math.dist((tree_pos.x, tree_pos.y), (0, 0)) > 25:  # Don't put trees in the village center
                objects.append(WorldObject(
                    id=self._get_next_id(), name="Pine Tree",
                    description="A tall, coniferous tree, smelling faintly of sap.",
                    position=tree_pos, size=Vector3D(x=1.5, y=1.5, z=self.rng.uniform(10, 20))
                ))

        return objects, characters

    def _generate_house(self, position: Vector3D, owner: Optional[Character]) -> List[WorldObject]:
        house_objects = []
        owner_id = owner.id if owner else None
        owner_name = owner.name if owner else "an unknown person"
        is_primary_character_house = owner_id == 0  # Specific check to place quest items

        # House structure
        house_id = self._get_next_id()
        house = WorldObject(
            id=house_id, name=f"{owner_name.capitalize()}'s Cottage" if owner else "Abandoned Cottage",
            description=f"A cottage made of wood and stone belonging to {owner_name}. It looks cozy.",
            position=position, size=Vector3D(x=10, y=8, z=5), properties={"type": "building"},
            owner_id=owner_id
        )
        house_objects.append(house)

        # Generate rooms inside the house
        house_objects.extend(self._generate_living_room(position, house_id, owner_id))
        house_objects.extend(
            self._generate_kitchen(Vector3D(x=position.x + 3, y=position.y - 2, z=0), house_id, owner_id,
                                   add_special_items=is_primary_character_house))
        house_objects.extend(
            self._generate_bedroom(Vector3D(x=position.x - 3, y=position.y + 2, z=0), house_id, owner_id))

        return house_objects

    def _generate_living_room(self, house_pos: Vector3D, house_id: int, owner_id: Optional[int]) -> List[WorldObject]:
        items = []
        # Fireplace
        fireplace_pos = Vector3D(x=house_pos.x, y=house_pos.y + 3.5, z=0)
        fireplace_id = self._get_next_id()
        items.append(WorldObject(
            id=fireplace_id, name="Stone Fireplace",
            description="A large, soot-stained stone fireplace. It's currently unlit.",
            position=fireplace_pos, size=Vector3D(x=2, y=0.5, z=3), owner_id=owner_id,
            relationships=[Relationship(target_id=house_id, type=RelationshipType.INSIDE)],
            properties={"is_lit": False}
        ))
        # Comfy Chair
        chair_pos = Vector3D(x=fireplace_pos.x + 1.5, y=fireplace_pos.y - 1.5, z=0)
        items.append(WorldObject(
            id=self._get_next_id(), name="Comfy Chair",
            description="A plush, well-worn armchair positioned near the fireplace.",
            position=chair_pos, size=Vector3D(x=1, y=1, z=1.2), owner_id=owner_id,
            relationships=[Relationship(target_id=house_id, type=RelationshipType.INSIDE)]
        ))
        return items

    def _generate_kitchen(self, base_pos: Vector3D, house_id: int, owner_id: Optional[int], add_special_items: bool) -> \
    List[WorldObject]:
        items = []
        # A table
        table_pos = base_pos
        table_id = self._get_next_id()
        table = WorldObject(
            id=table_id, name="Wooden Table", description="A simple but sturdy wooden dining table.",
            position=table_pos, size=Vector3D(x=2, y=1, z=0.8), owner_id=owner_id,
            relationships=[Relationship(target_id=house_id, type=RelationshipType.INSIDE)]
        )
        items.append(table)

        # A book on the table
        book_pos = Vector3D(x=table_pos.x, y=table_pos.y, z=table.size.z)
        items.append(WorldObject(
            id=self._get_next_id(), name="Old Book", description="A leather-bound book with a worn cover.",
            position=book_pos, size=Vector3D(x=0.3, y=0.2, z=0.05), owner_id=owner_id,
            relationships=[
                Relationship(target_id=table_id, type=RelationshipType.ON_TOP_OF),
                Relationship(target_id=house_id, type=RelationshipType.INSIDE)
            ]
        ))

        # A key also on the table, but only if specified
        if add_special_items:
            key_pos = Vector3D(x=table_pos.x + 0.5, y=table_pos.y, z=table.size.z)
            items.append(WorldObject(
                id=self._get_next_id(), name="Brass Key", description="A small, ornate brass key.",
                position=key_pos, size=Vector3D(x=0.1, y=0.02, z=0.01), owner_id=owner_id,
                relationships=[
                    Relationship(target_id=table_id, type=RelationshipType.ON_TOP_OF),
                    Relationship(target_id=house_id, type=RelationshipType.INSIDE)
                ]
            ))
        return items

    def _generate_bedroom(self, base_pos: Vector3D, house_id: int, owner_id: Optional[int]) -> List[WorldObject]:
        items = []
        # Bed
        bed_id = self._get_next_id()
        items.append(WorldObject(
            id=bed_id, name="Wooden Bed", description="A simple wooden bed with a straw mattress and a wool blanket.",
            position=base_pos, size=Vector3D(x=1, y=2, z=0.6), owner_id=owner_id,
            relationships=[Relationship(target_id=house_id, type=RelationshipType.INSIDE)]
        ))
        # Chest at the foot of the bed
        chest_pos = Vector3D(x=base_pos.x, y=base_pos.y - 1.5, z=0)
        chest_id = self._get_next_id()
        items.append(WorldObject(
            id=chest_id, name="Wooden Chest",
            description="A sturdy wooden chest, bound with iron straps. It is closed.",
            position=chest_pos, size=Vector3D(x=0.8, y=0.4, z=0.5), owner_id=owner_id,
            relationships=[Relationship(target_id=house_id, type=RelationshipType.INSIDE)],
            properties={"is_locked": True, "is_open": False}
        ))
        return items


# --- 3. The Game Master / World Simulation Manager (Heavily Updated) ---

class VirtualWorldManager:
    """
    Manages the state of the virtual world, its objects, and multiple characters.
    Acts as the Game Master. Now with SQLite persistence for the entire world state.
    """

    def __init__(self, db_path: str, character_names: List[str]):
        self.db_path = db_path
        self.turn_count = 0  # This will represent the current "round" number
        self.action_history: List[Tuple[int, int, str, str]] = []  # turn, char_id, instruction, result
        self.world_objects: Dict[int, WorldObject] = {}
        self.characters: Dict[int, Character] = {}
        self.next_character_id = 0
        self.next_object_id = 1

        if os.path.exists(self.db_path):
            print(f"VWM: Loading world from '{self.db_path}'...")
            self.load_world()
            # Ensure all requested characters exist, add if necessary
            existing_names = {c.name for c in self.characters.values()}
            for name in character_names:
                if name not in existing_names:
                    print(f"VWM: Adding new character '{name}' to existing world.")
                    self.add_character(name, Vector3D(x=0, y=0, z=0))  # Add at origin, they can be moved later
        else:
            print("VWM: No database found. Generating new world...")
            generator = WorldGenerator()
            # Start turns at 1 for a more natural count
            self.turn_count = 1
            all_objects, all_characters = generator.generate_world(character_names)
            self.world_objects = {obj.id: obj for obj in all_objects}
            self.characters = {char.id: char for char in all_characters}
            self.next_character_id = len(all_characters)
            self.next_object_id = max(obj.id for obj in all_objects) + 1 if all_objects else 1
            self.save_world()  # Initial save of the new world

        # For easy lookup of object relationships
        self._build_relationship_map()
        print(
            f"VWM: World initialized with {len(self.characters)} characters and {len(self.world_objects)} objects. Turn: {self.turn_count}")

    def _init_db(self, conn: sqlite3.Connection):
        """Creates the necessary database tables if they don't exist."""
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        # Characters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                position_json TEXT,
                conversation_partners_json TEXT
            );
        """)

        # Main table for all objects, with ownership and inventory status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS world_objects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner_id INTEGER,
                position_json TEXT,
                size_json TEXT,
                relationships_json TEXT,
                properties_json TEXT,
                inventory_of_character_id INTEGER,
                FOREIGN KEY (owner_id) REFERENCES characters(id),
                FOREIGN KEY (inventory_of_character_id) REFERENCES characters(id)
            );
        """)

        # Action history - composite key includes instruction to avoid simple turn/char clashes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_history (
                turn INTEGER,
                character_id INTEGER,
                instruction TEXT,
                result TEXT,
                PRIMARY KEY (turn, character_id, instruction),
                FOREIGN KEY (character_id) REFERENCES characters(id)
            );
        """)
        conn.commit()

    def save_world(self):
        """Saves the entire current world state to the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            self._init_db(conn)
            cursor = conn.cursor()
            cursor.execute('BEGIN')
            try:
                # Clear existing data for a full rewrite
                cursor.execute('DELETE FROM action_history')
                cursor.execute('DELETE FROM world_objects')
                cursor.execute('DELETE FROM characters')
                cursor.execute('DELETE FROM metadata')

                # Save characters
                char_data = [
                    (c.id, c.name, c.position.model_dump_json(), json.dumps(list(c.conversation_partners)))
                    for c in self.characters.values()
                ]
                cursor.executemany(
                    "INSERT INTO characters (id, name, position_json, conversation_partners_json) VALUES (?, ?, ?, ?)",
                    char_data)

                # Save all objects (world + all inventories)
                world_obj_data = [
                    (obj.id, obj.name, obj.description, obj.owner_id,
                     obj.position.model_dump_json(), obj.size.model_dump_json(),
                     json.dumps([r.model_dump() for r in obj.relationships]),
                     json.dumps(obj.properties), None)
                    for obj in self.world_objects.values()
                ]
                inv_obj_data = []
                for char in self.characters.values():
                    for item in char.inventory:
                        inv_obj_data.append(
                            (item.id, item.name, item.description, item.owner_id,
                             item.position.model_dump_json(), item.size.model_dump_json(),
                             json.dumps([r.model_dump() for r in item.relationships]),
                             json.dumps(item.properties), char.id)
                        )

                cursor.executemany("""
                    INSERT INTO world_objects (id, name, description, owner_id, position_json, size_json, relationships_json, properties_json, inventory_of_character_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, world_obj_data + inv_obj_data)

                # Save action history
                cursor.executemany(
                    "INSERT INTO action_history (turn, character_id, instruction, result) VALUES (?, ?, ?, ?)",
                    self.action_history)

                # Save metadata
                cursor.execute("INSERT INTO metadata (key, value) VALUES ('turn_count', ?)", (str(self.turn_count),))
                cursor.execute("INSERT INTO metadata (key, value) VALUES ('next_character_id', ?)",
                               (str(self.next_character_id),))
                cursor.execute("INSERT INTO metadata (key, value) VALUES ('next_object_id', ?)",
                               (str(self.next_object_id),))

                cursor.execute('COMMIT')
            except Exception as e:
                print(f"Database save failed: {e}. Rolling back transaction.")
                cursor.execute('ROLLBACK')
                raise

    def load_world(self):
        """Loads the world state from the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Make sure tables exist before trying to read
            self._init_db(conn)

            # Load metadata
            meta = {row['key']: row['value'] for row in cursor.execute("SELECT key, value FROM metadata").fetchall()}
            self.turn_count = int(meta.get('turn_count', 1))
            self.next_character_id = int(meta.get('next_character_id', 0))
            self.next_object_id = int(meta.get('next_object_id', 1))

            # Load action history
            self.action_history = [tuple(row) for row in cursor.execute(
                "SELECT turn, character_id, instruction, result FROM action_history ORDER BY turn")]

            # Load characters
            self.characters = {}
            for row in cursor.execute("SELECT * FROM characters").fetchall():
                char = Character(
                    id=row['id'], name=row['name'],
                    position=Vector3D(**json.loads(row['position_json'])),
                    conversation_partners=set(json.loads(row['conversation_partners_json']))
                )
                self.characters[char.id] = char

            # Load all objects and assign to world or inventories
            self.world_objects = {}
            for char in self.characters.values():
                char.inventory = []

            for row in cursor.execute("SELECT * FROM world_objects").fetchall():
                obj = WorldObject(
                    id=row['id'], name=row['name'], description=row['description'], owner_id=row['owner_id'],
                    position=Vector3D(**json.loads(row['position_json'])),
                    size=Vector3D(**json.loads(row['size_json'])),
                    relationships=[Relationship(**r) for r in json.loads(row['relationships_json'])],
                    properties=json.loads(row['properties_json'])
                )
                inv_char_id = row['inventory_of_character_id']
                if inv_char_id is not None:
                    if inv_char_id in self.characters:
                        self.characters[inv_char_id].inventory.append(obj)
                else:
                    self.world_objects[obj.id] = obj

    def add_character(self, name: str, position: Vector3D) -> Character:
        """Adds a new character to the world."""
        char_id = self.next_character_id
        new_char = Character(id=char_id, name=name, position=position)
        self.characters[char_id] = new_char
        self.next_character_id += 1
        print(f"Added character {name} (ID: {char_id}) to the world.")
        return new_char

    def _build_relationship_map(self):
        """Creates a reverse map for finding objects related to a target."""
        self.relationship_map: Dict[int, List[Tuple[int, RelationshipType]]] = {id: [] for id in
                                                                                self.world_objects.keys()}
        for obj in self.world_objects.values():
            for rel in obj.relationships:
                if rel.target_id in self.relationship_map:
                    self.relationship_map[rel.target_id].append((obj.id, rel.type))

    def _get_entities_near(self, position: Vector3D, radius: float) -> Tuple[List[WorldObject], List[Character]]:
        """Finds all world objects and characters within a given radius."""
        nearby_obj = []
        for obj in self.world_objects.values():
            distance = math.dist((position.x, position.y, position.z), (obj.position.x, obj.position.y, obj.position.z))
            if distance < radius + max(obj.size.x, obj.size.y) / 2:
                nearby_obj.append(obj)

        nearby_char = []
        for char in self.characters.values():
            distance = math.dist((position.x, position.y, position.z),
                                 (char.position.x, char.position.y, char.position.z))
            if distance < radius:
                nearby_char.append(char)

        return nearby_obj, nearby_char

    def _find_entity(self, name: str, context: Optional[str], search_area_obj: List[WorldObject],
                     search_area_char: List[Character]) -> Optional[WorldObject | Character]:
        """Finds a specific object or character by name."""
        name_lower = name.lower()
        # Search for characters first
        char_candidates = [c for c in search_area_char if name_lower in c.name.lower()]
        if char_candidates:
            return char_candidates[0]  # Assume unique names for now

        # Search for objects
        obj_candidates = [obj for obj in search_area_obj if name_lower in obj.name.lower()]
        if not obj_candidates:
            return None
        if len(obj_candidates) == 1:
            return obj_candidates[0]

        if context:
            for candidate in obj_candidates:
                if context.lower() in candidate.description.lower() or context.lower() in candidate.name.lower():
                    return candidate

        return obj_candidates[0]

    # --- Observer Agent ---
    def _describe_environment_at(self, position: Vector3D, acting_char_id: int) -> str:
        """The core of the Observer Agent. Generates a natural language description for a specific character."""
        nearby_objects, nearby_characters = self._get_entities_near(position, radius=15.0)
        acting_char = self.characters[acting_char_id]

        if not nearby_objects and not nearby_characters:
            return "You see nothing but an endless, uniform void."

        current_location_context = "in an open, grassy field"
        for obj in nearby_objects:
            if obj.properties.get("type") == "building" and math.dist((position.x, position.y),
                                                                      (obj.position.x, obj.position.y)) < max(
                obj.size.x, obj.size.y) / 2:
                current_location_context = f"inside {obj.name}"
                break

        object_summaries = []
        described_ids = set()

        # Describe other characters
        for char in nearby_characters:
            if char.id == acting_char_id:
                continue

            convo_status = ""
            if char.id in acting_char.conversation_partners:
                convo_status = f" You are currently in a conversation with them."
            elif char.conversation_partners:
                partners = [self.characters[p_id].name for p_id in char.conversation_partners if
                            p_id in self.characters]
                convo_status = f" They seem to be talking to {', '.join(partners)}."

            summary = f"- '{char.name}' is here.{convo_status}"
            object_summaries.append(summary)

        # Describe objects
        nearby_objects.sort(key=lambda o: math.dist((position.x, position.y), (o.position.x, o.position.y)))
        for obj in nearby_objects:
            if obj.id in described_ids or "Ground" in obj.name:
                continue

            summary = f"- A '{obj.name}': {obj.description}"
            related_items_info = []
            if obj.id in self.relationship_map:
                for related_id, rel_type in self.relationship_map[obj.id]:
                    if related_id in self.world_objects:
                        related_obj = self.world_objects[related_id]
                        related_items_info.append(f"  - There is a '{related_obj.name}' {rel_type.value} it.")
                        described_ids.add(related_id)

            if related_items_info:
                summary += "\n" + "\n".join(related_items_info)
            object_summaries.append(summary)
            described_ids.add(obj.id)

        prompt = f"""
        You are a descriptive narrator for a text-based RPG. Your task is to turn a structured list of nearby objects and people into a fluid, immersive paragraph.

        The character, {acting_char.name}, is currently located: {current_location_context}.

        Here is the structured list of things they see:
        ---
        {chr(10).join(object_summaries)}
        ---

        Please write a single, well-formed paragraph describing this scene from the character's point of view ("You see..."). Do not use lists or bullet points. Combine related items naturally. Be vivid and engaging.
        """
        description = main_llm.completion_text(
            LlmPreset.Default,
            [("system", "You are a helpful and descriptive narrator."), ("user", prompt)],
            CommonCompSettings(temperature=0.5, max_tokens=512)
        )
        return description if description else "You look around but your thoughts are a jumble."

    # --- Interaction Agent ---
    def _execute_interaction(self, instruction: str, acting_char_id: int) -> str:
        """The core of the Interaction Agent. Parses and executes a command for a given character."""
        acting_char = self.characters[acting_char_id]

        class ActionIntent(BaseModel):
            verb: str = Field(
                description="Normalized verb: 'go_to', 'pick_up', 'look_at', 'drop', 'give', 'open', 'wait', 'initiate_conversation', 'leave_conversation'.")
            target_name: Optional[str] = Field(None,
                                               description="The name of the primary target (e.g., 'book', 'key', 'cottage').")
            recipient_name: Optional[str] = Field(None,
                                                  description="For 'give' actions, the name of the character receiving the item (e.g., 'Tom', 'Lily').")
            target_context: Optional[str] = Field(None,
                                                  description="Extra descriptive words for the target (e.g., 'red', 'on the table').")

        prompt = f"Parse the game command into a structured action. Normalize the verb.\nCommand: \"{instruction}\"\nValid verbs: go_to, pick_up, look_at, drop, give, open, wait, initiate_conversation, leave_conversation."
        _, calls = main_llm.completion_tool(
            LlmPreset.Default, [("system", "You are a command parser for a text RPG."), ("user", prompt)],
            tools=[ActionIntent]
        )
        if not calls: return "You ponder your action, but can't decide what to do."
        intent: ActionIntent = calls[0]

        nearby_obj, nearby_char = self._get_entities_near(acting_char.position, radius=5.0)

        # Find target entity (object or character)
        target_entity = None
        if intent.target_name:
            search_scope_obj = nearby_obj if intent.verb not in ['go_to'] else list(self.world_objects.values())
            search_scope_char = nearby_char if intent.verb not in ['go_to'] else list(self.characters.values())
            target_entity = self._find_entity(intent.target_name, intent.target_context, search_scope_obj,
                                              search_scope_char)

        # --- Handle Verbs ---
        if intent.verb == 'wait':
            return "You wait for a moment, observing your surroundings."

        if intent.verb == 'look_at':
            if not target_entity: return f"You don't see a '{intent.target_name}' nearby."
            return f"You look closely at {target_entity.name}. {target_entity.description}"

        if intent.verb == 'go_to':
            if not target_entity: return f"You can't find a place or person called '{intent.target_name}' to go to."
            acting_char.position = target_entity.position
            return f"You walk over to {target_entity.name}."

        if intent.verb == 'pick_up':
            if not target_entity: return f"You don't see a '{intent.target_name}' here to pick up."
            if isinstance(target_entity, Character): return f"You can't pick up {target_entity.name}!"

            target_obj: WorldObject = target_entity
            if max(target_obj.size.x, target_obj.size.y,
                   target_obj.size.z) > 1.0: return f"The {target_obj.name} is too large to pick up."

            del self.world_objects[target_obj.id]
            acting_char.inventory.append(target_obj)
            self._build_relationship_map()

            ownership_text = ""
            if target_obj.owner_id is not None:
                if target_obj.owner_id == acting_char_id:
                    ownership_text = " your"
                else:
                    ownership_text = f" {self.characters[target_obj.owner_id].name}'s"
            return f"You pick up the{ownership_text} {target_obj.name} and put it in your inventory."

        if intent.verb == 'give':
            if not intent.target_name: return "What do you want to give?"
            if not intent.recipient_name: return "Who do you want to give it to?"

            # Find the item in inventory
            item_to_give = next(
                (item for item in acting_char.inventory if intent.target_name.lower() in item.name.lower()), None)
            if not item_to_give: return f"You don't have a '{intent.target_name}' to give."

            # Find the recipient character nearby
            recipient = next((char for char in nearby_char if intent.recipient_name.lower() in char.name.lower()), None)
            if not recipient: return f"There is no one named '{intent.recipient_name}' nearby to give anything to."
            if recipient.id == acting_char.id: return "You can't give something to yourself."

            # Perform the transfer
            acting_char.inventory.remove(item_to_give)
            recipient.inventory.append(item_to_give)
            return f"You give the {item_to_give.name} to {recipient.name}."

        if intent.verb == 'initiate_conversation':
            if not target_entity: return f"You don't see anyone named '{intent.target_name}' to talk to."
            if not isinstance(target_entity, Character): return f"You can't talk to {target_entity.name}."

            target_char: Character = target_entity
            if target_char.id == acting_char_id: return "You mutter to yourself."

            acting_char.conversation_partners.add(target_char.id)
            target_char.conversation_partners.add(acting_char.id)
            return f"You start a conversation with {target_char.name}."

        if intent.verb == 'leave_conversation':
            if not acting_char.conversation_partners: return "You are not in a conversation."

            parted_names = []
            for partner_id in list(acting_char.conversation_partners):
                if partner_id in self.characters:
                    partner_char = self.characters[partner_id]
                    partner_char.conversation_partners.discard(acting_char_id)
                    parted_names.append(partner_char.name)
            acting_char.conversation_partners.clear()
            return f"You leave the conversation with {', '.join(parted_names)}."

        return f"You're not sure how to '{instruction}'."

    # --- Public Tool Methods ---
    def advance_turn(self):
        """Advances the world simulation by one turn. Called by the simulation loop."""
        self.turn_count += 1
        print(f"VWM: Advanced to turn {self.turn_count}.")
        # No need to save here, interact will handle it.
        return self.turn_count

    def observe(self, character_id: int) -> str:
        """Generates a description of the character's surroundings."""
        if character_id not in self.characters: return "Error: Invalid character ID."
        char = self.characters[character_id]
        description = self._describe_environment_at(char.position, char.id)

        # Log this action to history
        self.action_history.append((self.turn_count, character_id, "observe", description))
        self.save_world()
        return description

    def interact(self, character_id: int, instruction: str) -> str:
        """Parses and executes a natural language command in the world for a character."""
        if character_id not in self.characters: return "Error: Invalid character ID."
        result = self._execute_interaction(instruction, character_id)

        # The turn number is now managed externally by advance_turn(). Just record the action.
        self.action_history.append((self.turn_count, character_id, instruction, result))
        self.save_world()
        return result

    def check_inventory(self, character_id: int) -> str:
        """Checks a specific character's inventory."""
        if character_id not in self.characters: return "Error: Invalid character ID."
        char = self.characters[character_id]
        if not char.inventory:
            result = "Your inventory is empty."
        else:
            item_lines = [f"- {item.name} ({item.description})" for item in char.inventory]
            result = "You check your inventory and find:\n" + "\n".join(item_lines)

        # Log this action to history
        self.action_history.append((self.turn_count, character_id, "check_inventory", result))
        self.save_world()
        return result


# --- 4. LLM Tool and Action Result Schemas (Unchanged) ---

class ObserveAction(BaseModel):
    """Use this action to look around and get a description of the current environment. Use this if you want to wait or do nothing for a turn."""
    pass


class InteractAction(BaseModel):
    """Use this action to perform an action or interact with an object or character. Describe what you want to do in plain English."""
    instruction: str = Field(...,
                             description="A natural language command, e.g., 'pick up the key', 'go to Lily's cottage', 'give key to Tom', 'wait'.")


class InventoryAction(BaseModel):
    """Use this action to check what items you are currently carrying."""
    pass


class WorldAction(BaseModel):
    """A tool for the AI to perceive or act within the virtual world. Choose one of the available actions."""
    action_type: str = Field(..., description="The type of action to perform: 'observe', 'interact', or 'inventory'.")
    action_params: Optional[Dict[str, Any]] = Field(None, description="The parameters for the chosen action, if any.")


class ActionResult(BaseModel):
    """The result of an action performed in the virtual world."""
    action_possible: bool = Field(description="Whether the action was possible to execute.")
    outcome_description: str = Field(description="A text description of what happened as a result of the action.")


# --- 5. Threading and Proxy Infrastructure (Updated for multi-character) ---

@dataclass(order=True)
class WorldTask:
    """A task to be processed by the VirtualWorldManager worker thread."""
    priority: int = field(default_factory=int)
    method_name: str = field(default_factory=str)
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    result_queue: queue.Queue = field(compare=False, default_factory=queue.Queue)


world_task_queue: "queue.PriorityQueue[WorldTask]" = queue.PriorityQueue()
_world_manager_instance: Optional[VirtualWorldManager] = None


def world_manager_worker(character_names: List[str], db_path: str):
    """The worker function that runs in a dedicated thread, processing world tasks."""
    global _world_manager_instance
    _world_manager_instance = VirtualWorldManager(db_path=db_path, character_names=character_names)
    print("VWM Worker: Thread started and manager initialized.")
    while True:
        task: WorldTask = world_task_queue.get()
        if task is None: break
        try:
            method_to_call = getattr(_world_manager_instance, task.method_name)
            result = method_to_call(*task.args, **task.kwargs)
            task.result_queue.put(result)
        except Exception as e:
            print(f"VWM Worker Error: Failed to execute task '{task.method_name}'. Error: {e}")
            task.result_queue.put(f"An error occurred in the world: {e}")
        finally:
            world_task_queue.task_done()


class VirtualWorldManagerProxy:
    """A thread-safe proxy to interact with the VirtualWorldManager from other threads."""

    def __init__(self, priority: int = 1):
        self.priority = priority

    def _execute_task(self, method_name: str, *args, **kwargs) -> Any:
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        task = WorldTask(priority=self.priority, method_name=method_name, args=args, kwargs=kwargs,
                         result_queue=result_q)
        world_task_queue.put(task)
        return result_q.get()

    def advance_turn(self) -> int:
        return self._execute_task('advance_turn')

    def observe(self, character_id: int) -> str:
        return self._execute_task('observe', character_id=character_id)

    def interact(self, character_id: int, instruction: str) -> str:
        return self._execute_task('interact', character_id=character_id, instruction=instruction)

    def check_inventory(self, character_id: int) -> str:
        return self._execute_task('check_inventory', character_id=character_id)

    def get_character_data(self) -> Dict[int, str]:
        if _world_manager_instance:
            return {char.id: char.name for char in _world_manager_instance.characters.values()}
        return {}


# --- 6. Standalone Test Block (Heavily Updated for Multi-Agent Simulation) ---

class SimulatedAgent:
    """A simple LLM-powered agent to simulate a character's decision-making."""

    def __init__(self, character_id: int, character_name: str, goal: str, world_proxy: VirtualWorldManagerProxy):
        self.char_id = character_id
        self.name = character_name
        self.goal = goal
        self.world = world_proxy
        self.memory: List[str] = []
        self.is_goal_complete = False

    def decide_and_act(self):
        """The core loop for the agent: observe, think, act."""
        print(f"\n--- {self.name}'s Turn (ID: {self.char_id}) ---")

        if self.is_goal_complete:
            print(f"Goal: {self.goal} (COMPLETED)")
            print("[Action] My goal is complete. I will wait and see what happens.")
            action_result = self.world.interact(self.char_id, "wait")
            print(f"[Outcome] {action_result}")
            return

        print(f"Goal: {self.goal}")

        # 1. Observe the environment
        observation = self.world.observe(self.char_id)
        print(f"[Observation] {observation}")
        self.memory.append(f"I saw: {observation}")

        # 2. Check inventory
        inventory = self.world.check_inventory(self.char_id)
        print(f"[Inventory] {inventory}")
        self.memory.append(f"I was carrying: {inventory}")
        if len(self.memory) > 10: self.memory.pop(0)

        # 3. Use LLM to decide on the next action
        class AgentDecision(BaseModel):
            """The AI agent's decision on what to do next."""
            thought: str = Field(
                description="A brief rationale for the chosen action, considering the goal and recent events.")
            action: InteractAction | ObserveAction | InventoryAction = Field(
                description="The specific action to perform in the world.")

        system_prompt = f"""
        You are an AI character named {self.name} in a virtual world.
        Your ultimate goal is: "{self.goal}"
        You need to decide what single action to take right now to progress towards your goal.
        Your available actions are to interact with things (e.g. 'go to', 'pick up', 'give', 'wait'), observe your surroundings, or check your inventory.
        Base your decision on your goal, your most recent observation, and your inventory.
        """

        nl = "\n"
        user_prompt = f"""
        Here is what just happened:
        - My current observation: {observation}
        - My current inventory: {inventory}
        - My recent memory: {nl.join(self.memory)}

        Considering my goal ("{self.goal}"), what is my next logical action?
        """

        _, calls = main_llm.completion_tool(
            LlmPreset.Default,
            [("system", system_prompt), ("user", user_prompt)],
            tools=[AgentDecision],
            comp_settings=CommonCompSettings(temperature=0.7)
        )

        if not calls:
            print("[Decision] The agent is confused and decides to wait.")
            action_result = self.world.interact(self.char_id, "wait")
            print(f"[Outcome] {action_result}")
            return

        decision: AgentDecision = calls[0]
        print(f"[Thought] {decision.thought}")

        # 4. Execute the chosen action (refined logic)
        action_result = "No action taken."
        action_text_for_memory = ""
        if isinstance(decision.action, InteractAction):
            print(f"[Action] Attempting to: {decision.action.instruction}")
            action_text_for_memory = decision.action.instruction
            action_result = self.world.interact(self.char_id, decision.action.instruction)
        elif isinstance(decision.action, ObserveAction):
            print("[Action] Decided to observe.")
            action_text_for_memory = "observe"
            action_result = self.world.observe(self.char_id)
        elif isinstance(decision.action, InventoryAction):
            print("[Action] Decided to check inventory.")
            action_text_for_memory = "check_inventory"
            action_result = self.world.check_inventory(self.char_id)

        print(f"[Outcome] {action_result}")
        self.memory.append(f"I did '{action_text_for_memory}' and the result was '{action_result}'.")

        # 5. Check for goal completion
        if "You give the Brass Key to Tom" in action_result:
            self.is_goal_complete = True
            print("[Goal Status] Lily's goal is complete!")


def run_simulation():
    """Main simulation loop."""
    print("--- Running Multi-Agent Simulation Test ---")

    llama_worker_thread = threading.Thread(target=llama_worker, daemon=True)
    llama_worker_thread.start()

    db_file = "multi_agent_world.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed old test database '{db_file}'.")

    character_names = ["Lily", "Tom"]

    # Start the world manager thread
    world_worker_thread = threading.Thread(target=world_manager_worker, args=(character_names, db_file), daemon=True)
    world_worker_thread.start()
    time.sleep(2)  # Give it time to initialize the world

    world_proxy = VirtualWorldManagerProxy()

    # Get character IDs from the running instance
    char_map = world_proxy.get_character_data()
    if not char_map:
        print("Error: Could not retrieve character data from world manager.")
        world_task_queue.put(None)
        world_worker_thread.join()
        return

    lily_id = next(id for id, name in char_map.items() if name == "Lily")
    tom_id = next(id for id, name in char_map.items() if name == "Tom")

    # Create agents with goals
    agents = [
        SimulatedAgent(lily_id, "Lily", "Find the brass key in my house and give it to Tom.", world_proxy),
        SimulatedAgent(tom_id, "Tom", "Go to Lily's Cottage and wait for her to give you the key.", world_proxy)
    ]

    # Simulation loop for a few turns
    for i in range(5):
        print(f"\n==================== SIMULATION ROUND {i + 1} ====================")

        # This is the crucial fix: advance the world's turn counter once per round.
        world_proxy.advance_turn()

        for agent in agents:
            # Check for Tom's goal completion
            if agent.name == "Tom":
                inventory_state = world_proxy.check_inventory(agent.char_id)
                if "Brass Key" in inventory_state:
                    if not agent.is_goal_complete:
                        print(f"[Goal Status] Tom received the key. His goal is complete!")
                        agent.is_goal_complete = True

            agent.decide_and_act()
            time.sleep(1)  # Pause between agent actions for readability

    print("\n--- Simulation Complete ---")
    world_task_queue.put(None)
    world_worker_thread.join(timeout=3)


if __name__ == "__main__":
    run_simulation()