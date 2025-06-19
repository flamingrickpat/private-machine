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
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

# Assuming pm_lida.py is in the same project and we can import from it.
# In a real project, these would be in a shared 'utils' or 'llm' module.
from pm_lida import main_llm, LlmPreset, CommonCompSettings, llama_worker


# --- 1. Core Data Models for the Virtual World (Unchanged) ---

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

    position: Vector3D = Field(default_factory=Vector3D, description="The object's center point in world coordinates.")
    size: Vector3D = Field(default_factory=lambda: Vector3D(x=1.0, y=1.0, z=1.0),
                           description="The object's bounding box dimensions.")

    relationships: List[Relationship] = Field(default_factory=list,
                                              description="List of relationships to other objects.")
    properties: Dict[str, Any] = Field(default_factory=dict,
                                       description="Arbitrary properties like 'color', 'is_openable', 'current_state'.")


class Character(BaseModel):
    """Represents the AI companion's state within the virtual world."""
    id: int = 0
    name: str
    position: Vector3D = Field(default_factory=Vector3D)
    inventory: List[WorldObject] = Field(default_factory=list)


# --- 2. Procedural World Generation (Unchanged) ---

class WorldGenerator:
    """Creates the initial state of the virtual world."""

    def __init__(self, seed: int = 1337):
        self.rng = random.Random(seed)
        self.current_id = 1

    def _get_next_id(self) -> int:
        id = self.current_id
        self.current_id += 1
        return id

    def generate_world(self) -> List[WorldObject]:
        """Generates a small village scene."""
        objects = []

        # Ground plane
        objects.append(WorldObject(id=self._get_next_id(), name="Grassy Ground",
                                   description="A vast, flat expanse of soft green grass.",
                                   position=Vector3D(x=0, y=0, z=-0.1), size=Vector3D(x=100, y=100, z=0.1)))

        # Generate a small cluster of houses
        for i in range(3):
            house_pos = Vector3D(x=self.rng.uniform(-20, 20), y=self.rng.uniform(-20, 20), z=0)
            is_furnished = (i == 0)  # Only furnish the first house
            objects.extend(self._generate_house(house_pos, is_furnished=is_furnished))

        # Generate a forest area
        for _ in range(50):
            tree_pos = Vector3D(x=self.rng.uniform(25, 45), y=self.rng.uniform(-25, 25), z=0)
            objects.append(WorldObject(
                id=self._get_next_id(), name="Oak Tree",
                description="A tall, sturdy oak tree with a thick trunk and sprawling branches.",
                position=tree_pos, size=Vector3D(x=2, y=2, z=15)
            ))

        return objects

    def _generate_house(self, position: Vector3D, is_furnished: bool) -> List[WorldObject]:
        house_objects = []

        # House structure
        house_id = self._get_next_id()
        house = WorldObject(
            id=house_id, name="Cozy Cottage",
            description="A small cottage made of wood and stone, with a chimney puffing gentle smoke.",
            position=position, size=Vector3D(x=10, y=8, z=5), properties={"type": "building"}
        )
        house_objects.append(house)

        if is_furnished:
            # Inside the house (relative positions)
            # A table
            table_pos = Vector3D(x=position.x, y=position.y, z=0)
            table_id = self._get_next_id()
            table = WorldObject(
                id=table_id, name="Wooden Table", description="A simple but sturdy wooden dining table.",
                position=table_pos, size=Vector3D(x=2, y=1, z=0.8),
                relationships=[Relationship(target_id=house_id, type=RelationshipType.INSIDE)]
            )
            house_objects.append(table)

            # A book on the table
            book_pos = Vector3D(x=table_pos.x, y=table_pos.y, z=table.size.z)
            book = WorldObject(
                id=self._get_next_id(), name="Old Book", description="A leather-bound book with a worn cover.",
                position=book_pos, size=Vector3D(x=0.3, y=0.2, z=0.05),
                relationships=[
                    Relationship(target_id=table_id, type=RelationshipType.ON_TOP_OF),
                    Relationship(target_id=house_id, type=RelationshipType.INSIDE)
                ]
            )
            house_objects.append(book)

            # A key also on the table
            key_pos = Vector3D(x=table_pos.x + 0.5, y=table_pos.y, z=table.size.z)
            house_objects.append(WorldObject(
                id=self._get_next_id(), name="Brass Key", description="A small, ornate brass key.",
                position=key_pos, size=Vector3D(x=0.1, y=0.02, z=0.01),
                relationships=[
                    Relationship(target_id=table_id, type=RelationshipType.ON_TOP_OF),
                    Relationship(target_id=house_id, type=RelationshipType.INSIDE)
                ]
            ))

        return house_objects


# --- 3. The Game Master / World Simulation Manager (UPDATED) ---

class VirtualWorldManager:
    """
    Manages the state of the virtual world, its objects, and the character.
    Acts as the Game Master for the AI companion. Now with SQLite persistence.
    """

    def __init__(self, character_name: str, db_path: str):
        self.db_path = db_path
        self.character_name = character_name
        self.turn_count = 0
        self.action_history: List[Tuple[int, str, str]] = []

        if os.path.exists(self.db_path):
            print(f"VWM: Loading world from '{self.db_path}'...")
            self.load_world()
        else:
            print("VWM: No database found. Generating new world...")
            generator = WorldGenerator()
            all_objects = generator.generate_world()
            self.world_objects: Dict[int, WorldObject] = {obj.id: obj for obj in all_objects}

            # Place character in the first furnished house
            main_house = next(obj for obj in self.world_objects.values() if obj.name == "Cozy Cottage")
            self.character = Character(name=character_name, position=main_house.position)
            self.save_world()  # Initial save of the new world

        # For easy lookup of object relationships
        self._build_relationship_map()
        print(
            f"VWM: World initialized. {self.character.name} is at {self.character.position.x:.1f}, {self.character.position.y:.1f}. Turn: {self.turn_count}")

    def _init_db(self, conn: sqlite3.Connection):
        """Creates the necessary database tables if they don't exist."""
        cursor = conn.cursor()

        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        # Main table for all objects
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS world_objects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                position_json TEXT,
                size_json TEXT,
                relationships_json TEXT,
                properties_json TEXT,
                is_in_inventory INTEGER DEFAULT 0
            );
        """)

        # Character state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS character_state (
                id INTEGER PRIMARY KEY,
                name TEXT,
                position_json TEXT
            );
        """)

        # Action history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_history (
                turn INTEGER PRIMARY KEY,
                instruction TEXT,
                result TEXT
            );
        """)
        conn.commit()

    def save_world(self):
        """Saves the entire current world state to the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            self._init_db(conn)
            cursor = conn.cursor()

            # Use a transaction for atomic writes
            cursor.execute('BEGIN')
            try:
                # Clear existing data
                cursor.execute('DELETE FROM world_objects')
                cursor.execute('DELETE FROM character_state')
                cursor.execute('DELETE FROM action_history')
                cursor.execute('DELETE FROM metadata')

                # Save all objects (world + inventory)
                all_objects_to_save = list(self.world_objects.values()) + self.character.inventory
                objects_data = [
                    (
                        obj.id, obj.name, obj.description,
                        obj.position.model_dump_json(), obj.size.model_dump_json(),
                        json.dumps([r.model_dump() for r in obj.relationships]),
                        json.dumps(obj.properties),
                        1 if obj in self.character.inventory else 0
                    ) for obj in all_objects_to_save
                ]
                cursor.executemany("""
                    INSERT INTO world_objects (id, name, description, position_json, size_json, relationships_json, properties_json, is_in_inventory)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, objects_data)

                # Save character state
                cursor.execute(
                    "INSERT INTO character_state (id, name, position_json) VALUES (?, ?, ?)",
                    (self.character.id, self.character.name, self.character.position.model_dump_json())
                )

                # Save action history
                cursor.executemany(
                    "INSERT INTO action_history (turn, instruction, result) VALUES (?, ?, ?)",
                    self.action_history
                )

                # Save metadata
                cursor.execute("INSERT INTO metadata (key, value) VALUES ('turn_count', ?)", (str(self.turn_count),))

                cursor.execute('COMMIT')
            except Exception as e:
                print(f"Database save failed: {e}. Rolling back transaction.")
                cursor.execute('ROLLBACK')

    def load_world(self):
        """Loads the world state from the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Access columns by name
            cursor = conn.cursor()

            # Load metadata
            cursor.execute("SELECT value FROM metadata WHERE key = 'turn_count'")
            row = cursor.fetchone()
            self.turn_count = int(row['value']) if row else 0

            # Load action history
            cursor.execute("SELECT turn, instruction, result FROM action_history ORDER BY turn")
            self.action_history = [tuple(row) for row in cursor.fetchall()]

            # Load character
            cursor.execute("SELECT * FROM character_state WHERE id = 0")
            char_row = cursor.fetchone()
            self.character = Character(
                name=self.character_name,
                position=Vector3D(**json.loads(char_row['position_json']))
            )

            # Load all objects
            cursor.execute("SELECT * FROM world_objects")
            self.world_objects = {}
            inventory_objects = []
            all_loaded_objects = {}

            for row in cursor.fetchall():
                obj = WorldObject(
                    id=row['id'], name=row['name'], description=row['description'],
                    position=Vector3D(**json.loads(row['position_json'])),
                    size=Vector3D(**json.loads(row['size_json'])),
                    relationships=[Relationship(**r) for r in json.loads(row['relationships_json'])],
                    properties=json.loads(row['properties_json'])
                )
                all_loaded_objects[obj.id] = obj

                if row['is_in_inventory']:
                    inventory_objects.append(obj)
                else:
                    self.world_objects[obj.id] = obj

            self.character.inventory = inventory_objects

    def _build_relationship_map(self):
        """Creates a reverse map for finding objects related to a target."""
        self.relationship_map = {id: [] for id in self.world_objects.keys()}
        for obj in self.world_objects.values():
            for rel in obj.relationships:
                if rel.target_id in self.relationship_map:
                    self.relationship_map[rel.target_id].append((obj.id, rel.type))

    def _get_objects_near(self, position: Vector3D, radius: float) -> List[WorldObject]:
        """Finds all world objects within a given radius of a position."""
        nearby = []
        for obj in self.world_objects.values():
            distance = math.dist((position.x, position.y, position.z), (obj.position.x, obj.position.y, obj.position.z))
            # A simple check that also considers the object's size
            if distance < radius + max(obj.size.x, obj.size.y) / 2:
                nearby.append(obj)
        return nearby

    def _find_object(self, name: str, context: str, search_area: List[WorldObject]) -> Optional[WorldObject]:
        """Finds a specific object by name, using context to disambiguate."""
        candidates = [obj for obj in search_area if name.lower() in obj.name.lower()]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Disambiguation with context (could be an LLM call for more complex cases)
        if context:
            for candidate in candidates:
                if context.lower() in candidate.description.lower() or context.lower() in candidate.name.lower():
                    return candidate

        return candidates[0]  # Fallback to the first match

    # --- Observer Agent ---
    def _describe_environment_at(self, position: Vector3D) -> str:
        """The core of the Observer Agent. Generates a natural language description."""
        nearby_objects = self._get_objects_near(position, radius=15.0)

        if not nearby_objects:
            return "You see nothing but an endless, uniform void."

        # Find the "ground" or building the character is on/in
        current_location_context = "an open, grassy field"
        for obj in nearby_objects:
            if obj.properties.get("type") == "building" and math.dist((position.x, position.y),
                                                                      (obj.position.x, obj.position.y)) < max(
                    obj.size.x, obj.size.y) / 2:
                current_location_context = f"inside the {obj.name}"
                break

        # Prepare a structured summary for the LLM
        object_summaries = []
        described_ids = set()

        # Sort objects by distance to describe closer things first
        nearby_objects.sort(key=lambda o: math.dist((position.x, position.y), (o.position.x, o.position.y)))

        for obj in nearby_objects:
            if obj.id in described_ids or obj.name == "Grassy Ground":
                continue

            summary = f"- A '{obj.name}': {obj.description}"

            # Find objects that have a relationship TO this object
            related_items_info = []
            if obj.id in self.relationship_map:
                for related_id, rel_type in self.relationship_map[obj.id]:
                    if related_id in self.world_objects:
                        related_obj = self.world_objects[related_id]
                        related_items_info.append(f"  - There is a '{related_obj.name}' {rel_type.value} it.")
                        described_ids.add(related_id)  # Mark as described

            if related_items_info:
                summary += "\n" + "\n".join(related_items_info)

            object_summaries.append(summary)
            described_ids.add(obj.id)

        # Use an LLM to "narrate" the structured list
        prompt = f"""
        You are a descriptive narrator for a text-based RPG. Your task is to turn a structured list of nearby objects into a fluid, immersive paragraph.

        The character, {self.character.name}, is currently located in: {current_location_context}.

        Here is the structured list of things they see:
        ---
        {chr(10).join(object_summaries)}
        ---

        Please write a single, well-formed paragraph describing this scene from the character's point of view ("You see..."). Do not use lists or bullet points. Combine related items naturally.
        """

        description = main_llm.completion_text(
            LlmPreset.Default,
            [("system", "You are a helpful and descriptive narrator."), ("user", prompt)],
            CommonCompSettings(temperature=0.5, max_tokens=512)
        )

        return description if description else "You look around but your thoughts are a jumble."

    # --- Interaction Agent ---
    def _execute_interaction(self, instruction: str) -> str:
        """The core of the Interaction Agent. Parses and executes a command."""

        # 1. Parse Intent with an LLM
        class ActionIntent(BaseModel):
            """The parsed structure of a player's command."""
            verb: str = Field(
                description="The core action, normalized to a verb like 'go_to', 'pick_up', 'look_at', 'drop', 'open'.")
            target_name: Optional[str] = Field(None,
                                               description="The name of the primary object of the action (e.g., 'book', 'cottage').")
            target_context: Optional[str] = Field(None,
                                                  description="Any extra descriptive words to help identify the target (e.g., 'red', 'on the table').")

        prompt = f"""
        Parse the following game command into a structured action. Normalize the verb.

        Command: "{instruction}"

        Valid verbs: go_to, pick_up, look_at, drop, open, put_on.
        """

        _, calls = main_llm.completion_tool(
            LlmPreset.Default,
            [("system", "You are a command parser for a text RPG."), ("user", prompt)],
            tools=[ActionIntent]
        )

        if not calls:
            return "You ponder your action, but can't decide what to do."

        intent: ActionIntent = calls[0]

        # 2. Validate and Execute the Intent
        nearby_objects = self._get_objects_near(self.character.position, radius=5.0)

        # Handle 'look_at' (it's just a more specific observation)
        if intent.verb == 'look_at' and intent.target_name:
            target_obj = self._find_object(intent.target_name, intent.target_context, nearby_objects)
            if target_obj:
                return f"You look closely at the {target_obj.name}. {target_obj.description}"
            else:
                return f"You don't see a '{intent.target_name}' nearby."

        # Handle 'go_to'
        if intent.verb == 'go_to' and intent.target_name:
            # Search all objects for destination, not just nearby
            target_obj = self._find_object(intent.target_name, intent.target_context, list(self.world_objects.values()))
            if target_obj:
                self.character.position = target_obj.position
                return f"You walk over to the {target_obj.name}."
            else:
                return f"You can't find a place called '{intent.target_name}' to go to."

        # Handle 'pick_up'
        if intent.verb == 'pick_up' and intent.target_name:
            target_obj = self._find_object(intent.target_name, intent.target_context, nearby_objects)
            if not target_obj:
                return f"You don't see a '{intent.target_name}' here to pick up."

            if max(target_obj.size.x, target_obj.size.y, target_obj.size.z) > 1.0:
                return f"The {target_obj.name} is too large to pick up."

            # Remove from world and relationships, add to inventory
            del self.world_objects[target_obj.id]
            self.character.inventory.append(target_obj)
            self._build_relationship_map()  # Rebuild map since an object was removed

            return f"You pick up the {target_obj.name} and put it in your inventory."

        return f"You're not sure how to '{instruction}'."

    # --- Public Tool Methods ---
    def observe(self) -> str:
        """Generates a description of the character's surroundings."""
        return self._describe_environment_at(self.character.position)

    def interact(self, instruction: str) -> str:
        """Parses and executes a natural language command in the world."""
        result = self._execute_interaction(instruction)
        self.turn_count += 1
        self.action_history.append((self.turn_count, instruction, result))
        self.save_world()  # Save state after every interaction
        return result

    def check_inventory(self) -> str:
        """Checks the character's inventory."""
        if not self.character.inventory:
            return "Your inventory is empty."

        item_names = [f"- {item.name} ({item.description})" for item in self.character.inventory]
        return "You check your inventory and find:\n" + "\n".join(item_names)


# --- 4. LLM Tool and Action Result Schemas (Unchanged) ---

class ObserveAction(BaseModel):
    """Use this action to look around and get a description of the current environment."""
    pass


class InteractAction(BaseModel):
    """Use this action to perform an action or interact with an object in the world. Describe what you want to do in plain English."""
    instruction: str = Field(...,
                             description="A natural language command describing the action, e.g., 'pick up the key' or 'go to the cozy cottage'.")


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


# --- 5. Threading and Proxy Infrastructure (Unchanged) ---

@dataclass(order=True)
class WorldTask:
    """A task to be processed by the VirtualWorldManager worker thread."""
    priority: int = field(default_factory=int)
    method_name: str = field(default_factory=str)
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    result_queue: queue.Queue = field(compare=False, default_factory=queue.Queue)


# The single queue for all world-related tasks.
world_task_queue: "queue.PriorityQueue[WorldTask]" = queue.PriorityQueue()
_world_manager_instance: Optional[VirtualWorldManager] = None


def world_manager_worker(character_name: str, db_path: str):
    """The worker function that runs in a dedicated thread, processing world tasks."""
    global _world_manager_instance
    _world_manager_instance = VirtualWorldManager(character_name=character_name, db_path=db_path)
    print("VWM Worker: Thread started and manager initialized.")

    while True:
        task: WorldTask = world_task_queue.get()
        if task is None:  # Sentinel for shutdown
            break

        try:
            method_to_call = getattr(_world_manager_instance, task.method_name)
            result = method_to_call(*task.args, **task.kwargs)
            task.result_queue.put(result)
        except Exception as e:
            print(f"VWM Worker Error: Failed to execute task '{task.method_name}'. Error: {e}")
            task.result_queue.put(f"An error occurred in the world: {e}")  # Propagate error
        finally:
            world_task_queue.task_done()


class VirtualWorldManagerProxy:
    """A thread-safe proxy to interact with the VirtualWorldManager from other threads."""

    def __init__(self, priority: int = 1):
        self.priority = priority

    def _execute_task(self, method_name: str, *args, **kwargs) -> Any:
        """Generic method to queue a task and wait for its result."""
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        task = WorldTask(
            priority=self.priority,
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            result_queue=result_q
        )
        world_task_queue.put(task)
        # Block until the result is available
        result = result_q.get()
        return result

    def observe(self) -> str:
        return self._execute_task('observe')

    def interact(self, instruction: str) -> str:
        return self._execute_task('interact', instruction=instruction)

    def check_inventory(self) -> str:
        return self._execute_task('check_inventory')


# --- 6. Standalone Test Block (UPDATED) ---

if __name__ == "__main__":
    worker_thread = threading.Thread(target=llama_worker, daemon=True)
    worker_thread.start()

    print("--- Running Virtual World Manager Standalone Test with Persistence ---")

    # In a real application, you'd get the character name from your config
    test_character_name = "Lily"
    db_file = "test_world.db"

    # Clean up previous test run if it exists
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed old '{db_file}' for a fresh test.")

    # --- FIRST RUN: Generate and Save World ---
    print("\n--- FIRST RUN ---")

    # Start the worker thread
    worker_thread = threading.Thread(target=world_manager_worker, args=(test_character_name, db_file), daemon=True)
    worker_thread.start()

    world = VirtualWorldManagerProxy()
    time.sleep(1)  # Give worker time to init

    print("\n[Test] Initial observation:")
    print(f"-> {world.observe()}")

    print("\n[Test] Picking up key:")
    print(f"-> {world.interact('pick up the brass key')}")

    print("\n[Test] Checking inventory:")
    print(f"-> {world.check_inventory()}")

    # Stop the first worker
    world_task_queue.put(None)
    worker_thread.join(timeout=2)
    print("\n--- First run complete. Worker stopped. World saved. ---")

    # --- SECOND RUN: Load World from DB ---
    print("\n\n--- SECOND RUN ---")

    # Start a new worker thread, which should load the DB
    worker_thread_2 = threading.Thread(target=world_manager_worker, args=(test_character_name, db_file), daemon=True)
    worker_thread_2.start()

    world_2 = VirtualWorldManagerProxy()
    time.sleep(1)

    print("\n[Test] Checking inventory immediately after load:")
    print(f"-> {world_2.check_inventory()}")

    print("\n[Test] Observing the cottage to see key is still gone:")
    print(f"-> {world_2.observe()}")

    print("\n[Test] Going to a new location:")
    print(f"-> {world_2.interact('go to an oak tree')}")

    print("\n--- Test Complete ---")
    world_task_queue.put(None)
    worker_thread_2.join(timeout=2)