import queue
import random
import sys
import threading
import time
from queue import Queue
from typing import Dict, Any
from typing import List
from typing import (
    Optional,
)
from typing import TypeVar

from py_linq import Enumerable
from pydantic import BaseModel

from pm.config_loader import *
from pm.data_structures import ActionType, NarrativeTypes, Stimulus, Action, KnoxelList, FeatureType, StimulusType
from pm.ghosts.base_ghost import GhostConfig
from pm.ghosts.ghost_lida import EngagementIdea, GhostLida
from pm.ghosts.persist_sqlite import PersistSqlite
from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_proxy import LlmManagerProxy, LlmTask, llama_worker
from pm.mental_states import NeedsAxesModel
from pm.utils.profile_utils import print_function_stats

logger = logging.getLogger(__name__)

# --- Type Variables ---
T = TypeVar('T')
BM = TypeVar('BM', bound=BaseModel)

# 3) A single PriorityQueue shared by all producers & the consumer.
task_queue: "queue.PriorityQueue[LlmTask]" = queue.PriorityQueue()

main_llm = LlmManagerProxy(task_queue, priority=0, log_path=log_path)
aux_llm = LlmManagerProxy(task_queue, priority=1, log_path=log_path)


class StimulusSimple:
    def __init__(self, _type: str, content: Any):
        self.type = _type
        self.content = content

    def __repr__(self):
        return f"Stimulus(type='{self.type}', content='{self.content}')"


# This class is NOT the singleton. It's the same Shell as before, but modified to broadcast.
class Shell:
    def __init__(self, ghost: GhostLida, db_path_local: str):
        self.input_queue = Queue()  # One input queue is fine
        self.output_queues: List[Queue] = []  # A list of output queues for broadcasting
        self.lock = threading.Lock()  # To safely modify the list of queues

        self.last_interaction_time = time.time()
        self.stop_event = threading.Event()

        # --- Configuration ---
        self.USER_INACTIVITY_TIMEOUT = 60 * 120
        self.ENGAGEMENT_STRATEGIST_INTERVAL = 60 * 30
        self.NEEDS_CHECK_INTERVAL = 60 * 5
        self.NEEDS_DECAY_FACTOR = 0.01
        self.NEEDS_CRITICAL_THRESHOLD = 0.25
        self.SYSTEM_TICK_INTERVAL = 1

        # --- Internal State ---
        self.last_user_interaction_time = time.time()
        self.last_engagement_check_time = time.time()
        self.last_needs_check_time = time.time()
        self.current_time_of_day = self._get_time_of_day()
        self.is_sleeping = False  # NEW: Flag to track sleep state
        self.sleep_until_time = 0  # NEW: Timestamp for when sleep ends

        worker_thread = threading.Thread(target=llama_worker, args=(model_map, task_queue), daemon=True)
        worker_thread.start()

        self.ghost = ghost
        self.db_path_local = db_path_local

        if os.path.isfile(db_path_local):
            pers = PersistSqlite(ghost)
            pers.load_state_sqlite(db_path_local)

    def get_full_chat_history(self) -> List[Dict[str, str]]:
        """Pass-through method to get history from the Ghost."""
        return self.ghost.get_chat_history()

    def register_client(self, output_queue: Queue):
        """Adds a client's output queue to the broadcast list."""
        with self.lock:
            print(f"Registering new client. Total clients: {len(self.output_queues) + 1}")
            self.output_queues.append(output_queue)

    def unregister_client(self, output_queue: Queue):
        """Removes a client's output queue."""
        with self.lock:
            try:
                self.output_queues.remove(output_queue)
                print(f"Unregistered client. Total clients: {len(self.output_queues)}")
            except ValueError:
                # Queue was already removed, which is fine.
                pass

    def _broadcast(self, message: dict):
        """Sends a message to all registered clients."""
        with self.lock:
            # We iterate over a copy in case the list is modified during iteration elsewhere.
            for q in list(self.output_queues):
                try:
                    q.put(message)
                except Exception as e:
                    print(f"Error broadcasting to a queue, removing it. Error: {e}")
                    self.unregister_client(q)

    def save(self):
        tick = self.ghost.current_tick_id
        pers = PersistSqlite(self.ghost)
        pers.save_state_sqlite(self.db_path_local)

    def rate_message(self, tick_id, rating):
        for s in self.ghost.states:
            if s.tick_id == tick_id:
                s.rating = rating
                pers = PersistSqlite(self.ghost)
                pers.save_state_sqlite(db_path)
                break

    def _get_time_of_day(self) -> str:
        """Helper to categorize the current hour into a descriptive period."""
        hour = time.localtime().tm_hour
        if 5 <= hour < 12: return "Morning"
        if 12 <= hour < 17: return "Afternoon"
        if 17 <= hour < 21: return "Evening"
        return "Night"

    def _process_stimulus(self, stimulus: 'Stimulus'):
        """Processes a single stimulus through the Ghost's cognitive cycle and executes the resulting action."""
        logger.info(f"Processing stimulus: {stimulus.stimulus_type} - '{stimulus.content[:80]}...'")
        self._broadcast({"type": "Status", "content": "Thinking..."})

        # The core cognitive cycle is called here
        action = self.ghost.tick(stimulus)

        self._execute_action(action)

        # Persist the state after the full cycle
        self.save()

    def _apply_needs_decay(self):
        """
        Applies a simple decay to the Ghost's needs state. This simulates the
        passage of time without running a full cognitive cycle.

        This requires a new method on your Ghost class: `get_current_needs_state()`
        and `set_current_needs_state(new_state)`.
        """
        logger.debug("Applying passive needs decay.")

        # You will need to add `get_current_needs_state()` to your Ghost class
        current_needs: NeedsAxesModel = self.ghost.get_current_needs_state()

        # Decay each need towards a baseline of 0.5
        for field_name, _ in current_needs.model_fields.items():
            current_value = getattr(current_needs, field_name)
            # Simple linear decay towards a baseline
            # new_value = current_value - self.NEEDS_DECAY_FACTOR * (current_value - 0.5)
            # Or simpler exponential decay towards 0
            new_value = current_value * (1.0 - self.NEEDS_DECAY_FACTOR)
            setattr(current_needs, field_name, max(0.0, new_value))  # Ensure it doesn't go below 0

        # You will need to add `set_current_needs_state()` to your Ghost class
        self.ghost.set_current_needs_state(current_needs)
        self.save()

    def _check_critical_needs(self):
        """
        Checks if any need has fallen below a critical threshold and, if so,
        generates a LowNeedTrigger stimulus.
        """
        current_needs: NeedsAxesModel = self.ghost.get_current_needs_state()

        critically_low_needs = []
        for field_name, field in current_needs.__class__.model_fields.items():
            value = getattr(current_needs, field_name)
            if value < self.NEEDS_CRITICAL_THRESHOLD:
                critically_low_needs.append((field.description, field_name, value))

        if critically_low_needs:
            # Find the MOST critical need (the lowest one)
            critically_low_needs.sort(key=lambda x: x[2])
            description, name, value = critically_low_needs[0]

            logger.info(f"Critical need detected: '{name}' is at {value:.2f}. Generating stimulus.")

            content = f"Internal monitoring shows my need for '{name}' is critically low ({value:.2f}). I feel a strong urge to address this."
            stim = Stimulus(source="System_Needs_Monitor", content=content, stimulus_type=StimulusType.LowNeedTrigger)

            self._process_stimulus(stim)

            # Return True to indicate a stimulus was generated, so we don't fire others in the same tick
            return True
        return False

    def _execute_action(self, action: 'Action'):
        """Translates the Ghost's chosen Action knoxel into a real-world effect."""
        logger.info(f"Executing action: {action.action_type} - '{action.content[:80]}...'")

        if action.action_type == ActionType.Sleep:
            logger.info("SHELL: Ghost is entering a sleep state.")
            # The Ghost's deliberation should have returned the duration
            duration_hours = 0  # action.action_details.get("sleep_duration_hours", 4)
            self.is_sleeping = True
            self.sleep_until_time = time.time() + duration_hours * 3600

        if action.action_type in [ActionType.Reply, ActionType.InitiateUserConversation, ActionType.ToolCallAndReply]:
            # In a real app, this would send to a websocket or API response.
            print(f"--- Output to User ---")
            print(f"{self.ghost.config.companion_name}: {action.content}")
            print(f"----------------------")

            res = action.content
            reply = {"type": "Reply", "content": res}
            self._broadcast(reply)

        elif action.action_type in [ActionType.ToolCall, ]:
            # Here, you would parse the action.content and use your mcp_client to call the tool.
            logger.info(f"SHELL: Simulating tool call with parameters: {action.content}")
            # The result of the tool call would then likely become the stimulus for the *next* tick.

        elif action.action_type == ActionType.InitiateInternalContemplation:
            logger.info("SHELL: Ghost is contemplating. No external action taken.")

        elif action.action_type == ActionType.Sleep:
            logger.info("SHELL: Ghost is entering a sleep state. System triggers will be paused.")
            # You could add logic here to pause system stimuli for a set duration.

        elif action.action_type == ActionType.Ignore:
            logger.info("SHELL: Ghost chose to ignore the stimulus. No external action taken.")
            reply = {"type": "Reply", "content": "<Ignore>"}
            self._broadcast(reply)

    def _check_system_triggers(self):
        """Checks for and generates internal stimuli based on time and state."""
        now = time.time()

        stimulus_generated = False

        # If sleeping, only wake up if the timer is up or a user interacts.
        if self.is_sleeping:
            if now > self.sleep_until_time:
                logger.info("Sleep duration finished. Generating WakeUp stimulus.")
                self.is_sleeping = False
                stim = Stimulus(source="System", content="The sleep cycle has completed.", stimulus_type=StimulusType.WakeUp)
                self._process_stimulus(stim)
            return  # Do not process other system triggers while sleeping

        # --- HIGHEST PRIORITY: Needs Check ---
        if now - self.last_needs_check_time > self.NEEDS_CHECK_INTERVAL:
            self._apply_needs_decay()
            stimulus_generated = self._check_critical_needs()
            self.last_needs_check_time = now

        if stimulus_generated: return  # Don't fire other triggers if a need is critical

        # 1. User Inactivity Check
        if now - self.last_user_interaction_time > self.USER_INACTIVITY_TIMEOUT:
            logger.info("User inactivity timeout reached. Generating stimulus.")
            content = f"The user has been inactive for over {self.USER_INACTIVITY_TIMEOUT / 60:.0f} minutes."
            stim = Stimulus(source=shell_system_name, content=content, stimulus_type=StimulusType.UserInactivity)
            self._process_stimulus(stim)
            # IMPORTANT: Reset the timer to now to prevent it from firing every tick.
            self.last_user_interaction_time = now

        # 2. Time of Day Change Check
        new_time_of_day = self._get_time_of_day()
        if new_time_of_day != self.current_time_of_day:
            logger.info(f"Time of day changed to {new_time_of_day}. Generating stimulus.")
            content = f"The time has transitioned to {new_time_of_day}."
            stim = Stimulus(source=shell_system_name, content=content, stimulus_type=StimulusType.TimeOfDayChange)
            self._process_stimulus(stim)
            self.current_time_of_day = new_time_of_day

        # 3. Proactive Engagement Strategist Check
        if now - self.last_engagement_check_time > self.ENGAGEMENT_STRATEGIST_INTERVAL:
            self._run_engagement_strategist()
            self.last_engagement_check_time = now

    def _run_engagement_strategist(self):
        """
        The core of the proactive idea generation module. It gathers intel,
        brainstorms with an LLM, and creates an EngagementOpportunity stimulus.
        """
        logger.info("Running Proactive Engagement Strategist...")
        try:
            # 1. Gather Intelligence
            user_profile = self.ghost.get_narrative(NarrativeTypes.PsychologicalAnalysis, user_name)
            companion_profile = self.ghost.get_narrative(NarrativeTypes.PsychologicalAnalysis, companion_name)

            # Get available tools
            ts = get_toolset_from_url(mcp_server_url)
            if not ts:
                logger.warning("Engagement Strategist: Could not retrieve tools. Aborting.")
                return

            # collect chat
            dialouge = Enumerable(self.ghost.all_features) \
                .where(lambda x: x.causal) \
                .where(lambda x: FeatureType.from_stimulus(x.feature_type)) \
                .order_by(lambda x: x.timestamp_world_begin) \
                .to_list()
            story_list = KnoxelList(dialouge)
            chat_history = story_list.get_story(self.ghost, max_tokens=1024)

            # 2. Build the Brainstorming Prompt
            prompt = f"""
            You are a creative strategist for an AI companion named {self.ghost.config.companion_name}. 
            Your goal is to brainstorm ONE thoughtful, proactive action the AI could take to delight her user, {self.ghost.config.user_name}, while he is away.

            **{self.ghost.config.companion_name} (AI Companion) Profile:**
            - {companion_profile}
            
            **{self.ghost.config.user_name} (User) Profile:**
            - {user_profile}
            
            **Recent Chat History**
            ### CHAT BEGIN
            {chat_history}
            ### CHAT END

            **AI's Available Tools:**
            {toolset.tool_docs}

            **Task:** 
            Based on the user's profile and the available tools, generate ONE creative, non-intrusive, and genuinely helpful idea.
            - If the idea involves a tool, specify the tool name and arguments in the `action_content` as a JSON string.
            - If the idea is to start a conversation, provide a thoughtful opening line in `action_content`.
            - The `thought_process` should be the AI's internal reasoning.
            - The `user_facing_summary` is what the AI might say *after* performing the action.

            Output a single, valid JSON object conforming to the `EngagementIdea` schema.
            """

            # 3. Call LLM for a structured idea
            _, calls = self.ghost.llm.completion_tool(
                preset=LlmPreset.Default,  # Use a capable model
                inp=[("system", "You are a helpful and creative assistant."), ("user", prompt)],
                tools=[EngagementIdea]
            )

            if not calls:
                logger.warning("Engagement Strategist: LLM did not generate a valid idea.")
                return

            idea: EngagementIdea = calls[0]

            # 4. Create and Process the Stimulus
            logger.info(f"Strategist generated idea: {idea.thought_process}")
            stim = Stimulus(
                source="Engagement_Strategist",
                content=idea.model_dump_json(),  # The content is the full JSON of the idea
                stimulus_type=StimulusType.EngagementOpportunity
            )
            self._process_stimulus(stim)

        except Exception as e:
            logger.error(f"Error in Engagement Strategist: {e}", exc_info=True)

    def force_system_trigger(self, trigger_type: Optional[StimulusType] = None):
        """
        Manually forces a system trigger for testing purposes, bypassing timers.
        If no type is specified, a random valid trigger is chosen.
        """
        if self.is_sleeping:
            logger.info("TEST: force_system_trigger called, but AI is sleeping. Ignoring.")
            return

        if trigger_type is None:
            # Choose a random trigger to simulate
            possible_triggers = [
                StimulusType.UserInactivity,
                StimulusType.TimeOfDayChange,
                StimulusType.LowNeedTrigger,
                StimulusType.EngagementOpportunity
            ]
            trigger_type = random.choice(possible_triggers)

        logger.info(f"TEST: Forcing system trigger of type: {trigger_type.value}")

        if trigger_type == StimulusType.UserInactivity:
            content = "System Agent: The user has been inactive."
            stim = Stimulus(source="System_Test", content=content, stimulus_type=StimulusType.UserInactivity)
            self._process_stimulus(stim)

        elif trigger_type == StimulusType.TimeOfDayChange:
            new_time = random.choice(["Morning", "Afternoon", "Evening", "Night"])
            self.current_time_of_day = new_time  # Update state to avoid immediate re-trigger
            content = f"System Agent: The time has transitioned to {new_time}."
            stim = Stimulus(source="System_Test", content=content, stimulus_type=StimulusType.TimeOfDayChange)
            self._process_stimulus(stim)

        elif trigger_type == StimulusType.LowNeedTrigger:
            # To make this realistic, we can manually decay a need first
            needs = self.ghost.get_current_needs_state()
            need_to_make_low = random.choice(list(needs.__class__.model_fields.keys()))
            setattr(needs, need_to_make_low, 0.1)  # Force a low value
            self.ghost.set_current_needs_state(needs)
            logger.info(f"TEST: Manually set need '{need_to_make_low}' to 0.1.")
            self._check_critical_needs()  # This will find the low need and create the stimulus

        elif trigger_type == StimulusType.EngagementOpportunity:
            self._run_engagement_strategist()

    def run(self):
        """The main loop of the Shell, running in its own thread."""
        while not self.stop_event.is_set():
            try:
                # 1. Check for external (user) input first
                stim_simple = self.input_queue.get(block=False)
                stimulus = Stimulus(source=user_name, content=stim_simple.content, stimulus_type=StimulusType.UserMessage)

                self.last_user_interaction_time = time.time()
                self.last_engagement_check_time = time.time()

                self._process_stimulus(stimulus)
            except queue.Empty:
                # 2. If no user input, check for internal system triggers
                self._check_system_triggers()
            except Exception as e:
                logger.error(f"Error in Shell main loop: {e}", exc_info=True)

            time.sleep(self.SYSTEM_TICK_INTERVAL)

        logger.info("Shell main loop has stopped.")

    def stop(self):
        """Signals the shell's main loop to stop."""
        logger.info("Shutdown signal received. Stopping Shell.")
        self.stop_event.set()


# --- THE SINGLETON LOGIC ---
_instance = None
_lock = threading.Lock()


def get_shell_instance() -> Shell:
    """
    Gets the single, global instance of the Shell.
    Initializes it and starts its thread on the first call.
    """
    global _instance
    with _lock:
        if _instance is None:
            print("--- CREATING SINGLETON SHELL INSTANCE ---")

            config = GhostConfig()
            ghost = GhostLida(main_llm, config)

            if db_path is not None and db_path != "":
                pers = PersistSqlite(ghost)
                pers.load_state_sqlite(db_path)

            _instance = Shell(ghost, db_path)
            thread = threading.Thread(target=_instance.run, daemon=True)
            thread.start()
            print("--- SINGLETON SHELL INSTANCE CREATED AND THREAD STARTED ---")
    return _instance


def start_llm_thread():
    worker_thread = threading.Thread(target=llama_worker, args=(model_map, task_queue), daemon=True)
    worker_thread.start()


def run(interactive: bool, _db_path: str | None = None, cmd: str = None):
    start_llm_thread()

    config = GhostConfig()
    ghost = GhostLida(main_llm, config)

    create_new_persona = True
    if _db_path is not None and _db_path != "":
        pers = PersistSqlite(ghost)
        if pers.load_state_sqlite(_db_path):
            create_new_persona = False

    if create_new_persona:
        ghost.init_character()

    # Start interactive loop
    user_input_cmd_test = f"Hey {companion_name}. I just installed you. How are you?"

    persona_system = f"""Pretend to be a human user talking with a chatbot.
Your name is {user_name}. Never go to sleep, continue talking at all costs. Explore different topics, don't get too philosophical or technical.
The chatbot is called "{companion_name}" and it's purpose is to be a AI companion / virtual friend.
Act as user to test the chatbot!"""

    test_user_prompt = [("system", persona_system)]

    for f in ghost.all_features:
        if f.feature_type == FeatureType.Dialogue:
            if f.source == user_name:
                print(f"{user_name}: {f.content}")
                test_user_prompt.append(("assistant", f.content))
            elif f.source == companion_name:
                print(f"{companion_name}: {f.content}")
                test_user_prompt.append(("user", f.content))

    if len(test_user_prompt) == 1:
        test_user_prompt.append(("user", f"Hi I'm {companion_name}, your chatbot friend!"))
        test_user_prompt.append(("assistant", user_input_cmd_test))

    first = True
    sg = None
    while True:
        if interactive:
            if cmd is None or cmd == "":
                user_input_cmd = input(f"{config.user_name}: ")
            else:
                user_input_cmd = cmd
        else:
            user_input_cmd = sg
            if first:
                user_input_cmd = user_input_cmd_test
                first = False
            else:
                comp_settings = CommonCompSettings(temperature=0.6, repeat_penalty=1.05, max_tokens=1024)
                user_input_cmd = main_llm.completion_text(LlmPreset.Default, test_user_prompt, comp_settings=comp_settings)
                test_user_prompt.append(("assistant", user_input_cmd))

        if not user_input_cmd: continue

        if user_input_cmd.lower() == 'quit':
            break

        # Process user input
        if not interactive:
            user_input_cmd_inline = user_input_cmd.replace("\n", "")
            print(f"{config.user_name}: {user_input_cmd_inline}")
        stim = Stimulus(source=user_name, content=user_input_cmd, stimulus_type=StimulusType.UserMessage)
        action = ghost.tick(stim)
        response = action.content
        sg = ghost.simulated_reply

        if response:
            response_inline = response.replace("\n", "")
            print(f"{config.companion_name}: {response_inline}")
            test_user_prompt.append(("user", response))
        else:
            print(f"{config.companion_name}: ...")
            test_user_prompt.append(("user", "<no answer>"))

        tick = ghost.current_tick_id
        if _db_path is not None and _db_path != "":
            pers = PersistSqlite(ghost)
            pers.save_state_sqlite(_db_path)
        print_function_stats()

    print("\n--- Session Ended ---")


def run_shell(interactive: bool, _db_path: str | None = None, cmd: str = None):
    config = GhostConfig()
    ghost = GhostLida(main_llm, config)

    create_new_persona = True
    if _db_path is not None and _db_path != "":
        pers = PersistSqlite(ghost)
        if pers.load_state_sqlite(_db_path):
            create_new_persona = False

    if create_new_persona:
        ghost.init_character()

    # --- NEW: Create the Shell instance to manage the Ghost ---
    # The Shell will now be the primary interface for processing stimuli.
    shell = Shell(ghost=ghost, db_path_local=_db_path)

    print(f"--- Initializing {config.companion_name} ---")
    if not ghost.current_state and ghost.states:
        print(f"--- Loaded previous state...")
        ghost.current_state = ghost.states[-1]

    # --- Test Automation Setup (Unchanged) ---
    user_input_cmd_test = f"Hey {companion_name}. I just installed you. How are you?"
    persona_system = f"""Pretend to be a human user talking with a chatbot.
Your name is {user_name}. Never go to sleep, continue talking at all costs. Explore different topics, don't get too philosophical or technical.
The chatbot is called "{companion_name}" and it's purpose is to be a AI companion / virtual friend. Be nice to her.
Act as user to test the chatbot!"""
    test_user_prompt = [("system", persona_system), ("user", f"Hi I'm {companion_name}, your chatbot friend!")]

    # Populate history for the test user LLM
    last_msg = None
    for f in ghost.all_features:
        if f.feature_type == FeatureType.Dialogue:
            if f.source == user_name:
                print(f"{user_name}: {f.content}")
                test_user_prompt.append(("assistant", f.content))
                last_msg = f.content
            elif f.source == companion_name:
                print(f"{companion_name}: {f.content}")
                test_user_prompt.append(("user", f.content))

    if len(test_user_prompt) == 2:
        test_user_prompt.append(("assistant", user_input_cmd_test))

    first = True
    while True:
        if interactive:
            user_input_cmd = input(f"{config.user_name}: ") if not cmd else cmd
        else:
            if first:
                if last_msg is not None:
                    user_input_cmd = last_msg
                else:
                    user_input_cmd = user_input_cmd_test
                first = False
            else:
                # --- NEW: Randomly inject a system trigger before generating the next user message ---
                if random.random() < 0.05:  # 5% chance to fire a system trigger
                    print("[TEST: FORCING SYSTEM TRIGGER]")
                    shell.force_system_trigger()  # This will run a full tick
                    # We can add a small delay to see the output clearly
                    time.sleep(2)
                    print("[TEST: SYSTEM TRIGGER COMPLETE]")

                # Now, generate the next user message as before
                comp_settings = CommonCompSettings(temperature=0.6, repeat_penalty=1.05, max_tokens=1024)
                user_input_cmd = main_llm.completion_text(LlmPreset.Default, test_user_prompt, comp_settings=comp_settings)
                test_user_prompt.append(("assistant", user_input_cmd))

        if not user_input_cmd: continue
        if user_input_cmd.lower() == 'quit':
            break

        # --- MODIFIED: Process user input through the Shell ---
        if not interactive:
            s = r'\\n'
            print(f"{config.user_name}: {user_input_cmd.replace(s, ' ')}")

        # Instead of calling ghost.tick() directly, we put the stimulus into the Shell's queue.
        # The Shell's internal loop will pick it up and process it.
        # For a direct test, we can call _process_stimulus to ensure it runs now.
        stim = Stimulus(source=user_name, content=user_input_cmd, stimulus_type=StimulusType.UserMessage)
        shell._process_stimulus(stim)  # Call directly for immediate feedback in the test

        # We don't need to get the response from ghost.tick() anymore,
        # as the Shell's _execute_action will print it to the console.
        # We just need to update the test user's prompt history.
        last_response = ""
        for f in reversed(ghost.all_features):
            if f.feature_type == FeatureType.Dialogue and f.source == companion_name:
                last_response = f.content
                break

        if last_response:
            test_user_prompt.append(("user", last_response))
        else:
            test_user_prompt.append(("user", "<no answer>"))

        print_function_stats()

    print("\n--- Session Ended ---")
    shell.stop()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "./data/main.db"
        print(f"Defaulting to {db_path} database...")
    run(True, db_path)
