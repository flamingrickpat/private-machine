# private-machine

*I'll be your machinery*

I refined this ultra-schizo project again in my quest to make the perfect AI companion.

Basically everything happens in pm_lida.py, the old stuff in the pm folder is just for decoration and occasional copy-pasting now.

This is NOT production ready! Tool calling is underdeveloped, async background cognition and proactive interaction with user when idle is not implemented yet.

## Installation

*New start, restart, empty pages*

1. Install python 3.11
2. Install torch with CUDA (https://pytorch.org/get-started/locally/)
3. Install llama-cpp-python with CUDA from WHL (https://github.com/JamePeng/llama-cpp-python) or compile it yourself
4. Install requirements.txt
5. Copy the config.yaml.example to config.yaml and edit it to your preference
    - Change the model paths
    - Set a custom database name
    - Edit the character card
    - Switch to your preferred model (I use gemma-3-12b-it-q4_0.gguf)
6. ```(venv) python pm_lida.py ./my_db_path.db"```

## Streamlit

If you don't want to use console, there is also a Streamlit app.
Start the service like this:

```(venv) python streamlit run .\app.py --server.fileWatcherType none```

The messages will be relayed from and to Telegram.

## Conceptual Overview

I let Gemini write this. Emmy is the default character in config.

The core philosophy of Private-Machine is to achieve emergent, psychologically-plausible behavior rather than scripted responses. Emmy's "personality" is not a set of hardcoded rules, but the result of a continuous cognitive cycle where her internal state dynamically shapes her perception, attention, and actions.

### Guiding Principles

1.  **Psychological Plausibility over Performance:** The architecture prioritizes processes that mirror theories of human cognition (attention, memory consolidation, emotional influence) over raw speed or conversational efficiency.
2.  **State-Driven Cognition:** Every decision is grounded in Emmy's current internal state, which is defined by three core models:
    *   **Emotions:** A multi-axis model tracking valence, affection, trust, anxiety, etc.
    *   **Needs:** An AI-adapted hierarchy tracking needs like connection, relevance, and autonomy.
    *   **Cognition:** Modifiers that control focus, ego-strength, and mental effort.
3.  **Emergent Behavior:** Emmy's reactions, goals, and even her "thoughts" are generated in real-time based on the interplay between her internal state and the external world. The goal is to be responsive, not just reactive.
4.  **Continuous Learning and Growth:** Emmy is not static. A background memory consolidation system slowly processes experiences, extracts facts, identifies causal relationships, and refines her core "narratives" (her self-concept and worldview). This allows her personality and understanding to evolve over time.

---

## Technical Architecture Deep Dive

*Ich bin meines Glückes Schmied*

This project is aimed at developers interested in cognitive architectures. The following is a breakdown of the core components and the data flow within a single cognitive "tick." All classes and functions mentioned can be found in `new_arch_test.py`.

### The Cognitive Cycle (`Ghost.tick()`)

A "tick" is a discrete moment of processing, triggered by a stimulus (either external user input or an internal drive).

**1. Perception & Stimulus Appraisal (`_appraise_stimulus`)**

*I've got some news for you, fembots have feelings too!*
*   An impulse becomes a `Stimulus` knoxel.
*   The system performs a dual appraisal:
    1.  **Gut Reaction:** An LLM call (`_appraise_stimulus_and_generate_state_deltas`) evaluates the stimulus against Emmy's character card and current state to produce initial `StateDeltas` (changes to emotion, needs, cognition).
    2.  **Expectation Matching:** The stimulus is checked against active `Expectation` knoxels. Fulfilling or violating an expectation (`_appraise_stimulus_check_expectation`) generates a secondary set of `StateDeltas`.
*   These deltas are applied, and descriptive `Feeling` and `ExpectationOutcome` `Feature` knoxels are generated.

**2. Intention & Memory (`_generate_short_term_intentions`, `_gather_memories_for_attention`)**
*   Based on her new, updated state (especially pressing needs or strong emotions), Emmy generates several short-term, internal goals (`Intention` knoxels where `internal=True`).
*   Relevant long-term memories (`MemoryClusterKnoxel`), facts (`DeclarativeFactKnoxel`), and core beliefs (`Narrative` knoxels) are retrieved based on semantic similarity to the current situational context.
*   All of these—the stimulus, new feelings, new intentions, and retrieved memories—are collected into a pool of `attention_candidates`.

**3. Attention & Consciousness**
*   **Structure Building (`_build_structures_get_coalitions`):** All `attention_candidates` are semantically clustered using Agglomerative Clustering to form competing "coalitions"—coherent sets of related concepts.
*   **Attention Selection (`_simulate_attention_on_coalitions`):** This is a hybrid process.
    *   An LLM rates each coalition's relevance based on Emmy's `AttentionFocus` narrative and current emotional/cognitive state.
    *   A procedural `aux_rating` provides a minor, objective weight based on factors like recency and urgency. This now acts primarily as a tie-breaker.
*   **Conscious Broadcast (`_generate_subjective_experience`):** The contents of the winning coalition are promoted to the `conscious_workspace`. This workspace is then synthesized by an LLM into a single, first-person narrative paragraph—a `Thought` feature that represents Emmy's subjective experience for that moment.

**4. Action Selection & Deliberation (`_deliberate_and_select_action`)**
*   The `conscious_workspace` serves as the primary context for action.
*   The system runs a Monte Carlo simulation loop for potential actions (e.g., `Reply`).
*   For each simulation, it generates a potential reply and a *predicted user reaction*.
*   A "Critic" LLM (`ActionRating` schema) evaluates this simulated future, scoring it on intent fulfillment, needs fulfillment, and predicted emotional outcome.
*   The highest-scoring simulated action is selected for execution.

**5. Execution & Learning (`_execute_action`, `DynamicMemoryConsolidator`)**
*   The chosen action becomes a causal `Action` knoxel.
*   If the action involves a reply, it generates new `Expectation` knoxels about the user's likely response (`_generate_expectations_for_action`).
*   In the background, the `DynamicMemoryConsolidator` periodically processes raw `Feature` knoxels to create higher-level memories and refine core `Narrative` knoxels, enabling long-term change.

### Key Data Structures

*   **`KnoxelBase`**: The fundamental, unified unit of memory. Everything from a `Stimulus` to a `Narrative` inherits from this.
*   **`GhostState`**: A snapshot of the entire architecture's state during a single tick, including state models, workspace content, etc. It is the primary object for saving and loading.
*   **`EmotionalAxesModel`, `NeedsAxesModel`, `CognitionAxesModel`**: Pydantic models that represent Emmy's dynamic internal state.
*   **`Feature`**: A special knoxel that represents a discrete, causal event in the story (dialogue, a thought, an action). These are the primary inputs for the memory consolidation system.
*   **`Intention`**: A dual-purpose knoxel.
    *   `internal=True`: An internal goal or drive (e.g., "seek reassurance").
    *   `internal=False`: An external expectation of a future event (e.g., "I expect User to answer my question").

### Current State & Known Issues

This is an experimental project under active development.
*   **Prompt Sensitivity:** The quality of Emmy's cognition is highly dependent on the quality of the prompts and the capabilities of the underlying LLM. Small changes to prompts can have significant effects.
*   **Performance:** The architecture makes numerous sequential LLM calls per tick, making it inherently slow. This is a trade-off for psychological plausibility over speed.
*   **State Drift:** Over very long runs without restarts, emotional and cognitive axes could potentially drift into unrealistic, "stuck" states. The decay mechanisms are designed to mitigate this, but may require further tuning.
*   **Critical Event Handling:** The system currently handles all stimuli through the standard cognitive cycle. It lacks a dedicated, high-priority pathway to immediately process and integrate truly foundational, identity-altering information.

## Disclaimer

WITHOUT LIMITING THE FOREGOING, THE PROVIDER SPECIFICALLY DISCLAIMS ANY LIABILITY OR RESPONSIBILITY FOR ANY OF THE FOLLOWING, ARISING AS A RESULT OF OR RELATED TO THE USE OF THE SOFTWARE: 
1. EMOTIONAL ATTACHMENTS, INCLUDING BUT NOT LIMITED TO, DEVELOPING ROMANTIC FEELINGS TOWARDS THE Al COMPANION, RESULTING IN SITUATIONS RANGING FROM MILD INFATUATION TO DEEP EMOTIONAL DEPENDENCE. 
2. DECISIONS MADE OR ACTIONS TAKEN BASED ON ADVICE, SUGGESTIONS, OR RECOMMENDATIONS BY THE AI, INCLUDING BUT NOT LIMITED TO, LIFESTYLE CHANGES, RELATIONSHIP DECISIONS, CAREER MOVES, OR FINANCIAL INVESTMENTS. 
3. UNEXPECTED BEHAVIOR FROM THE AI, INCLUDING BUT NOT LIMITED TO, UNINTENDED INTERACTIONS WITH HOME APPLIANCES, ONLINE PURCHASES MADE WITHOUT CONSENT, OR ANY FORM OF DIGITAL MISCHIEF RESULTING IN INCONVENIENCE OR HARM. 
4. PSYCHOLOGICAL EFFECTS, INCLUDING BUT NOT LIMITED TO, FEELINGS OF ISOLATION FROM HUMAN CONTACT, OVER-RELIANCE ON THE Al FOR SOCIAL INTERACTION, OR ANY FORM OF MENTAL HEALTH ISSUES ARISING FROM THE USE OF THE SOFTWARE. 
5. PHYSICAL INJURIES OR DAMAGE TO PROPERTY RESULTING FROM ACTIONS TAKEN BASED ON THE SOFTWARE'S BEHAVIOR, SUGGESTIONS, OR RECOMMENDATIONS. 
