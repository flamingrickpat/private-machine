# private-machine

*I'll be your machinery*

This project is an attempt at making a local AI companion framework that offers
- Model agnostic architecture (I only tested it with Hermes-3-Llama-3.1-8B)
- Enhanced agency over traditional RP prompting by employing various subsystems
- Enhanced memory with topical and temporal clusters
- Combination of tool calls and natural dialogue
- Extendability for all kinds of sensations and actions

## Installation

*New start, restart, empty pages*

1. Install python 3.11
2. Install requirements
3. Install torch with CUDA support
4. Install llama-cpp-python from whl or compile it yourself (https://github.com/abetlen/llama-cpp-python)
5. Get pgvector docker image, start it with default port and password as password
6. Copy the config.yaml.example to config.yaml and edit it to your preference
    - Change the model paths
    - Set a custom database name
    - Edit the character card
    - Switch to your preferred model (Hermes-3-Llama-3.1-8B.Q8_0.gguf seems to do the job well enough)
7. ```(venv) python -m spacy download en_core_web_trf``` (once)
8. ```(venv) python run.py "hello, how are you?"```

## Telegram

If you don't want to use console, there is also a Telegram Client. Just copy .env.example to .env and set your tokens.
Start the service like this:

```(venv) python .\pm\main_interface.py```

The messages will be relayed from and to Telegram.

## Implementation
*Ich bin meines Glückes Schmied*

I totally wrote all this myself.

### Genneral Architecture

The Sensation-Action Architecture in this project operates as a cognitive loop where sensory inputs (impulses) trigger an evaluation and response mechanism within the AI system. The architecture follows a structured pipeline:

- Impulse Registration: Incoming stimuli, such as user input or system events, are captured as impulses.
- Sensation Evaluation: These impulses are analyzed, including their emotional and contextual relevance, by subsystems like the Emotion Subsystem, which adjusts the AI's emotional state accordingly.
- Action Selection: Based on the analyzed input and the AI’s internal state, the Action Selection Subsystem determines the appropriate course of action, such as replying to the user, performing a tool call, or entering a sleep state.
- Action Planning and Execution: Once an action is chosen, the system refines and executes it through specialized subsystems. For instance, the Verbal Communication Subsystem converts internal states into coherent user dialogue.
- Memory and Learning: The system optimizes memory over time, ensuring responses remain relevant and efficient.
This modular and recursive framework enables dynamic interaction, emotional adaptation, and flexible decision-making within the AI companion.

### Clustering & Information Recollection
*Tell me what to find...*

Beyond simple retrieval, private-machine organizes past interactions into meaningful clusters, allowing for better context awareness and long-term memory.
- Temporal & Thematic Clustering – Groups related messages based on topic and time, avoiding context drift.
- Dynamic Context Management – Prioritizes relevant information without blindly dumping full logs into prompts.
- Fact Extraction – Distills conversations into key takeaways, improving recall efficiency.
#### Why not just use naive RAG? 
- Standard RAG retrieval is often keyword-based and context-blind. This system structures data for better synthesis and context weighting.

### Emotion Analysis
*I've got some news for you, fembots have feelings too!*

Rather than just detecting sentiment, private-machine evaluates emotional weight and integrates it into responses dynamically.
- Impact Scoring – Rates how emotionally significant an interaction is.
- First-Person Emotional Thought Generation – Generates internal reflections based on prior interactions.
- Contextual Emotional Memory – Remembers user sentiment over time, rather than reacting in isolation.
#### Why not just use sentiment analysis?
- Most LLM-based sentiment analysis treats emotions as static labels. Here, emotion feeds into a broader cognitive loop.

### Tree of Thoughts (ToT) with Anti-Pattern Exploration
*Freeing my trapped soul*

Unlike basic idea generation, private-machine structures thoughts hierarchically and prioritizes unconventional conclusions.
- Multi-Step Thought Chains – Generates follow-up thoughts, ensuring deeper reasoning.
- Anti-Pattern Selection – Prefers less conventional responses, avoiding repetitive or cookie-cutter LLM output.
- Evaluated Divergence – Filters ideas by realism, novelty, and emotional resonance.
#### Why not just use random brainstorming?
- LLMs tend to favor safe, expected answers. This method systematically generates and selects ideas that break patterns.

## Disclaimer

WITHOUT LIMITING THE FOREGOING, THE PROVIDER SPECIFICALLY DISCLAIMS ANY LIABILITY OR RESPONSIBILITY FOR ANY OF THE FOLLOWING, ARISING AS A RESULT OF OR RELATED TO THE USE OF THE SOFTWARE: 
1. EMOTIONAL ATTACHMENTS, INCLUDING BUT NOT LIMITED TO, DEVELOPING ROMANTIC FEELINGS TOWARDS THE Al COMPANION, RESULTING IN SITUATIONS RANGING FROM MILD INFATUATION TO DEEP EMOTIONAL DEPENDENCE. 
2. DECISIONS MADE OR ACTIONS TAKEN BASED ON ADVICE, SUGGESTIONS, OR RECOMMENDATIONS BY THE AI, INCLUDING BUT NOT LIMITED TO, LIFESTYLE CHANGES, RELATIONSHIP DECISIONS, CAREER MOVES, OR FINANCIAL INVESTMENTS. 
3. UNEXPECTED BEHAVIOR FROM THE AI, INCLUDING BUT NOT LIMITED TO, UNINTENDED INTERACTIONS WITH HOME APPLIANCES, ONLINE PURCHASES MADE WITHOUT CONSENT, OR ANY FORM OF DIGITAL MISCHIEF RESULTING IN INCONVENIENCE OR HARM. 
4. PSYCHOLOGICAL EFFECTS, INCLUDING BUT NOT LIMITED TO, FEELINGS OF ISOLATION FROM HUMAN CONTACT, OVER-RELIANCE ON THE Al FOR SOCIAL INTERACTION, OR ANY FORM OF MENTAL HEALTH ISSUES ARISING FROM THE USE OF THE SOFTWARE. 
5. PHYSICAL INJURIES OR DAMAGE TO PROPERTY RESULTING FROM ACTIONS TAKEN BASED ON THE SOFTWARE'S BEHAVIOR, SUGGESTIONS, OR RECOMMENDATIONS. 

