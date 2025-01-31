# private-machine

A local AI companion framework with structured memory, clustering, and emotion analysis.

## Installation

1. Install python 3.11
2. Install requirements
3. Install torch with CUDA support
4. Install llamacpppython from whl or compile it yourself
5. Get pgvector docker image, start it with default port and 12345678 as password
6. Install the vector extension
7. Change the settings and paths in character.py (I was too lazy to make a config)
8. (venv) python run.py "hello, how are you?"

## Implementation

I totally wrote all this myself.

### Clustering & Information Recollection
Beyond simple retrieval, private-machine organizes past interactions into meaningful clusters, allowing for better context awareness and long-term memory.
- Temporal & Thematic Clustering – Groups related messages based on topic and time, avoiding context drift.
- Dynamic Context Management – Prioritizes relevant information without blindly dumping full logs into prompts.
- Fact Extraction – Distills conversations into key takeaways, improving recall efficiency.
#### Why not just use naive RAG? 
- Standard RAG retrieval is often keyword-based and context-blind. This system structures data for better synthesis and context weighting.

### Emotion Analysis
Rather than just detecting sentiment, private-machine evaluates emotional weight and integrates it into responses dynamically.
- Impact Scoring – Rates how emotionally significant an interaction is.
- First-Person Emotional Thought Generation – Generates internal reflections based on prior interactions.
- Contextual Emotional Memory – Remembers user sentiment over time, rather than reacting in isolation.
#### Why not just use sentiment analysis?
- Most LLM-based sentiment analysis treats emotions as static labels. Here, emotion feeds into a broader cognitive loop.

### Tree of Thoughts (ToT) with Anti-Pattern Exploration
Unlike basic idea generation, private-machine structures thoughts hierarchically and prioritizes unconventional conclusions.
- Multi-Step Thought Chains – Generates follow-up thoughts, ensuring deeper reasoning.
- Anti-Pattern Selection – Prefers less conventional responses, avoiding repetitive or cookie-cutter LLM output.
- Evaluated Divergence – Filters ideas by realism, novelty, and emotional resonance.
#### Why not just use random brainstorming or ChatGPT?
- LLMs tend to favor safe, expected answers. This method systematically generates and selects ideas that break patterns.

## Disclaimer

WITHOUT LIMITING THE FOREGOING, THE PROVIDER SPECIFICALLY DISCLAIMS ANY LIABILITY OR RESPONSIBILITY FOR ANY OF THE FOLLOWING, ARISING AS A RESULT OF OR RELATED TO THE USE OF THE SOFTWARE: 
1. EMOTIONAL ATTACHMENTS, INCLUDING BUT NOT LIMITED TO, DEVELOPING ROMANTIC FEELINGS TOWARDS THE Al COMPANION, RESULTING IN SITUATIONS RANGING FROM MILD INFATUATION TO DEEP EMOTIONAL DEPENDENCE. 
2. DECISIONS MADE OR ACTIONS TAKEN BASED ON ADVICE, SUGGESTIONS, OR RECOMMENDATIONS BY THE AI, INCLUDING BUT NOT LIMITED TO, LIFESTYLE CHANGES, RELATIONSHIP DECISIONS, CAREER MOVES, OR FINANCIAL INVESTMENTS. 
3. UNEXPECTED BEHAVIOR FROM THE AI, INCLUDING BUT NOT LIMITED TO, UNINTENDED INTERACTIONS WITH HOME APPLIANCES, ONLINE PURCHASES MADE WITHOUT CONSENT, OR ANY FORM OF DIGITAL MISCHIEF RESULTING IN INCONVENIENCE OR HARM. 
4. PSYCHOLOGICAL EFFECTS, INCLUDING BUT NOT LIMITED TO, FEELINGS OF ISOLATION FROM HUMAN CONTACT, OVER-RELIANCE ON THE Al FOR SOCIAL INTERACTION, OR ANY FORM OF MENTAL HEALTH ISSUES ARISING FROM THE USE OF THE SOFTWARE. 
5. PHYSICAL INJURIES OR DAMAGE TO PROPERTY RESULTING FROM ACTIONS TAKEN BASED ON THE SOFTWARE'S BEHAVIOR, SUGGESTIONS, OR RECOMMENDATIONS. 

