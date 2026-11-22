# private-machine

<!-- Introduction to be written by the maintainer. -->

## How it works

Each persona runs in its own process and saves its state in a character home. Start with a character card and personality settings. You can add past dialogue or let the character build a history from the first session. The home keeps what happened, how the character interpreted it, and what it decided to do. [Character homes](docs/subsystems/character-import.md)

The design models parts of human cognition in software. It borrows attention and action selection from LIDA, and the distinction between acting and reflecting from MIDCA. PM uses language models for judgments and language. The runtime decides which records those calls can see, which decisions to accept, and which actions to execute. [LIDA](https://ccrg.cs.memphis.edu/tutorial/mindAccordingToLIDA/Brief-Account.pdf) and [MIDCA](https://ojs.aaai.org/index.php/AAAI/article/view/9886)

### Why use a cognitive loop?

Tell a language model to "pretend you're Socrates" and it can produce a convincing scene. Add the earlier chat and it may remember that you asked it to stop turning every answer into a debate. The next completion has to infer which parts of that history still matter, whether it agreed to change, and how to put that agreement into practice.

PM gives those things a place to persist. An earlier promise can become an intention. A correction can revise a belief while the original exchange remains available. A new observation can change what the character does next. The advantage is that these changes can be carried forward and inspected. How well they work still depends on the models, prompts, and tests.

### From an event to an action

Imagine a Socrates persona that promised to keep asking questions during a game of Minecraft. Then you write, "We need shelter before dark. Help me find wood, and hold the questions for now." This example shows how the pieces fit together.

1. **Intake and triage decide what needs attention.** PM records who said what and groups related events. A message from you, a game observation, and a private thought each have a source. Triage helps decide whether ordinary interaction can proceed or whether the character needs to reconsider its current direction. The latest chat line is one event in a world that keeps changing.
2. **Appraisal asks what the event means to this character.** The cycle checks open expectations against new evidence and updates recorded emotional and relationship state. A sudden change of plan can create surprise or urgency. The personality stays recognizable while the situation changes how it responds. [Cognitive pipeline](docs/subsystems/cognition.md)
3. **Retrieval and codelets bring relevant material into consideration.** Memory can supply the earlier promise to ask questions. One codelet might surface the shift in tone. Another might offer an analogy. These findings have sources and can disagree. The character can bring useful history into the exchange without giving every process the entire chat. [Experienced memory](docs/subsystems/experienced-memory.md) and [codelets](docs/subsystems/codelets.md)
4. **Attention builds a small picture of the current situation.** The cycle groups related findings and selects what fits in the global workspace. It keeps sections for the self, other people, and workflows. The request for shelter and the earlier promise can both matter. The full workspace also supports a record of how the situation seems to the character, which later phases can use. [Workspace](docs/subsystems/workspace.md)
5. **Goals and action selection turn that picture into a choice.** Socrates can defer the questions, keep the broader intention to help, and choose to look for wood. Routing selects a workflow that can act on that choice. A goal can remain active across several turns. The cycle can also record what it expects the action to achieve, so a later failure gives it something specific to revisit. [Decisions, routing, and expectations](docs/subsystems/decisions-and-routing.md)
6. **Commit, act, and check the result.** The runtime saves accepted state before releasing the action. A Minecraft worker can report what it observed, what it tried, and what remains blocked. A speech workflow uses the evidence to reply in character. Those results become new inputs to the loop. A promise to look for wood, a search that found none, and a stack in inventory are different records. [Public speech](docs/subsystems/public-publication.md)

The example could still produce a bad reply. With a role prompt alone, the main evidence to inspect is the prompt and the line it produced. Here you can check whether PM retrieved the promise, dropped the urgent request during attention, chose the wrong workflow, or received a failed run receipt. You can also see where the extra work costs time.

### Between messages

A private thought loop can revisit a memory, compare explanations, or propose a new focus. It can run between conversations and give later decisions something to consider. Ordinary conversation can use the latest accepted guidance while deeper cognition is still working. Critical input can require a new decision before the companion speaks. [Private thought](docs/subsystems/private-loop.md) and [ordinary interaction](docs/subsystems/ordinary-interaction.md)

### When the character has a body

A question about current health calls for a single live read. Gathering wood for a shelter is a longer task. The body has to observe the area, find a source, and act in the game. Cognition can hand that goal to a named Minecraft workflow, and the worker can receive updates or cancellation while it runs. An admission receipt can prompt an acknowledgment. A later reply can report the result from the worker's evidence. The same handoff can serve future workflows in other domains. [Minecraft](docs/workflows/minecraft.md) and [task replies](docs/subsystems/public-publication.md#managed-task-replies)

## Current status

PM is in alpha. Character homes, dialogue, background thought, and managed Minecraft workflows are in place. The repeatable simulations use synthetic characters such as Jesse and Spamuel to test how those pieces work together.

A recent Minecraft run admitted a task and recorded an acknowledgment action. The worker crashed at startup before Spamuel spoke. After the fix and restart, Spamuel replied that he had yet to look around. Timely acknowledgment and successful co-play still need end-to-end verification.

Speed depends on the amount of work behind a reply. The full cycle can call the model for appraisal, codelets, action selection, and speech. These calls can compete for the same endpoint. The simulation profiler records what ran, how long it took, and which prompts and records it used. The immediate performance work is to measure the delay from a message to speech or action, then cut avoidable calls and repeated prompt material. [Simulation guide](readme_simulation.md)

## Roadmap

1. **Run the regression scenarios.** Capture a baseline for chat, multi-character interaction, and Minecraft. Measure reply delay, task admission, game actions, and recovery after interruption.
2. **Make workflow handoffs reliable.** Verify prompt acknowledgment, useful progress reports, and cancellation while the companion stays available to talk. Use the same lifecycle for future workflows.
3. **Try it with a persistent persona.** Check conversation, romance, and co-play in real use. Verify that an existing character home can carry forward.
4. **Build sleep around memory maintenance.** Extend the existing consolidation workflow to summarize dialogue and private thoughts, extract durable facts, and reduce duplication in active memory.
5. **Expand codelets and mental state.** Give weakly connected needs and cognition axes observable effects. Add observation-based updates where a mechanism still needs an owner.

The [beta specification](docs/continuous-character-milestone.md) and [implementation plan](dev/work_items/beta.md) cover the longer-term work.

## Start locally

You need Python 3.12, Git, and [uv](https://docs.astral.sh/uv/). On Windows, tool workflows also need Git Bash on `PATH`. PM connects to OpenAI-compatible chat and embedding endpoints. Configure their model names and embedding settings before starting a character. [Model configuration](docs/models.md)

```powershell
uv sync --locked
Copy-Item config.example.yaml config.yaml
```

Edit your local `config.yaml`, then create Sunny with an empty history and start the web chat.

```powershell
uv run --locked python -m pm.app.character_cli init --home .runtime/characters/sunny --companion-name Sunny --user-name Alex --card examples/characters/sunny/card.md --personality examples/characters/sunny/personality.json
uv run --locked python main.py --home .runtime/characters/sunny --config config.yaml
```

Open the URL printed by the launcher. Start with the same `--home` next time to keep the character's saved state. Use your own card and optional chat history to create a different persona. [Character import](docs/subsystems/character-import.md) and [example characters](docs/workflows/example-characters.md)

### Run a simulation

The Windows launchers cover one companion in chat, two companions in chat, two companions in Minecraft, and Jesse with Spamuel in Minecraft. The Minecraft scenarios use a private Fabric world and a local `pm-minecraft` service. The launcher selects the ports and creates the player directories. [Simulation setup](readme_simulation.md)

Once `config_simulation.yaml` is set up, check the installation and launch a run from the repository root.

```powershell
.\simulation\simulation_1_jesse_spamuel.ps1 -Preflight
.\simulation\simulation_1_jesse_spamuel.ps1 -RunId baseline01
```

### Inspect a character or run

Use the simulation debugger to read the conversation, see workflows on a wall-time timeline, and open the recorded model inputs, outputs, and linked knoxels. The chat view can hide the conversation inherited from the seed checkpoint. The standalone memory inspector reads a saved character home without starting a session or calling a model.

```powershell
uv run --locked python main.py --home .runtime/characters/sunny --inspector
```

[Operator console](docs/workflows/console.md) and [engineering manual](docs/index.md)

## Research

- **Attention and learning.** The LIDA cognitive cycle and its account of how affect can influence attention and learning. [The Mind According to LIDA](https://ccrg.cs.memphis.edu/tutorial/mindAccordingToLIDA/Brief-Account.pdf) · [The Timing of the Cognitive Cycle](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0014803) · [Learning in LIDA](https://doi.org/10.1016/j.cogsys.2020.11.001)
- **Intentions and reflection.** Long-term commitments, separate levels for acting and reflecting, and traces that let an agent revisit earlier decisions. [Modeling Long-Term Intentions and Narratives](https://doi.org/10.1142/S2705078521500107) · [MIDCA](https://ojs.aaai.org/index.php/AAAI/article/view/9886) · [How Metacognitive Architectures Remember Their Own Thoughts](https://arxiv.org/abs/2503.13467) · [Generative Agents](https://arxiv.org/abs/2304.03442)
- **Internal state and inspection.** Representing and reporting internal state, and retaining evidence about what an agent did. [On the Practical Nature of Artificial Qualia](https://www.conscious-robots.com/papers/arrabales-a-paper-aiib10.pdf) · [Towards an Ontology for Robot Introspection and Metacognition](https://doi.org/10.3233/FAIA231137)

The [research notes](docs/work_items/alpha-plan/cognitive-sources.md) list the wider reading and the mechanisms adopted from each source.

## Disclaimer

WITHOUT LIMITING THE FOREGOING, THE PROVIDER SPECIFICALLY DISCLAIMS ANY LIABILITY OR RESPONSIBILITY FOR ANY OF THE FOLLOWING, ARISING AS A RESULT OF OR RELATED TO THE USE OF THE SOFTWARE: 
1. EMOTIONAL ATTACHMENTS, INCLUDING BUT NOT LIMITED TO, DEVELOPING ROMANTIC FEELINGS TOWARDS THE Al COMPANION, RESULTING IN SITUATIONS RANGING FROM MILD INFATUATION TO DEEP EMOTIONAL DEPENDENCE. 
2. DECISIONS MADE OR ACTIONS TAKEN BASED ON ADVICE, SUGGESTIONS, OR RECOMMENDATIONS BY THE AI, INCLUDING BUT NOT LIMITED TO, LIFESTYLE CHANGES, RELATIONSHIP DECISIONS, CAREER MOVES, OR FINANCIAL INVESTMENTS. 
3. UNEXPECTED BEHAVIOR FROM THE AI, INCLUDING BUT NOT LIMITED TO, UNINTENDED INTERACTIONS WITH HOME APPLIANCES, ONLINE PURCHASES MADE WITHOUT CONSENT, OR ANY FORM OF DIGITAL MISCHIEF RESULTING IN INCONVENIENCE OR HARM. 
4. PSYCHOLOGICAL EFFECTS, INCLUDING BUT NOT LIMITED TO, FEELINGS OF ISOLATION FROM HUMAN CONTACT, OVER-RELIANCE ON THE Al FOR SOCIAL INTERACTION, OR ANY FORM OF MENTAL HEALTH ISSUES ARISING FROM THE USE OF THE SOFTWARE. 
5. PHYSICAL INJURIES OR DAMAGE TO PROPERTY RESULTING FROM ACTIONS TAKEN BASED ON THE SOFTWARE'S BEHAVIOR, SUGGESTIONS, OR RECOMMENDATIONS. 