# private-machine

private-machine (pm) is a cognitive runtime for **persistent characters**.

A character in pm is not a system prompt and a chat log. It is inspectable internal state: a persistent mental-state
vector, episodic memory with emotional consequences, needs and goals the character can defer or refuse, parallel
single-concern appraisers, and an attention mechanism that decides what occupies its workspace this cycle. Every
spoken reply is downstream of that state.

The design goal is **auditability**. Behaviour is determined by records you can open, diff, and tune, rather than by
prompt text you hope the model still attends to. A character is therefore debuggable like a program, and it can stay
consistent over weeks of interaction instead of over one context window. Both the [review desk](dev/review/README.md)
and the [memory inspector](#inspect-a-memory-without-the-character) read this state without running a cycle.

Nothing in the runtime is specific to any one use. Build a companion you talk to daily, a non-player character that
remembers you 200 hours later, a character with a body that plays Minecraft, or a persona behind a game's dialogue
line. Character creation and [character-card import](docs/subsystems/character-import.md) are built in, including an
existing chatlog as starting history. Personas, model selection, and prompt templates are yours; the runtime is
neutral about what the character is for.

Two design decisions are load-bearing and worth knowing before you read further:

- **Replies are not regenerated.** A character said what it said. It can notice the effect and revise its beliefs, but
  a turn is never rolled back. State advances monotonically.
- **Cognition is separated from presentation.** Thinking happens in cycles that run whether or not anyone is
  speaking. The character keeps thinking between your messages.

## How one cycle works

A cycle takes pending stimuli and produces at most one action per stimulus group.

1. **Appraisal.** [Appraisals](workflows/main_cognition/affect.py) read the stimuli against current state and move the
   [mental-state vector](pm/records/mental_state.py).
2. **Codelets.** The registered [codelets](workflows/main_cognition/codelets.yaml) that match the stimulus run in
   parallel. Each holds one concern at full strength and offers its own recommendation. None of them balances itself.
3. **Workspace.** Those recommendations, plus retrieved episodes, current worries, and the
   [situational model](docs/concepts/architecture.md), land as shards in the [global workspace](workflows/main_cognition/phases/attention.py).
4. **Attention.** Shards compete. Selection stops at a capacity limit, so what the character is aware of right now is
   a chosen subset, not everything relevant. Conflicts can survive selection.
5. **Goals and decision.** [Goals](workflows/main_cognition/phases/goals.py) are formed, reprioritized, deferred, or
   refused against needs, then one action per stimulus group is selected.
6. **Speech.** [Public speech](workflows/public_speech/) renders the action using the character's
   [private ontology](workflows/public_speech/ontology.py), so its words for feelings and relationships track its
   actual recorded state instead of drifting.

Between messages, the [inner-thought loop](workflows/inner_thought/) runs the same workspace with no new stimulus, and
[focus management](workflows/focus_management/) can recall an episode and propose a new subject of its own accord.

## Features

Everything below is state with a record type behind it, not a prompt section.

- **Emotions:** Joy, jealousy, boredom, tenderness, and trust live in a persistent [mental-state vector](pm/records/mental_state.py).
  [Appraisals](workflows/main_cognition/affect.py) change that state, and those changes shape attention and expression.

- **Mental processes:** Small processes called [codelets](workflows/main_cognition/codelets.yaml) recall old episodes, notice disrespect, imagine another person's perspective, or find something funny.
  Each offers its own take on the situation.

- **Attention:** A worry, a memory, and something you just said can compete for attention in the [global workspace](workflows/main_cognition/phases/attention.py).
  What wins helps shape the next decision.

- **Inner thoughts:** A separate [private thought loop](workflows/inner_thought/) gives the character time to think without a new message.
  It can also [recall past events and propose a new focus](workflows/focus_management/).

- **Memory:** The character can [recall experiences](pm/memory/retrieval.py) and their emotional consequences.
  [Memory consolidation](pm/memory/reconsolidation/) builds episodes and broader narratives from the original events.

- **Personality, needs, goals:** Stable traits sit alongside changing needs for connection, autonomy, and creative expression.
  The character can [form, reprioritize, defer, or refuse goals](workflows/main_cognition/phases/goals.py).

- **Expectations:** It [records what it expects](workflows/main_cognition/expectations.py) to happen after an action.
  Later events can fulfill those expectations or surprise it, with consequences for its emotional state.

- **Metacognition:** The character can [inspect parts of its own code and records](pm/introspection/) to revise its beliefs about itself.
  Experimental [reflection](workflows/reflection/) tests changes to attention and subjective beliefs against later choices.

- **Consistent Vocabulary** A [private ontology](workflows/public_speech/ontology.py) connects words for emotions, relationships, and mental actions to recorded state.
  Those links give the character consistent terms for its own experience.

- **Body in Minecraft:** The optional [Minecraft body](workflows/minecraft_body/) lets the character move, gather, mine, and build.
  Larger [projects](workflows/minecraft_project/) keep plans and progress across many actions.

- **Tools:** Through [MCP tools](workflows/tool_call/), the character can carry out tasks, remember the outcome, and tell you about it in its own voice.

- **Inspection:** The [review desk](dev/review/README.md) shows conversations alongside private thoughts, appraisals, codelet output, and the memories behind later decisions.
  The [memory inspector](#inspect-a-memory-without-the-character) shows the stored records, the workspace shards of each cycle, and the text each record adds to a prompt.

You choose the [character](docs/subsystems/character-import.md) and the [models](docs/models.md).
With local model servers, the character's memory and conversation stay on your machine.

## Requirements

**As of the current revision.** The runtime is a client of two OpenAI-compatible endpoints and hosts nothing itself.

| Need | Minimum |
| --- | --- |
| Chat endpoint | OpenAI-compatible `/v1`, at least 98k context |
| Chat model | Qwen 3.6 35B3A. Qwen 3.8 27B and Next work better |
| Embedding endpoint | OpenAI-compatible, at least 2048 input tokens. EmbeddingGemma 300M is fine |
| Python | 3.12, with [uv](https://docs.astral.sh/uv/) and Git |
| Windows only | Git Bash on `PATH` for tool workflows |
| Optional: Minecraft body | Java 1.19.4 server, Node.js 20+, RCON access, and a running [`pm-minecraft`](https://github.com/flamingrickpat/pm-minecraft) MCP service |
| Optional: other tools | Their own running MCP servers |

You also need patience. See [Status and performance](#status-and-performance).

## Status and performance

This is **alpha**. Be accurate about what that costs you:

- **It is slow, and the reason is architectural, not a bug.** Twelve registered codelets run per cycle. Eight execute
  as two-call chains, four as tool-using [pm-coder](https://github.com/flamingrickpat/pm-coder) sessions with memory
  access, and the codelet phase sits on top of the appraisal, attention, goal, decision, and speech phases. Context
  assembly fills up to `context_input_fraction` of the window (0.8 by default), so per-cycle token volume, not model
  price, is the dominant cost. On two 35B-A3B instances delivering about 400 tok/s combined, a single character is at
  the edge of usable.
- **Per-cycle token cost is currently unmeasured.** The [inspector](#inspect-a-memory-without-the-character) reports
  the prompt cost of every shard and flags content several shards pay for twice. Making that number the headline of a
  cycle is a prerequisite for the performance work below, not a side effect of it.
- **Model output still needs repair sometimes.** Character quality depends heavily on your model and prompts.
- **Long-run consistency is the open research problem, not latency.** Several characters running for weeks against
  each other is the failure mode to expect: retrieval contamination, persona bleed, self-contradiction, and a
  character acting on information it was never present for. Latency is an engineering task. This is not.
- **Multiple characters means multiple processes today.** Each gets its own character home, `--web-port`, MCP
  configuration, and (for simultaneous bodies) distinct MCP, body, and viewer ports. There is no world-level arbiter
  yet; see [Roadmap](#roadmap).

The tradeoff is deliberate. Quality of the internal model came before throughput, and the throughput work is now
constrained by measurements rather than by taste.

## Roadmap

Roughly in the order it is planned. Items below the line are not implemented.

- **Knobs for the cognitive simulation.** The codelet registry already takes `execution`, `families`, and
  `stimulus_kinds` per codelet, and `cognitive_control` bounds deliberate waits. Planned: per-role model assignment
  (small models for narrow appraisers, a strong one for attention and speech), magnitude gating so a codelet only
  wakes when the relevant state variable actually crossed a threshold, and a step cap on agent-mode codelets.
- **Prefix caching.** The persona, trait, and stable-memory prefix is nearly identical across a character's calls.
  Cached input is several times cheaper, and this is expected to be the largest single cost lever available.
- **Async tier split.** Separate a reflex path that answers quickly with no memory writes, a conversational path that
  produces the spoken reply, an asynchronous cognition path for the full codelet swarm, and an offline consolidation
  pass. Only the reflex path needs to meet a latency budget. This is what makes real-time response affordable.
- **Thin codelet contexts.** Give each codelet a distinct few-k-token slice of relevant records instead of a filled
  workspace.

---

- **OpenAI-compatible server endpoint.** pm currently *consumes* such endpoints. The plan is also to *serve* one, so
  an existing mod or client that already points at a chat-completions URL can swap pm in and get long-horizon memory
  and persona state for free.

  This is harder than it looks, and the difficulty is the API contract rather than the plumbing. `messages` alone is
  not enough for a persistent character: the runtime needs **who is speaking** (a stable entity id, or every
  interlocutor collapses into one relationship), **world and scene state**, and **time passing** with no message. The
  host's own system prompt will also fight the character's persona, so the endpoint needs an explicit precedence mode
  rather than a concatenation. Design notes land under `docs/subsystems/` as they are settled.

- **Multi-character world arbitration.** A scheduler that decides which character acts when two want the same thing,
  and resolves contention over shared world state. This is the prerequisite for factions rather than for
  parallel-and-unaware characters.

- **Belief gating in retrieval.** A character must be unable to recall what it was not present for. Requires an
  explicit per-entity model of what others know, applied as a retrieval filter. Without this, everything downstream
  reads as fake to anyone paying attention.

- **A long-horizon interaction benchmark.** This builds on an existing base rather than starting from nothing:
  [A18B cognitive believability evaluation](docs/subsystems/cognitive-evaluation.md) runs complete conversations over
  a versioned 16-scenario repertoire, freezes its cases, criteria, and thresholds per run, and calibrates its
  reviewers against planted defects. Scenario authoring is in
  [cognitive-evaluation-authoring](docs/subsystems/cognitive-evaluation-authoring.md).

  What A18B does not do is time. The planned benchmark adds multi-month simulated interaction and reports **drift
  curves rather than a single score**, plus negative metrics that belief-oriented suites omit: confabulated memories,
  contradictions, persona bleed across characters, and correct *refusal* to know something the character was not
  present for. Scoring recall alone rewards retrieval and ignores invented pasts, which is the failure that actually
  breaks a long-running character. Cost and token volume per interaction go on the same chart. Competing open
  projects run at pinned versions with their defaults documented, and transcripts plus judge prompts ship with the
  results so any number can be checked. Until those numbers exist, statements about how pm compares to other projects
  are opinion, and this README will not make them.

## Related projects

[Project AIRI](https://github.com/moeru-ai/airi) and the Neuro-sama-shaped projects listed in
[awesome-ai-vtubers](https://github.com/proj-airi/awesome-ai-vtubers) optimize presentation: Live2D and VRM avatars,
voice and ASR latency, and game integration. AIRI is far more widely deployed than this project and already does
realtime voice chat, Minecraft, and Factorio. pm has none of that and does not compete with it.

pm works on the other axis: persistent state, and whether a character is still itself after a hundred hours.

The two are complementary, and the axis they differ on is one worth measuring rather than asserting. That is what the
benchmark above is for.

## Start locally

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Git, and OpenAI-compatible chat and embedding servers.
On Windows, tool workflows also need Git Bash on `PATH`.

```powershell
uv sync --locked
```

The [example configuration](config.example.yaml) uses chat on port **8080** and embeddings on **8081**.
Set the model aliases and embedding dimensions to match your servers.
Select your configuration explicitly with `--config`.
Your character data, prompt logs, and model cache stay on your machine when both endpoints are local.

Choose [Sunny or Rowan](docs/workflows/example-characters.md).
Both include a card, personality values, and an optional synthetic chatlog.
For a new Sunny with empty history:

```powershell
uv run --locked python -m pm.app.character_cli init --home .runtime/characters/sunny --companion-name Sunny --user-name Alex --card examples/characters/sunny/card.md --personality examples/characters/sunny/personality.json
uv run --locked python main.py --home .runtime/characters/sunny --config config.example.yaml
```

Open the web address printed by the launcher.
The character home holds its accepted memory in `ghost.pickle` and saves later changes there.
An older home that holds only `ghost.json` still loads.
Use separate homes for separate characters.

## Bring your own character

Supply a character card to the same `init` command.
You can also supply personality values and an optional chatlog with `name,message,timestamp` columns.
The [character import guide](docs/subsystems/character-import.md) explains the format and commands.
An empty history works without an import.

Spamuel is the synthetic development persona. Bare startup still selects it; use `--home` for your own character.

## Inspect a memory without the character

The memory inspector serves one character checkpoint in a browser. It reads one `ghost.pickle` and starts no
session. It runs no cycle, calls no model, and saves nothing. Use it to look at stored records and the workspace
shards of each cycle. It also shows the text that each record adds to a prompt.

```powershell
uv run --locked python main.py --home .runtime/characters/sunny --config config.example.yaml --inspector
```

`--config` has no effect in this mode, because the inspector needs no model server.
Open the address that the launcher prints, for example `http://127.0.0.1:8771`.
The launcher prints the record and cycle counts, then waits. Press `Ctrl-C` to stop.
To read one checkpoint file instead of a whole home, pass `--checkpoint path\to\ghost.pickle`.
If the port is busy, pass `--inspector-port` with a free port.

The pages build on each other. Every id in a table opens the page of that id:

- **Cycles** lists each accepted cycle with the prompt cost it recorded. One click opens that cycle.
- **A cycle** shows its shards, its situational model, its new stimuli, and its repeated content.
- **A shard** shows the rendered block, the token cost of every line, and the token counts the cycle stored.
- **The story prefix** shows the whole text that the cycle put in front of the model.
- **Records** and **Types** list every stored record. Each one links to its story line, its relations, and the cycles that held it.
- **Clusters** shows the episodic hierarchy: member events, child clusters, and parents.
- **Prompts** shows the templates that these pages render with.

The inspector uses the same code that builds the story lines and the shard blocks of a real cycle. The text in
the browser is the text the model receives. It reports the cost of the labels and headers around those lines, and
it marks content that several shards pay for twice. A checkpoint from an older revision can miss a record or a
field. The pages mark those gaps. They do not hide them. The inspector loads no embeddings, so it shows no
similarity results.

## Connect a character to Minecraft

Keep your existing Minecraft server running.
The character uses a separate `pm-minecraft` MCP service for its Minecraft body.
The current MCP requires Minecraft Java 1.19.4, Node.js 20 or newer, and RCON access to the same server.
These examples use Sunny's existing character home and an installed MCP checkout at `C:/source/pm/pm-minecraft`.
Replace the paths, player name, and ports for your character and server.
If the MCP service already runs for this character, continue at step 3.

You can download it here: https://github.com/flamingrickpat/pm-minecraft

If you're interested in real survival gameplay, add this babymode-mod for easier combat: https://github.com/flamingrickpat/pm-agentic-babymode

1. In the MCP checkout, copy your working server configuration once:

   ```powershell
   Set-Location C:\source\pm\pm-minecraft
   Copy-Item .env_real .env.sunny
   ```

   Edit these entries in `.env.sunny`:

   ```dotenv
   MINECRAFT_HOST=127.0.0.1
   MINECRAFT_PORT=12345
   MINECRAFT_VERSION=1.19.4
   MINECRAFT_PLAYER=Sunny
   MINECRAFT_RCON_HOST=127.0.0.1
   MINECRAFT_RCON_PORT=25575
   MINECRAFT_RCON_PASSWORD=your-server-rcon-password
   MCMCP_AGENT_HOME=C:/source/pm/private-machine/.runtime/characters/sunny/minecraft
   MCP_HOST=127.0.0.1
   MCP_PORT=6767
   MCMCP_BODY_PORT=6768
   MCMCP_VIEWER_PORT=6769
   ```

   Keep the working `NODE_EXE`, `MCMCP_BODY_ENTRYPOINT`, and `BROWSER_EXECUTABLE_PATH` entries from the copied configuration.
   `MINECRAFT_PLAYER` names the Minecraft body. `MCMCP_AGENT_HOME` holds its files separately from pm's character memory.

2. In that terminal, start only the MCP service:

   ```powershell
   .venv/Scripts/python.exe main.py .env.sunny
   ```

   Keep this terminal open. The MCP creates missing character directories on its first start.

3. In a second PowerShell terminal, add the Minecraft body to your model configuration and start pm:

   ```yaml
   minecraft:
     mcp_url: http://127.0.0.1:6767/mcp
     request_timeout_seconds: 660
     probe_timeout_seconds: 5
   ```

   Put the `minecraft` section in the file that you give with `--config`.
   `mcp_url` uses the MCP HTTP port, not the game port.
   pm asks the endpoint once at startup and exits when nothing answers.
   Without the section, the character has no body: no world poll, no live inventory read, and no Minecraft route.
   `MINECRAFT_MCP_URL` remains a test override and is not needed for normal operation.
   The `PM_MINECRAFT_HOST` and `PM_MINECRAFT_PORT` variables supply the game address for connection checks.
   Repeat these environment assignments in each new terminal before you start pm.

4. Open the web address printed by pm.
   Ask the character: "What is in your Minecraft inventory?" or "Where are you in Minecraft?"
   These requests use a live read. A successful read confirms more than an open server port.
   Then ask for a Minecraft goal through the same chat.

Use a separate player name, MCP configuration, and `MCMCP_AGENT_HOME` for each character.
For simultaneous characters, also assign distinct MCP, body, and viewer ports.
Give each pm process a distinct `--web-port`.
Point each pm process at its character's MCP endpoint.
Stopping pm leaves the MCP service and Minecraft server running.
Read the [Minecraft workflow guide](docs/workflows/minecraft.md) for managed runs and cancellation.

## More information

- [Example characters and optional histories](docs/workflows/example-characters.md)
- [Local and remote model configuration](docs/models.md)
- [Operator console](docs/workflows/console.md)
- [Engineering manual](docs/index.md)
- [Development and tests](docs/workflows/development.md)

Installations use the revisions in `uv.lock`. Dependency repositories must be accessible during installation.

## Disclaimer

WITHOUT LIMITING THE FOREGOING, THE PROVIDER SPECIFICALLY DISCLAIMS ANY LIABILITY OR RESPONSIBILITY FOR ANY OF THE FOLLOWING, ARISING AS A RESULT OF OR RELATED TO THE USE OF THE SOFTWARE: 
1. EMOTIONAL ATTACHMENTS, INCLUDING BUT NOT LIMITED TO, DEVELOPING ROMANTIC FEELINGS TOWARDS THE Al COMPANION, RESULTING IN SITUATIONS RANGING FROM MILD INFATUATION TO DEEP EMOTIONAL DEPENDENCE. 
2. DECISIONS MADE OR ACTIONS TAKEN BASED ON ADVICE, SUGGESTIONS, OR RECOMMENDATIONS BY THE AI, INCLUDING BUT NOT LIMITED TO, LIFESTYLE CHANGES, RELATIONSHIP DECISIONS, CAREER MOVES, OR FINANCIAL INVESTMENTS. 
3. UNEXPECTED BEHAVIOR FROM THE AI, INCLUDING BUT NOT LIMITED TO, UNINTENDED INTERACTIONS WITH HOME APPLIANCES, ONLINE PURCHASES MADE WITHOUT CONSENT, OR ANY FORM OF DIGITAL MISCHIEF RESULTING IN INCONVENIENCE OR HARM. 
4. PSYCHOLOGICAL EFFECTS, INCLUDING BUT NOT LIMITED TO, FEELINGS OF ISOLATION FROM HUMAN CONTACT, OVER-RELIANCE ON THE Al FOR SOCIAL INTERACTION, OR ANY FORM OF MENTAL HEALTH ISSUES ARISING FROM THE USE OF THE SOFTWARE. 
5. PHYSICAL INJURIES OR DAMAGE TO PROPERTY RESULTING FROM ACTIONS TAKEN BASED ON THE SOFTWARE'S BEHAVIOR, SUGGESTIONS, OR RECOMMENDATIONS. 