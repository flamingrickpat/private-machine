# private-machine

private-machine (pm) is a highly experimental **AI companion** harness with persistent memory, internal thought, and a 
configurable persona.

You can create a new character or import a character card and an old chatlog.

The ultimate goal is to make an extremely **immersive** and **functionally self-aware** AI companion you can run with local models
and interact via text or in Minecraft. 

I don't claim to have invented machine-consciousness, I'm not that crazy (yet). It should just be a really immersive 
simulation where the illusion is never broken. It goes against the design to regenerate replies, and you can't do it.

Due to the crazy amount of agents and background services this project is **SLOW AF**. Still working on that, quality 
first. Even with 2 Qwen 3.6 35B3A instances with combined 400tps, it's barely useable.

### Requirements

- OpenAI compatible endpoint for the LLM
- Qwen 3.6 35B3A is the absolute minimum, Qwen 3.8 27B or Next work better
- At least 98k context
- OpenAI compatible endpoint for the embedding model
- At least 2048 input tokens
- EmbeddingGemma 300M is fine
- Patience

## Features

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

- **Metacognition** The character can [inspect parts of its own code and records](pm/introspection/) to revise its beliefs about itself.
  Experimental [reflection](workflows/reflection/) tests changes to attention and subjective beliefs against later choices.

- **Consistent Vocabulary** A [private ontology](workflows/public_speech/ontology.py) connects words for emotions, relationships, and mental actions to recorded state.
  Those links give the character consistent terms for its own experience.

- **Body in Minecraft:** The optional [Minecraft body](workflows/minecraft_body/) lets the character move, gather, mine, and build.
  Larger [projects](workflows/minecraft_project/) keep plans and progress across many actions.

- **Tools:** Through [MCP tools](workflows/tool_call/), the character can carry out tasks, remember the outcome, and tell you about it in its own voice.

- **Inspection:** The [review desk](dev/review/README.md) shows conversations alongside private thoughts, appraisals, codelet output, and the memories behind later decisions.

You choose the [character](docs/subsystems/character-import.md) and the [models](docs/models.md).
With local model servers, the character's memory and conversation stay on your machine.

The project is in alpha. Model output can still need repair, and companion quality depends on your model and prompts.
Minecraft and other MCP tools are optional. They require their own running services.

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
The character home holds its accepted memory in `ghost.json` and saves later changes there.
Use separate homes for separate characters.

## Bring your own character

Supply a character card to the same `init` command.
You can also supply personality values and an optional chatlog with `name,message,timestamp` columns.
The [character import guide](docs/subsystems/character-import.md) explains the format and commands.
An empty history works without an import.

Spamuel is the synthetic development persona. Bare startup still selects it; use `--home` for your own character.

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