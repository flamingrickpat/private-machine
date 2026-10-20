# private-machine

private-machine (pm) is an experimental AI companion with persistent memory, private thought, and a configurable character.
It connects conversation, remembered experiences, and optional tools through one runtime.
You can create a new character or import a character card and an old chatlog.

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

## How it behaves

The companion uses conversation, memory, and its internal state to choose a response or an action.
Private thought runs separately and yields inference priority to cognition and dialogue between completions.
Invalid structured answers receive validation feedback and repair turns while context space remains.

Model responses and embeddings use a local cache.
Completion response counters restart in each process, which helps reproduce a run from a backup.
Read [model connections and caching](docs/models.md) for the exact behavior and cache bypass.

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