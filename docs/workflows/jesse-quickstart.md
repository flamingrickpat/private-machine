# Your own test Spamuel: local LLM and real Minecraft

## Summary

You chat as **Jesse**. The bot remains **Spamuel**.
This guide creates a separate test instance with its own memory and Minecraft character files.
Spamuel uses your local LLM and joins your existing Minecraft server.

Parent: [engineering manual](../index.md).
These commands use Windows PowerShell and your current folder locations.

## What runs where

| Part | Purpose | Address in this guide |
| --- | --- | --- |
| Local LLM | Produces Spamuel's thoughts and replies | `http://127.0.0.1:8080/v1` |
| Minecraft server | Owns your real world | `127.0.0.1:12345` |
| Minecraft MCP | Connects Spamuel to his Minecraft body | `http://127.0.0.1:6777/mcp` |
| Console | Lets you chat as Jesse | `http://127.0.0.1:8091` |

The MCP also uses ports `6778` and `6779` for its body and viewer.
These ports differ from the retained A13 service ports.
Each program needs its own PowerShell window.

## 1. Check the requirements

Your computer needs Git, `uv`, Python 3.12, Node.js 20 or newer, and Chrome or Edge.
Your existing Minecraft server needs Java, Fabric, and Agentic Babymode 0.5.0 or newer.

The current body supports **Minecraft Java 1.19.4** and uses **offline authentication**.
Your server at `C:\source\minecraft-server-real` already has the required mod and offline configuration.
It uses game port `12345` and RCON port `25575`.

If you choose another server, check its version, authentication mode, mod, and RCON configuration first.
A normal server with `online-mode=true` requires Microsoft authentication support, which this body does not expose.
Changing the player name cannot provide that authentication.

Offline servers do not authenticate player identities. Keep this setup on your trusted network.

## 2. Install the two projects

Open PowerShell in the Spamuel repository:

```powershell
Set-Location C:\source\pm\pm_next_v2
uv sync --upgrade-package pm-workflows --upgrade-package pm-coder
```

This command fetches the current `main` heads for the two Git dependencies.

Install the Minecraft body from its existing checkout:

```powershell
Set-Location C:\source\pm\pm-minecraft
.\setup.ps1
```

The setup installs its Python dependencies and builds the Node.js body.

## 3. Create your private test files

Return to the Spamuel repository:

```powershell
Set-Location C:\source\pm\pm_next_v2
New-Item -ItemType Directory -Force .runtime\jesse-test
Copy-Item config.alpha.example.yaml .runtime\jesse-test\model.yaml
Copy-Item C:\source\pm\pm-minecraft\.env_real .runtime\jesse-test\minecraft.env
```

Run the copy commands once. Later runs can overwrite your edited configuration.
The `.runtime` directory is ignored by Git, so these private files stay local.
The existing `.env_real` supplies your RCON password and installed browser path.

The original Spamuel seed already names **Jesse** as the user.
Keep `data\spamuel_agentic_canon.yaml` unchanged.
The new console instance loads that history into a separate checkpoint.
It does not inherit the retained A13 task or its player inventory.

## 4. Connect the local LLM

Start your usual local LLM server with its OpenAI-compatible API enabled.
Leave that server running.

In PowerShell, request its model list:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models
```

The response contains a model `id`. Your current local alias is normally `qwen`.

Open your model configuration:

```powershell
notepad C:\source\pm\pm_next_v2\.runtime\jesse-test\model.yaml
```

Under **both** `chat.Default` and `chat.UnitTestEvaluator`, set:

```yaml
base_url: http://127.0.0.1:8080/v1
model: qwen
api_key: ""
startup_script: ""
```

If your model list uses another ID, replace `qwen` with that exact ID.
If your LLM uses another port, change both `base_url` values.
If your LLM requires a key, set `api_key` in this private file.

Leave `startup_script` empty because you start the LLM yourself.
The local LLM must support tool calls for Minecraft workflows.

The runtime normally reads the serving context size from the local `/props` endpoint.
If your server has no `/props`, add `context_window` to both chat entries.
Use its actual configured context size, rather than the model's advertised maximum.

### Embeddings still need a connection

Embeddings are the model calls that support memory search.
The copied configuration uses OpenRouter embeddings, even though chat uses your local LLM.

In the `embedding` entry, change the cache path:

```yaml
cache_dir: ./embeddings
```

This path resolves beside `model.yaml` and keeps new cache files inside your test directory.

Before starting the console, supply your OpenRouter key in that PowerShell window:

```powershell
$env:OPENROUTER_API_KEY = Read-Host "Paste your OpenRouter key" -MaskInput
```

`-MaskInput` requires PowerShell 7.
For Windows PowerShell 5.1, use this alternative:

```powershell
$credential = Get-Credential -UserName OpenRouter -Message "Paste the key into the password field"
$env:OPENROUTER_API_KEY = $credential.GetNetworkCredential().Password
```

The key must exist in the window that starts the console. Other windows do not inherit it.

For fully local operation, configure a working local embedding endpoint instead.
Its model, dimensions, and response model must match its actual output.
The earlier local embedding service on port `8081` returned invalid vectors, so it is unsuitable as a default.

## 5. Connect the Minecraft body

Open the copied Minecraft configuration:

```powershell
notepad C:\source\pm\pm_next_v2\.runtime\jesse-test\minecraft.env
```

Replace the corresponding entries with these values:

```dotenv
MINECRAFT_HOST=127.0.0.1
MINECRAFT_PORT=12345
MINECRAFT_VERSION=1.19.4
MINECRAFT_PLAYER=Spamuel_test
MINECRAFT_RCON_HOST=127.0.0.1
MINECRAFT_RCON_PORT=25575
MCMCP_AGENT_HOME=C:/source/pm/pm_next_v2/.runtime/jesse-test/character
MCP_HOST=127.0.0.1
MCP_PORT=6777
MCMCP_BODY_HOST=127.0.0.1
MCMCP_BODY_PORT=6778
MCMCP_VIEWER_PORT=6779
MCMCP_TEST_MODE=false
```

Keep the copied `MINECRAFT_RCON_PASSWORD`, `NODE_EXE`, `MCMCP_BODY_ENTRYPOINT`, and `BROWSER_EXECUTABLE_PATH` entries.
RCON must connect to the same Minecraft server as the bot.

`Spamuel_test` is his player name in Minecraft. His companion name remains Spamuel.
This separate player name avoids reusing the retained acceptance player.

If your server runs on another computer, replace both Minecraft host entries with that computer's address.
If your server uses a whitelist, add `Spamuel_test` through your normal server administration.
Keep the bot in survival mode.

### Window A: your real Minecraft server

If the server already runs, leave it running.
Otherwise, start your existing server:

```powershell
Set-Location C:\source\minecraft-server-real
.\start-server.ps1
```

Wait for the server's `Done` message.
Use the normal server launcher. The wipe script and alpha test wrappers are unnecessary for this setup.

### Window B: Spamuel's Minecraft body

Open another PowerShell window:

```powershell
Set-Location C:\source\pm\pm-minecraft
uv run --locked python main.py C:\source\pm\pm_next_v2\.runtime\jesse-test\minecraft.env
```

Leave this window open.
Your server player list must show `Spamuel_test`.

## 6. Start Spamuel and chat as Jesse

### Window C: the console

Use the window where you supplied the embedding key.

```powershell
Set-Location C:\source\pm\pm_next_v2
$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH
$env:MINECRAFT_MCP_URL = 'http://127.0.0.1:6777/mcp'
$env:PM_MINECRAFT_HOST = '127.0.0.1'
$env:PM_MINECRAFT_PORT = '12345'
uv run --locked python -m console.server --port 8091 --root .runtime/jesse-test/console
```

The PATH entry supplies Git Bash for workflow scripts.
If your Minecraft server runs elsewhere, change the `PM_MINECRAFT_HOST` value too.
If its game port differs, change `PM_MINECRAFT_PORT` too.

Open **http://127.0.0.1:8091** in your browser.

Deploy an instance with these values:

| Console field | Value |
| --- | --- |
| Instance name | `spamuel-local-test` |
| Persona YAML | `C:/source/pm/pm_next_v2/data/spamuel_agentic_canon.yaml` |
| Model configuration | `C:/source/pm/pm_next_v2/.runtime/jesse-test/model.yaml` |
| Cognition | `real` |

Select **Deploy**, then **Start**.
Set the chat user name to **Jesse**.
Send one message:

> Hey Spamuel, Jesse here. We are trying your Minecraft body on my server today.

You use Jesse's conversation role. The Minecraft bot still appears as `Spamuel_test`.
No persona edit or Minecraft account named Jesse is necessary.

## 7. Ask for one small Minecraft task

Open **Connections** and select **Check connections**.
The checks must show your local chat endpoint and MCP address on port `6777`.
These checks discover services. They do not prove that the bot can act.

Send this first request:

> Start the managed minecraft workflow. Use only minecraft_observe. Report your current position, health, food, and inventory. Make no changes.

Select the workflow's tool event in the trace to inspect the actual tool result.
A spoken claim alone does not establish that a tool ran.
Natural-language dispatch still depends on cognition choosing the workflow.

After a successful observation, request a small action:

> Start the managed minecraft workflow. Survey your surroundings, gather one nearby log safely, and report the inventory result. Avoid other players' buildings.

Chat through the console during this test.
This setup does not automatically forward your in-game Minecraft chat to Spamuel.
The console can send instructions and ask for managed-task status.

## Stop and resume

1. Select **Stop** on the console instance.
2. If the worker does not stop, select **Kill**.
3. Press **Ctrl+C** in the Minecraft MCP window.
4. Press **Ctrl+C** in the console window.

Your actual Minecraft server can remain running for other players.
If you also want to stop the provided server launcher, press **Ctrl+D** in its window.
That launcher sends `stop` and exits its restart loop.
Stop your local LLM through its own launcher when you no longer need it.

For a later session, start the same services and select the same console instance.
The checkpoint restores its accepted memory.
Changing the seed does not replace an existing checkpoint.
For another independent experiment, deploy another instance name.
Keep one active instance per Minecraft body.

## When something fails

| Symptom | First action |
| --- | --- |
| Local LLM connection refused | Start the LLM and repeat the `/v1/models` request. |
| Unknown model ID | Copy the actual `id` into both chat entries. |
| Context discovery error | Set the actual `context_window` in both chat entries. |
| Missing embedding credential | Supply the key before starting the console. |
| Wrong MCP port | Set `MINECRAFT_MCP_URL` before starting the console and worker. |
| Required body or browser file missing | Run `setup.ps1` and inspect the copied path entries. |
| Babymode version error | Install or enable Agentic Babymode 0.5.0 or newer on the selected server. |
| RCON connection error | Check its host, port, password, and `enable-rcon=true`. |
| Player login or session error | Check offline authentication compatibility and the whitelist. |
| Spamuel says he cannot play | Inspect the connection checks and request the explicit managed workflow. |
| Old memories return | Restart the same instance for continuity, or deploy another instance for a fresh copy. |

## More detail

- [Model connections](../models.md) explains context discovery and embedding configuration.
- [Console procedure](console.md) explains traces, memory, and worker controls.
- [Minecraft procedure](minecraft.md) explains MCP discovery and real action evidence.
- [`main.build_app`][], [`runtime.models.load_config`][], and [`console.server.serve`][] are the application and configuration entry points.
- [pm-minecraft configuration](https://github.com/flamingrickpat/pm-minecraft/blob/main/mcmcp/config.py) defines the supported environment entries.
- [Current body authentication](https://github.com/flamingrickpat/pm-minecraft/blob/main/body/src/bot.ts) explicitly selects offline mode.
- [Mineflayer API](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md#mineflayercreatebotoptions) describes its account connection options.

The commands and fields match the current source. This guide does not claim a new live gameplay acceptance run.
