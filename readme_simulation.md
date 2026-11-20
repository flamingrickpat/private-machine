# Run a companion simulation on Windows

From the PM checkout, choose one PowerShell launcher. Each scenario runs six scenes over three story days. The first two use chat only.

| Script in `simulation\` | Participants and setting |
| --- | --- |
| `simulation_1_jesse_spamuel.ps1` | Jesse and Spamuel in chat |
| `simulation_2_jesse_two_spamuels.ps1` | Jesse, Spamuel 1 and Spamuel Alpha in chat |
| `simulation_3_two_spamuels_minecraft.ps1` | Spamuel 1 and Spamuel Alpha in one private Minecraft world |
| `simulation_4_jesse_spamuel_minecraft.ps1` | Jesse and Spamuel in a private Minecraft world |

## First use

1. Install the PM environment with `uv sync --locked`. Use the same chat and embedding endpoint settings that you use for PM. Start with `config.example.yaml` if you need a model configuration.
2. Run `Copy-Item config_simulation.yaml.example config_simulation.yaml` once. Set `model_config_path` to your model configuration. Relative paths in the simulation configuration start at its directory.
3. Set `base_checkpoint` to a trusted character's `ghost.pickle`, or leave it `null` to use the supplied Spamuel canon. PM forks a home for each participant. Use a synthetic persona for baseline archives.
4. For Minecraft, set `mcp_checkout` to your installed `pm-minecraft` checkout. Run its `setup.ps1` to install the MCP environment and build the Node body. Set `server_installation` to your Fabric 1.19.4 server directory. On Rick's PC these paths are `../pm-minecraft` and `../../minecraft-server-real`.

The Fabric installation supplies Java server JARs, libraries, Fabric API, the Agentic Babymode mod, and your accepted `eula.txt`. The MCP also needs Node 20 or later and a compatible Chromium browser. The launcher selects the ports and creates the world and player directories.

## Check, then launch

```powershell
.\simulation\simulation_1_jesse_spamuel.ps1 -Preflight
.\simulation\simulation_3_two_spamuels_minecraft.ps1 -Preflight
```

`-Preflight` checks local settings and files. To test the full Minecraft installation, use `-CheckServices`. It starts a disposable world, checks each spawned player's identity and camera, then stops the services. Each service check has its own world. Neither check starts paid inference.

```powershell
.\simulation\simulation_3_two_spamuels_minecraft.ps1 -CheckServices
```

To start a paid run, call the chosen script without a check switch. Keep its terminal open. Launch different scenarios in separate terminals to run them together.

```powershell
.\simulation\simulation_1_jesse_spamuel.ps1 -RunId baseline01
```

The default run ID is `local`. Use a new ID to repeat a completed scenario. Use `-Config C:\path\to\config_simulation.yaml` to select another installation. The default paid campaign requires the same OpenRouter key for all chat and embedding routes, with a lifetime cap of $100 or less. To use another endpoint or spending policy, explicitly set `require_capped_openrouter_key: false`.

## Find a setting or a failed startup

Open `.runtime/simulation/instances/<run-id>-<scenario>/`. Preflight creates the following editable copies for a Minecraft scenario. Repeated preflights preserve your edits.

| File | What to change |
| --- | --- |
| `config_pm_minecraft.yaml` | Browser executable, Node executable, MCP Python, compiled body entry point |
| `config_minecraft_server.yaml` | Java executable, heap size, startup timeout |
| `server.properties` | Minecraft server settings. Keep the game port, RCON port, password, and seed placeholders. |
| `agentic_babymode.json` | Player, environment, mob, sleepiness, nutrition, and chat rules |
| `pickup-range-server.json` | Item and experience pickup ranges |

If preflight cannot find Brave, set `browser` in the named instance's `config_pm_minecraft.yaml` to your browser executable. Run the same command with the same ID. To change defaults for future instances, edit the corresponding file in [simulation/setup](simulation/setup).

Each launch writes a fresh server directory under `launches/<timestamp>/`. Service checks use `service-checks/<timestamp>/`. The launcher copies the server binaries and mods, applies the instance's mod settings and generates a new world. Your installation's existing world stays in its original directory.

Open `services.json` for the seed, ports, player homes, process IDs and mod hashes. Read `server.log` for Java startup and `players/<participant>/mcp.log` for MCP startup. Generated connection files contain local credentials. Keep the instance folder private.

PM saves the conversation, prompt logs, checkpoints, and workflow evidence under `.runtime/simulation/runs/<run-id>-<scenario>/`. Open the memory debugger from the simulation workspace. For export and comparison, see the [campaign guide](docs/subsystems/baseline-campaign.md).
