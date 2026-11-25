# Jesse and Spamuel in Minecraft

See how Spamuel connects Jesse’s requests with events in the shared world.

Use PowerShell in the `private-machine` folder. First [install the Minecraft
mods and MCP](simulation-minecraft-server.md). Those steps create
`config_simulation.yaml` and connect the launcher to your Fabric installation.

## Configure your models

```powershell
if (-not (Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
notepad config.yaml
```

Set the chat and embedding endpoints in `config.yaml`. See [model configuration](../models.md).
The example simulation requires a capped OpenRouter key. For local inference, set
`require_capped_openrouter_key: false` in `config_simulation.yaml`.
Leave `base_checkpoint: null` to start from the synthetic Spamuel canon.

## Start the simulation

```powershell
& ./simulation/simulation_4_jesse_spamuel_minecraft.ps1 -RunId first -Preflight
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed.' }
& ./simulation/simulation_4_jesse_spamuel_minecraft.ps1 -RunId first -CheckServices
if ($LASTEXITCODE -ne 0) { throw 'Minecraft service check failed.' }
& ./simulation/simulation_4_jesse_spamuel_minecraft.ps1 -RunId first
```

If preflight reports a Java or browser path, [set the path](simulation-minecraft-server.md#fix-a-preflight-error-about-java-or-the-browser).
If the service check fails, open the server or MCP log listed in PowerShell.
The last command starts a new world and connects Spamuel to it.
Keep the terminal open. Use a new run ID for a fresh run.
Find transcripts and traces under `.runtime/simulation/runs/first-jesse-spamuel-minecraft/`.
Open the [simulation workspace](simulation-workspace.md) to inspect them.
