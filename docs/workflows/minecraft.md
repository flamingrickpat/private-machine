# Play Minecraft with the companion

A body lets the companion observe a shared world and act on it. The same
memory and intentions that guide chat can inform those actions.

To start a disposable world, use the [Jesse and Spamuel simulation](simulation-jesse-spamuel-minecraft.md).
For two companions, use the [two Spamuels simulation](simulation-two-spamuels-minecraft.md).

## Connect a character to your world

Install Git, uv, and Node.js 20 or newer. Open PowerShell in the PM checkout.
Prepare a [Minecraft server](simulation-minecraft-server.md).

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
if (-not (Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
```

Start pm-minecraft against your world using its checkout instructions.
Enter its MCP URL under `minecraft.mcp_url` in `config.yaml`. Use the example
`minecraft` section at the end of the YAML. Set the chat and embedding endpoints.

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
New-Item -ItemType Directory -Force .runtime/characters | Out-Null
if (-not (Test-Path .runtime/characters/jesse-spamuel/ghost.pickle)) {
    & ./.venv/Scripts/python.exe -c "from pathlib import Path; from simulation.simulation_launcher import prepare_seed; prepare_seed(None, Path('.runtime/characters/jesse-spamuel'))"
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare Spamuel.' }
}
& ./.venv/Scripts/python.exe main.py --home .runtime/characters/jesse-spamuel --config config.yaml
```

Open the address printed by PM. Ask Spamuel to describe the world.
Use the [Minecraft evidence guide](minecraft-evidence.md) to inspect the action.
