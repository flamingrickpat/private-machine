# Chat with Spamuel as Jesse

The shipped synthetic canon gives Spamuel shared memories with Jesse.

Install Git, uv, and Node.js 20 or newer. Open PowerShell in the PM checkout.

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
if (-not (Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
```

Set the chat and embedding endpoints in `config.yaml`. See [model configuration](../models.md).

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

Type as Jesse. Use the same home to continue the conversation.
See [character creation and chat import](example-characters.md).
