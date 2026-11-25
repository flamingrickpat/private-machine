# Import an old chat

Give your character shared history to draw on during new conversations.
The importer reads CSV and uses the configured model for reconsolidation.

Install Git, uv, and Node.js 20 or newer. Open PowerShell in the PM checkout.

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
if (-not (Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
& ./.venv/Scripts/python.exe -m pm.app.character_cli init --home .runtime/characters/sunny --companion-name Sunny --user-name Alex --card examples/characters/sunny/card.md --personality examples/characters/sunny/personality.json
```

Set the [model endpoints](../models.md). Put your CSV in the checkout or
use the sample. Make sure the speaker names match the home.
See the [CSV format](../subsystems/character-import.md).

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
& ./.venv/Scripts/python.exe -m pm.app.character_cli import --home .runtime/characters/sunny --csv examples/characters/sunny/chat.csv --config config.yaml
if ($LASTEXITCODE -ne 0) { throw 'Chat import failed.' }
& ./.venv/Scripts/python.exe main.py --home .runtime/characters/sunny --config config.yaml
```

Open the address that PM prints. If you used the sample, ask which pot holds the mint.
