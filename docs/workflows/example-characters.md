# Create a character

Give your companion a name, a character card, and personality values.
An imported chat gives the companion shared history to draw on.

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

Set the chat and embedding endpoints in `config.yaml`. See [model configuration](../models.md).
For your own companion, substitute your names, card, and personality file.

Use the [chat import guide](import-chat.md) to add a history before starting.

## Start chat

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
& ./.venv/Scripts/python.exe main.py --home .runtime/characters/sunny --config config.yaml
```

Open the address that PM prints.
