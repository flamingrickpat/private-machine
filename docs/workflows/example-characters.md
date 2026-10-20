# Start with an example character

## Summary

Sunny is cheerful and curious. Rowan is quiet and practical.
Both examples include a character card, personality values, and an optional synthetic chatlog with Alex.
The logs contain no real person's conversation.

Parent: [manual](../index.md).
Related: [model connections](../models.md), [character import](../subsystems/character-import.md).

## Create a home

Install the locked dependencies with `uv sync --locked`.
Start your chat and embedding servers.
The example configuration uses ports 8080 and 8081.
Adjust its model aliases and embedding dimensions to match your servers.

On Windows, add Git Bash to `PATH`:

```powershell
$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH
```

Create Sunny in a new home:

```powershell
uv run --locked python -m pm.app.character_cli init --home .runtime/characters/sunny --companion-name Sunny --user-name Alex --card examples/characters/sunny/card.md --personality examples/characters/sunny/personality.json
```

For Rowan, replace `sunny` with `rowan` and `Sunny` with `Rowan`.
An existing home is not overwritten.
Each home has separate memory.

## Optional history

For an empty history, skip this section.
For Sunny's sample history, run:

```powershell
uv run --locked python -m pm.app.character_cli import --home .runtime/characters/sunny --csv examples/characters/sunny/chat.csv --config config.example.yaml
```

The import uses your configured model for memory reconsolidation.
The runtime marks each imported record with its chatlog source and row.
Separate transfer records have an explicit synthetic mark.
The sample history names Alex. A renamed user requires matching names in the CSV.

## Start a conversation

```powershell
uv run --locked python main.py --home .runtime/characters/sunny --config config.example.yaml
```

Open the web address printed by the launcher.
With imported history, ask which pot holds the mint.
Without imported history, introduce yourself and start a new conversation.

To create your own character, supply your own card and personality file to the same `init` command.
The [import guide](../subsystems/character-import.md) describes the CSV columns and personality values.

The examples demonstrate setup and import. Their unit tests do not establish conversation quality.
