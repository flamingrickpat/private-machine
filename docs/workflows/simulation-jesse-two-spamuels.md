# Jesse and two Spamuels

See how each companion follows a group conversation from its own point of view.

Run these commands in PowerShell from the PM checkout. Install Git, uv, and Node.js 20 or newer first.

## Install and copy the configuration

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
if (-not (Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
if (-not (Test-Path config_simulation.yaml)) { Copy-Item config_simulation.yaml.example config_simulation.yaml }
```

Set the chat and embedding endpoints in `config.yaml`. See [model configuration](../models.md).
The example simulation requires a capped OpenRouter key. For local inference, set
`require_capped_openrouter_key: false` in `config_simulation.yaml`.
Leave `base_checkpoint: null` to start from the synthetic Spamuel canon.

## Run

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
& ./simulation/simulation_2_jesse_two_spamuels.ps1 -RunId first -Preflight
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed.' }
& ./simulation/simulation_2_jesse_two_spamuels.ps1 -RunId first
```

Keep the terminal open. Use a new run ID for a fresh run.
Find transcripts and traces under `.runtime/simulation/runs/first-jesse-two-spamuels/`.
Open the [simulation workspace](simulation-workspace.md) to inspect them.
