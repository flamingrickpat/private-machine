# Overnight Spamuel baseline

Jesse and Spamuel played Minecraft overnight on November 24 to 25, 2026.
The run reached the completed state at six times normal story speed. Use its recorded
conversation and timings to compare your own run. The archive contains six
completed scenes and 44 timeline entries.

## Inspect the recording

Install Git, uv, and Node.js 20 or newer. Open PowerShell in the PM checkout.

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
if (-not (Test-Path .runtime/spamuel-baseline/runs)) {
    Expand-Archive test_data/simulation/baseline-v1/spamuel-overnight-evidence.zip .runtime/spamuel-baseline
}
& ./.venv/Scripts/python.exe -m simulation --data .runtime/spamuel-baseline --port 8095
```

Open [the baseline conversation](http://127.0.0.1:8095/#run/interrupt-overnight-20261124-jesse-spamuel-minecraft/conversation).
Open [the profiler](http://127.0.0.1:8095/debugger/#/run/interrupt-overnight-20261124-jesse-spamuel-minecraft/1/profiler)
to compare cycle timings. The archive also includes the scenario plan, world
events, and the first attempt's report.
Open [cycle 1000](http://127.0.0.1:8095/debugger/#/run/interrupt-overnight-20261124-jesse-spamuel-minecraft/1/overview/cycle/primary/1000)
for an example of captured cycle evidence.

## Run the same scenario

Use the [Jesse and Spamuel in Minecraft guide](simulation-jesse-spamuel-minecraft.md).
Set `time_multiplier: 6.0` in your simulation configuration. Leave
`base_checkpoint: null` to start from the shipped Spamuel canon.
The overnight run used OpenRouter with `deepseek/deepseek-v4.1-flash-1017`
and `perplexity/pplx-embed-v1-0.6b` (1,024 dimensions). Chat context used
a 32,000-token override. Source revision was `a3d0dd5c45ee` with local changes.
Copy `test_data/simulation/baseline-v1/spamuel-overnight-model.example.yaml` to a local file.
Enter your OpenRouter key and point `model_config_path` to that file.
Compare the conversation and cycle timing in the two runs.

## Inspect the final memory

Install Git LFS. From the PM checkout, use PowerShell to restore the portable
JSON checkpoint into the recording's participant home.

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
git lfs pull --include=test_data/baselines/spamuel-overnight-checkpoint.zip
if ($LASTEXITCODE -ne 0) { throw 'Git LFS download failed.' }
$baselineHome = '.runtime/spamuel-baseline/runs/interrupt-overnight-20261124-jesse-spamuel-minecraft/participants/primary'
if (-not (Test-Path "$baselineHome/ghost.pickle")) {
    Expand-Archive test_data/baselines/spamuel-overnight-checkpoint.zip $baselineHome -Force
    Move-Item "$baselineHome/checkpoint.json" "$baselineHome/ghost.pickle"
}
& ./.venv/Scripts/python.exe -m simulation --data .runtime/spamuel-baseline --port 8095
```

The ZIP contains JSON memory. PM detects its format by reading the file.
The name `ghost.pickle` is the character home's checkpoint filename.
Select that checkpoint in the debugger to inspect its records.

## Continue as Jesse

Copy the restored home. Set the same embedding model and dimensions in
`config.yaml`. See [model configuration](../models.md).

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
if (-not (Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
$baselineHome = '.runtime/spamuel-baseline/runs/interrupt-overnight-20261124-jesse-spamuel-minecraft/participants/primary'
$continuation = '.runtime/characters/spamuel-overnight'
if (-not (Test-Path $continuation)) {
    New-Item -ItemType Directory -Force .runtime/characters | Out-Null
    Copy-Item $baselineHome $continuation -Recurse
}
& ./.venv/Scripts/python.exe main.py --home $continuation --config config.yaml
```

This starts a chat from Spamuel's final memory. The Minecraft world requires
its own server and world files.
