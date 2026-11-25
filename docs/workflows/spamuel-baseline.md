# Overnight Spamuel baseline

Jesse and Spamuel played Minecraft overnight on November 24 to 25, 2026.
The run completed six scenes at six times normal story speed. The recording
contains 44 timeline entries, cycle evidence, and workflow timings.

## Inspect the recording

Open PowerShell in the `private-machine` folder. Install Git and uv first.
This step uses a small archive and needs no model key or Minecraft server.

```powershell
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw 'PM dependency installation failed.' }
$baseline = '.runtime/spamuel-baseline'
if (-not (Test-Path "$baseline/runs")) {
    Expand-Archive test_data/simulation/baseline-v1/spamuel-overnight-evidence.zip $baseline
}
& ./.venv/Scripts/python.exe -m simulation --data $baseline --port 8095
```

Keep PowerShell open. Choose a view from this list.

- [Conversation and world events](http://127.0.0.1:8095/#run/interrupt-overnight-20261124-jesse-spamuel-minecraft/conversation)
- [Cycle 1000 in the debugger](http://127.0.0.1:8095/debugger/#/run/interrupt-overnight-20261124-jesse-spamuel-minecraft/1/overview/cycle/primary/1000)
- [Profiler](http://127.0.0.1:8095/debugger/#/run/interrupt-overnight-20261124-jesse-spamuel-minecraft/1/profile)

## Restore the full archive

Install [Git LFS](https://git-lfs.com/). Stop the workspace with Ctrl+C if it is
open. From the `private-machine` folder, paste this block into PowerShell.
The three archives total about 454 MB. Extraction uses more disk space.

```powershell
git lfs install --local
if ($LASTEXITCODE -ne 0) { throw 'Git LFS setup failed.' }
$largeArchives = 'test_data/baselines/spamuel-overnight-checkpoint.zip,test_data/baselines/spamuel-overnight-logs-world.tar.zst'
git lfs pull --include $largeArchives
if ($LASTEXITCODE -ne 0) { throw 'Git LFS download failed.' }

$manifest = Get-Content -LiteralPath test_data/baselines/spamuel-overnight-manifest.json -Raw | ConvertFrom-Json
foreach ($archive in $manifest.archives.PSObject.Properties.Value) {
    $sha256 = (Get-FileHash -LiteralPath $archive.path -Algorithm SHA256).Hash
    if ($sha256 -ne $archive.sha256) { throw "Archive hash mismatch for $($archive.path)." }
}

$baseline = '.runtime/spamuel-baseline'
$run = "$baseline/runs/$($manifest.run_id)"
if (-not (Test-Path "$run/state.json")) {
    Expand-Archive $manifest.archives.debugger_evidence.path $baseline -Force
}
$baselineHome = "$run/participants/primary"
if (-not (Test-Path "$baselineHome/ghost.pickle")) {
    Expand-Archive $manifest.archives.checkpoint.path $baselineHome -Force
    Move-Item "$baselineHome/checkpoint.json" "$baselineHome/ghost.pickle"
}
tar -xf $manifest.archives.logs_and_world.path -C $baseline
if ($LASTEXITCODE -ne 0) { throw 'Prompt and world extraction failed.' }
```

Restart the workspace with the [inspection command](#inspect-the-recording).
Open [Records](http://127.0.0.1:8095/debugger/#/run/interrupt-overnight-20261124-jesse-spamuel-minecraft/1/records)
and choose `ghost.pickle` to read the final memory. PM reads the JSON checkpoint
using the character home's standard filename. Open [Calls](http://127.0.0.1:8095/debugger/#/run/interrupt-overnight-20261124-jesse-spamuel-minecraft/1/calls)
and select a call to read its prompt, response, and token counts.

| What you restored | Path under `.runtime/spamuel-baseline` |
| --- | --- |
| Final Spamuel memory and character | `runs/<run-id>/participants/primary/` |
| Companion model logs | `runs/<run-id>/attempts/1/evidence/primary/logs/prompts/` |
| Scenario controller model logs | `runs/<run-id>/attempts/1/model-logs/` |
| Minecraft world and player data | `minecraft/simulation-world/` |
| Minecraft mod settings | `minecraft/config/` |

The archives contain the final state, 34,189 companion prompt files, and 309
controller prompt files. The initial run checkpoint pointer refers to a file
on the capture machine. That file and the full frozen attempt directory are
absent from the archive. For a fresh run, use the shipped canon and scenario.

## Compare with a new run

Open the [Jesse and Spamuel in Minecraft guide](simulation-jesse-spamuel-minecraft.md).
Set `time_multiplier: 6.0` and `base_checkpoint: null` in `config_simulation.yaml`.
To match the recorded model settings, copy the model template.

```powershell
$modelConfig = '.runtime/spamuel-baseline/model-config.yaml'
New-Item -ItemType Directory -Force .runtime/spamuel-baseline | Out-Null
if (-not (Test-Path $modelConfig)) {
    Copy-Item test_data/simulation/baseline-v1/spamuel-overnight-model.example.yaml $modelConfig
}
notepad $modelConfig
notepad config_simulation.yaml
```

Enter your OpenRouter key in the copied YAML. Set
`model_config_path: .runtime/spamuel-baseline/model-config.yaml` in the simulation configuration.
The baseline used DeepSeek V4.1 Flash, a 32,000-token chat context override, and
`perplexity/pplx-embed-v1-0.6b` with 1,024 dimensions.

When the new run completes, keep the baseline on port 8095. Open the new run
in a second workspace on port 8096.

```powershell
& ./.venv/Scripts/python.exe -m simulation --data .runtime/simulation --port 8096
```

Open the [new run workspace](http://127.0.0.1:8096/). Compare the same scene,
then compare workflow durations and call token counts in both debuggers.
Model responses and Minecraft actions can vary between runs.

## Continue as Jesse

Restore the full archive first. Use the copied model configuration from the
[comparison steps](#compare-with-a-new-run). Keep the same embedding model and
1,024 dimensions to read the existing vectors.

Paste this block into PowerShell to start a chat session. The first run uses
`--clean` to write accepted history into a new character home. Later runs use
that home.

```powershell
$runId = 'interrupt-overnight-20261124-jesse-spamuel-minecraft'
$baselineHome = ".runtime/spamuel-baseline/runs/$runId/participants/primary"
$continuation = '.runtime/characters/spamuel-overnight-clean'
if (-not (Test-Path "$baselineHome/ghost.pickle")) { throw 'Restore the full archive first.' }
if (-not (Test-Path "$continuation/ghost.pickle")) {
    New-Item -ItemType Directory -Force "$continuation/runtime" | Out-Null
    Copy-Item "$baselineHome/character.yaml", "$baselineHome/personality.json" $continuation
    Copy-Item "$baselineHome/runtime/clock.json" "$continuation/runtime/clock.json"
    & ./.venv/Scripts/python.exe main.py --home $baselineHome --clean "$continuation/ghost.pickle"
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the clean checkpoint.' }
}
& ./.venv/Scripts/python.exe main.py --home $continuation --config .runtime/spamuel-baseline/model-config.yaml
```

Open the Web chat URL printed in PowerShell. To bring the Minecraft world into
the same session, use the next section before you start the chat.

When you finish, press Ctrl+C once and wait for the PowerShell prompt.
If work is active, PM saves a checkpoint before it waits.
It saves again when the work finishes.
The saved delivery receipt keeps the reply in Spamuel's history after restart.
If you close the process before that save, a reply visible in chat can be
absent when you reopen the home.

## Continue with the saved Minecraft world

Restore the full archive and [install the mods and MCP](simulation-minecraft-server.md).
The numbered scenario launchers create fresh worlds. Use a standalone Fabric
server and MCP for the archived world.

### Start Fabric in the first PowerShell window

Open PowerShell in the `private-machine` folder. Enter your installed Fabric
folder and your Java executable when prompted. The commands make a writable
copy of the world and keep its settings on subsequent launches.

```powershell
$fabric = (Resolve-Path -LiteralPath (Read-Host 'Path to the installed Fabric server folder')).Path
$java = Read-Host 'Path to java.exe, or java if it is on PATH'
$archive = '.runtime/spamuel-baseline/minecraft'
$server = '.runtime/spamuel-world-server'
if (-not (Test-Path "$server/simulation-world/level.dat")) {
    New-Item -ItemType Directory -Force $server | Out-Null
    foreach ($file in 'fabric-server-launch.jar', 'server.jar', 'fabric-server-launcher.properties', 'eula.txt') {
        Copy-Item -LiteralPath (Join-Path $fabric $file) -Destination $server -ErrorAction Stop
    }
    foreach ($folder in 'libraries', 'mods', 'versions') {
        $path = Join-Path $fabric $folder
        if (Test-Path -LiteralPath $path) {
            Copy-Item -LiteralPath $path -Destination $server -Recurse -ErrorAction Stop
        }
    }
    $password = [guid]::NewGuid().ToString('N')
    (Get-Content "$archive/server.properties" -Raw) `
        -replace '(?m)^server-ip=.*', 'server-ip=0.0.0.0' `
        -replace '(?m)^server-port=.*', 'server-port=25565' `
        -replace '(?m)^rcon\.port=.*', 'rcon.port=25575' `
        -replace '(?m)^rcon\.password=.*', "rcon.password=$password" |
        Set-Content "$server/server.properties" -Encoding ASCII
    Copy-Item "$archive/config", "$archive/simulation-world" -Destination $server -Recurse -ErrorAction Stop
}
Push-Location $server
try { & $java -Xmx2048M -jar fabric-server-launch.jar nogui }
finally { Pop-Location }
```

Wait for the server to print `RCON running` after `Done`. Leave this window
open. When you finish playing, type `stop` in this window to save the world.

### Start MCP in the second PowerShell window

Open a new PowerShell window in the `private-machine` folder. Paste this block.
Enter the full path to your Chromium browser executable when prompted.

```powershell
$mcp = (Resolve-Path '../pm-minecraft').Path
$browser = Read-Host 'Full path to chrome.exe, msedge.exe or brave.exe'
$playerHome = Join-Path (Get-Location).Path '.runtime/spamuel-world-mcp'
New-Item -ItemType Directory -Force $playerHome | Out-Null
$passwordLine = Select-String -LiteralPath '.runtime/spamuel-world-server/server.properties' -Pattern '^rcon\.password='
$password = $passwordLine.Line.Split('=', 2)[1]
$bodySettings = Join-Path $playerHome 'body.env'
$settings = @"
BROWSER_EXECUTABLE_PATH=$browser
MCMCP_BODY_ENTRYPOINT=$mcp/body/dist/main.js
MCMCP_AGENT_HOME=$playerHome
MCP_HOST=0.0.0.0
MCP_PORT=6767
MCMCP_BODY_PORT=6768
MCMCP_VIEWER_PORT=6769
MCMCP_CONNECTION_TOOLS=true
MINECRAFT_HOST=127.0.0.1
MINECRAFT_PORT=25565
MINECRAFT_PLAYER=spamuel_solo
MINECRAFT_RCON_HOST=127.0.0.1
MINECRAFT_RCON_PORT=25575
MINECRAFT_RCON_PASSWORD=$password
"@
[System.IO.File]::WriteAllText($bodySettings, $settings)
Push-Location $mcp
try { & ./.venv/Scripts/python.exe main.py $bodySettings }
finally { Pop-Location }
```

Keep `spamuel_solo` as the player name. Fabric uses this offline identity to
load the saved position and inventory. Wait for MCP to print its listening URL.

### Connect the companion in the third PowerShell window

Open `.runtime/spamuel-baseline/model-config.yaml`. Replace its `minecraft: null` line with
this configuration.

```yaml
minecraft:
  mcp_url: http://127.0.0.1:6767/mcp
```

Run the [Continue as Jesse](#continue-as-jesse) commands. If the model file is
open in another application, save it before starting PM.

## Capture settings

The original Minecraft service used `pm-minecraft` revision `6a12a4f`.
Revision `c8d3b39` fixes stale oxygen after death. The
[manifest](https://github.com/flamingrickpat/private-machine/blob/beta-context/test_data/baselines/spamuel-overnight-manifest.json)
lists both revisions, mod hashes, archive hashes, and the PM revision that
matches the recorded runtime source.
