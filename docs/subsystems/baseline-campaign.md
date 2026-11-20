# Companion regression campaign

Run the same six scenes over three story dates for each scenario. Each launch forks its own participant homes. Use a new campaign ID for each candidate revision. Keep the completed campaign available for comparison.

The four templates read the synthetic home at `~/.pm/private-machine/spamuel-simulation-a`. On a fresh installation, select **+ Spamuel** in the simulation workspace and enter `spamuel-simulation-a` as the folder name. Keep this seed home separate from personal chat. Each exported participant contains its starting memory at `runtime/snapshots/initial.pickle`. Use the same starting memory when comparing revisions.

## Prepare the models

The four numbered PowerShell launchers provision private Minecraft worlds and player homes. Each launcher uses one fixed scenario and one installation configuration. Start with the [Windows simulation setup guide](https://github.com/flamingrickpat/private-machine/blob/HEAD/readme_simulation.md). The manual world and port assignments below apply to the shared campaign launcher when you operate the Minecraft services yourself.

Use a local model configuration with `profile_cognition: true`. Configure every chat preset and the embedding connection with the same campaign OpenRouter key. Set a lifetime key limit of $100 or less and leave its reset interval unset. The running development simulation can continue with its own key. Store the campaign configuration under `.runtime`.

## Prepare two Minecraft worlds

Use the server, updated mod, and MCP checkout intended for the release. Record the mod build and MCP Git revision with your campaign. The existing installation instructions cover Java, Node, RCON and the per-body environment. [Minecraft setup](https://github.com/flamingrickpat/private-machine/blob/HEAD/README.md#connect-a-character-to-minecraft)

Create two disposable server directories, each with its own world folder and RCON port. Set `level-seed=798112` before either world generates. For repeat measurements, start both worlds from their saved initial copies. The suite reads its expected player identities and server addresses from [the world layout](https://github.com/flamingrickpat/private-machine/blob/HEAD/test_data/simulation/baseline-v1/minecraft.json).

| World | Game port | RCON port | Player | MCP port | Body port | Viewer port |
| --- | --- | --- | --- | --- | --- | --- |
| Shared | 12345 | 25575 | spamuel_1 | 6777 | 6779 | 6781 |
| Shared | 12345 | 25575 | spamuel_alpha | 6778 | 6780 | 6782 |
| Solo | 12346 | 25576 | spamuel_solo | 6783 | 6784 | 6785 |

In the MCP checkout, make a local environment file for each row. Set `MINECRAFT_HOST=127.0.0.1`, that world's game and RCON ports, and the matching RCON password. Set `MINECRAFT_PLAYER`, `MCP_PORT`, `MCMCP_BODY_PORT`, and `MCMCP_VIEWER_PORT` from the table. Give each player a separate `MCMCP_AGENT_HOME`. Set the listener `MCP_HOST=0.0.0.0`. Preserve the working body and browser executable settings from your MCP installation.

Start both servers. In three PowerShell terminals at the MCP checkout, start each body with its corresponding environment file.

```powershell
.venv/Scripts/python.exe main.py .env.spamuel_1
.venv/Scripts/python.exe main.py .env.spamuel_alpha
.venv/Scripts/python.exe main.py .env.spamuel_solo
```

## Check and launch

From the PM repository, validate the fixed plans and their source homes.

```powershell
.\simulation\baseline.ps1 validate candidate01
.\simulation\baseline.ps1 preflight candidate01 -Config .runtime\campaign.yaml
```

Preflight identifies all selected bodies, checks that each is spawned on its assigned server, turns each camera, observes the resulting heading, and returns it to its starting heading. It then reads the OpenRouter key limit. Use disposable worlds because preflight changes the camera heading briefly.

```powershell
.\simulation\baseline.ps1 launch candidate01 -Config .runtime\campaign.yaml
.\simulation\baseline.ps1 status candidate01
.\simulation\baseline.ps1 resume candidate01 -Config .runtime\campaign.yaml
```

Use `-Scenario jesse-spamuel`, `jesse-two-spamuels`, `two-spamuels-minecraft`, or `jesse-spamuel-minecraft` to select one run. The full suite opens each run individually, drains its initial exchange, then starts the parallel segment. Before launch, the controller checks Windows commit headroom. The suite reserves 16 GiB plus 8 GiB per starting participant. During execution it requests a pause if free commit falls below 16 GiB. Resume requires the same configuration and worlds. A stale Minecraft world cannot safely accompany a restored character checkpoint. Keep the failure receipt and start a new campaign if the world and characters have diverged.

## Inspect and export

Run `simulation\baseline.ps1 metrics candidate01` to measure time from the first waiting actor input to the next visible participant speech. Several inputs can share one wait. The report includes p95, the longest wait, and participants still waiting. Participant cost totals include only calls with a price in the provider receipt. Harness calls can also consume campaign credit.

Open **Memory debugger** in the simulation workspace. Chat and Records let you select participant memory or a saved snapshot. Calls opens exact requests and responses. The status header checks the controller every ten seconds. Select **Reload** to read new prompts and checkpoint data.

The profiler uses real elapsed time. Each lane identifies a participant, process, thread and operation type. Select a codelet to inspect its input record IDs, wave and score. Select a cognition commit to see its control generation, saved records and cycle. A Minecraft tool span includes its journal sequence when the body dispatches the call. Calls link to their logged payloads and carry a SHA-256 digest when available.

With `profile_cognition` enabled, each process writes timing events in its participant home's `runtime/profile` folder. The events stay outside the persona checkpoint. Historical attempts show the timing evidence recorded by their original runtime.

Export a completed or quiescent campaign. The exporter stores each distinct file content once and checks for recognizable credentials before publication. Review the synthetic bundle before committing it. Fetch Git LFS objects before opening the bundle in another checkout.

```powershell
.\simulation\baseline.ps1 export candidate01
.\simulation\baseline.ps1 unpack candidate01 -Bundle test_data\baselines\candidate01.zip -Destination .runtime\review-candidate01
.venv/Scripts/python.exe -m pm.tools.timeline --runs-root .runtime\review-candidate01
```

The unpack destination must be new. The debugger reads the restored evidence from its run folders. A participant's synthetic character home contains its checkpoint and character configuration.
