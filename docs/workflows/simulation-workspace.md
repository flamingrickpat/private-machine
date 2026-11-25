# Inspect a simulation

Use the workspace to read the conversation and find the cycle that produced a reply.

From the PM checkout, use PowerShell. Install Git, uv, and Node.js 20 or newer first.

```powershell
uv sync --locked
if (-not (Test-Path ../pm-minecraft)) {
    git clone https://github.com/flamingrickpat/pm-minecraft.git ../pm-minecraft
}
& ../pm-minecraft/setup.ps1
if (-not (Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
& ./.venv/Scripts/python.exe -m simulation --config config.yaml --port 8095
```

Open [the local workspace](http://127.0.0.1:8095/). Select a run, then open its
memory debugger. Use the timeline to select a cycle and inspect its records,
prompt, and timing. See [the simulation subsystem](../subsystems/simulation.md).
