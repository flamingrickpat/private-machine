param(
    [int]$Port = 8095
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    uv run --locked python -m simulation --config config.yaml --openrouter-only --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Simulation exited with code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
