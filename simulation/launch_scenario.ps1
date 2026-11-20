param(
    [Parameter(Mandatory)][string]$Scenario,
    [string]$Config = 'config_simulation.yaml',
    [string]$RunId = 'local',
    [switch]$Preflight,
    [switch]$CheckServices
)
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Install the PM virtual environment in $root first. See $root/readme_simulation.md."
    exit 2
}
Push-Location $root
try {
    $arguments = @('-m','simulation.simulation_launcher','--config',$Config,'--sim_config',"test_data/simulation/baseline-v1/$Scenario.json",'--run-id',$RunId)
    if ($Preflight) { $arguments += '--preflight' }
    if ($CheckServices) { $arguments += '--check-services' }
    & $python @arguments
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
