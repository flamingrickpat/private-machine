param(
    [Parameter(Mandatory)][ValidateSet('validate','preflight','launch','resume','status','metrics','export','unpack')][string]$Operation,
    [Parameter(Mandatory)][string]$Campaign,
    [string]$Scenario,
    [string]$Bundle,
    [string]$Destination,
    [string]$Config = 'config.yaml'
)
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    $arguments = @('-m','simulation.baseline',$Operation,'--campaign',$Campaign,'--config',$Config)
    if ($Scenario) { $arguments += @('--scenario',$Scenario) }
    if ($Bundle) { $arguments += @('--bundle',$Bundle) }
    if ($Destination) { $arguments += @('--destination',$Destination) }
    & .\.venv\Scripts\python.exe @arguments
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
