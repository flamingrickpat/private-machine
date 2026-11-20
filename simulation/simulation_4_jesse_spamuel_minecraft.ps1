param([string]$Config = 'config_simulation.yaml', [string]$RunId = 'local', [switch]$Preflight, [switch]$CheckServices)
& "$PSScriptRoot/launch_scenario.ps1" -Scenario 'jesse-spamuel-minecraft' -Config $Config -RunId $RunId -Preflight:$Preflight -CheckServices:$CheckServices
exit $LASTEXITCODE