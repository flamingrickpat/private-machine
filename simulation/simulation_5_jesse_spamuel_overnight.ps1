param(
    [string]$Config = 'config.yaml',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')][string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss'),
    [switch]$SmokeOnly,
    [switch]$Interactive,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')][string]$ContinueSmokeFrom,
    [switch]$Worker
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo '.venv/Scripts/python.exe'
$configPath = (Resolve-Path -LiteralPath $Config).Path
$output = Join-Path $repo ".runtime/simulation/overnight"
$log = Join-Path $output "$RunId.log"
if ($Worker) {
    $arguments = '-u -m simulation.overnight --config "{0}" --run-id {1}' -f $configPath, $RunId
    if ($SmokeOnly) { $arguments += ' --smoke-only' }
    if ($ContinueSmokeFrom) { $arguments += " --continue-smoke-from $ContinueSmokeFrom" }
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $repo -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    exit $process.ExitCode
}

if (Test-Path -LiteralPath (Join-Path $output $RunId)) {
    throw "Run $RunId already exists. Use a new run ID."
}
New-Item -ItemType Directory -Path $output -Force | Out-Null
$taskName = "PM-Simulation-5-$RunId"
$shell = (Get-Process -Id $PID).Path
$taskArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -Config "{1}" -RunId {2} -Worker' -f $PSCommandPath, $configPath, $RunId
if ($SmokeOnly) { $taskArguments += ' -SmokeOnly' }
if ($ContinueSmokeFrom) { $taskArguments += " -ContinueSmokeFrom $ContinueSmokeFrom" }
$action = New-ScheduledTaskAction -Execute $shell -Argument $taskArguments -WorkingDirectory $repo
$logonType = 'S4U'
if ($Interactive) { $logonType = 'Interactive' }
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType $logonType -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([timespan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -ErrorAction Stop | Out-Null
Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
Write-Output "Scheduled task $taskName"
Write-Output "Controller log $log"
