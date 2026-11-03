# Remove superseded local evidence; retain the current immersion review receipts.
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$runtimeRoot = [IO.Path]::GetFullPath((Join-Path $workspace '.runtime'))
if ((Split-Path -Parent $runtimeRoot) -ne $workspace) { throw 'Runtime root escaped the repository.' }
if ((Get-Item -LiteralPath $runtimeRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Runtime root must not be a link.'
}
$preserved = Join-Path $runtimeRoot 'actor-immersion-20261103'
$targets = @(Get-ChildItem -LiteralPath $runtimeRoot -Force | Where-Object { $_.FullName -ne $preserved })
foreach ($target in $targets) {
    $resolvedTarget = [IO.Path]::GetFullPath($target.FullName)
    if ((Split-Path -Parent $resolvedTarget) -ne $runtimeRoot) { throw "Target escaped runtime: $resolvedTarget" }
    if ($target.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Target is a link: $resolvedTarget" }
    if ($target.PSIsContainer) {
        $links = @(Get-ChildItem -LiteralPath $resolvedTarget -Recurse -Force |
            Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
        if ($links.Count) { throw "Target contains links: $resolvedTarget" }
    }
}
Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(pythonw?|chrome|chromium)\.exe$' -and
    $_.CommandLine -and $_.CommandLine.Contains($workspace)
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
foreach ($target in $targets) {
    Write-Host "Removing $($target.FullName)"
    Remove-Item -LiteralPath $target.FullName -Recurse -Force
}
Write-Host "Preserved current review evidence: $preserved"
