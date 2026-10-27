<#
.SYNOPSIS
Open the local development board with the repository's locked Python environment.
.PARAMETER Port
The loopback port. The default is 8767.
.PARAMETER Data
An existing board file. Relative paths start in the caller's directory.
.PARAMETER NoBrowser
Print the address without opening a browser.
.EXAMPLE
.\start.ps1 -Port 8768 -NoBrowser
.NOTES
Run this script from any directory. Stop the server with Ctrl+C.
Each completed save remains in the selected JSON file.
#>
param(
    [int]$Port = 8767,
    [string]$Data = (Join-Path $PSScriptRoot 'board.json'),
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$repositoryPath = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$serverPath = Join-Path $PSScriptRoot 'server.py'
$kanbanArguments = @('run', '--locked', '--project', $repositoryPath, 'python',
                     $serverPath, '--port', $Port, '--data', $Data)
if ($NoBrowser) {
    $kanbanArguments += '--no-browser'
}
& uv @kanbanArguments
exit $LASTEXITCODE
