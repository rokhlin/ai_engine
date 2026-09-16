<#
.SYNOPSIS
    Runner & Development Launcher for Media Cataloger AI Engine.

.DESCRIPTION
    Launches and manages the Media Cataloger Python AI Engine and FastAPI remote daemon.

.EXAMPLE
    .\run.ps1 api             # Start Python Cataloger API service (port 8001)
    .\run.ps1 scan            # Run offline media scanning / cataloging CLI
    .\run.ps1 test            # Run pytest suite
    .\run.ps1 info            # Display environment configurations
    .\run.ps1 db:status       # Display database status
    .\run.ps1 up              # Run Docker container
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "api",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Determine Python Executable
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$VenvPython2 = Join-Path $ScriptDir "venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
} elseif (Test-Path $VenvPython2) {
    $PythonExe = $VenvPython2
} else {
    $PythonExe = "python"
}

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "       Media Cataloger AI Engine - Command Runner        " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# Launch command via manage.py
$ManagePy = Join-Path $ScriptDir "manage.py"
$FullArgs = @($Command)
if ($ExtraArgs) {
    $FullArgs += $ExtraArgs
}

& $PythonExe $ManagePy @FullArgs
