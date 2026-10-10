#Requires -Version 7.0
<#!
.SYNOPSIS
    Run the repository checks that do not require proprietary GPU SDKs.

.DESCRIPTION
    This is the local counterpart of the core GitHub Actions checks. It keeps
    formatting, locked dependency resolution, tests, Python syntax checks and
    optional Tauri checks in one reproducible entry point. GPU SDK validation
    remains in build-windows.ps1 because it is host and backend specific.
#>
[CmdletBinding()]
param(
    [switch]$Tauri,
    [switch]$AllFeatures
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$initialDirectory = Get-Location

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    Write-Host "> $Program $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program exited with code $LASTEXITCODE"
    }
}

try {
    Set-Location -LiteralPath $repoRoot
    Invoke-Checked cargo @("fmt", "--all", "--", "--check")

    $workspaceArgs = @("--workspace")
    if (-not $AllFeatures) {
        $workspaceArgs += @("--exclude", "vocal-tensorrt")
    }
    Invoke-Checked cargo (@("check") + $workspaceArgs + @("--locked"))
    Invoke-Checked cargo (@("test") + $workspaceArgs + @("--locked"))

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $env:PYTHONDONTWRITEBYTECODE = "1"
        Invoke-Checked $python.Source @("-m", "compileall", "-q", "scripts")
    } else {
        Write-Warning "python was not found; skipping Python syntax checks."
    }

    if ($Tauri) {
        $appRoot = Join-Path $repoRoot "apps/tauri"
        Set-Location -LiteralPath $appRoot
        if (-not (Test-Path -LiteralPath "node_modules")) {
            Invoke-Checked npm.cmd @("ci")
        }
        Invoke-Checked npm.cmd @("test")
        Invoke-Checked npm.cmd @("run", "build")
    }

    Invoke-Checked git @("diff", "--check")
    Write-Host "Repository verification passed." -ForegroundColor Green
} finally {
    Set-Location -LiteralPath $initialDirectory
}
