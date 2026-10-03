[CmdletBinding()]
param(
    [switch]$IncludeOnnx,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $repoRoot "models\shared\manifest.json"
$manifest = Get-Content -Raw $manifestPath | ConvertFrom-Json

function Get-RelativePath($file) {
    return Join-Path $repoRoot (Join-Path "models\shared" $file.path)
}

function Get-Sha256($path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($path)
        try {
            return ([System.BitConverter]::ToString($sha.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha.Dispose()
    }
}

function Test-File($file, $path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        return $false
    }
    if ($null -ne $file.bytes) {
        $actualLength = (Get-Item -LiteralPath $path).Length
        if ($actualLength -ne [int64]$file.bytes) {
            Write-Warning "Size mismatch for $($file.path): got $actualLength, expected $($file.bytes)"
            return $false
        }
    }
    if ($file.sha256) {
        $actualHash = Get-Sha256 $path
        if ($actualHash -ne $file.sha256.ToLowerInvariant()) {
            Write-Warning "SHA-256 mismatch for $($file.path): got $actualHash, expected $($file.sha256)"
            return $false
        }
    }
    return $true
}

foreach ($file in $manifest.files) {
    if ($file.path.EndsWith(".onnx") -and -not $IncludeOnnx) {
        continue
    }

    $destination = Get-RelativePath $file
    $destinationDir = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force $destinationDir | Out-Null

    if (Test-File $file $destination) {
        Write-Host "verified $($file.path)"
        continue
    }
    if ($VerifyOnly) {
        throw "missing or invalid model: $($file.path)"
    }

    Write-Host "downloading $($file.path)"
    & curl.exe --fail --location --retry 3 --retry-delay 3 --continue-at - --output $destination $file.url
    if ($LASTEXITCODE -ne 0) {
        throw "download failed for $($file.path)"
    }
    if (-not (Test-File $file $destination)) {
        throw "downloaded file failed verification: $($file.path)"
    }
    Write-Host "verified $($file.path)"
}

Write-Host "shared model preparation complete"
