#Requires -Version 7.0
# Install the pinned native SDK in ignored build storage. No global PATH edits.
[CmdletBinding()]
param([string]$Destination)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not $Destination) {
    $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
    $Destination = Join-Path (Split-Path -Parent $scriptDirectory) "target/directml-sdk/package"
}
$sdkVersion = "1.15.4"
$headerCommit = "8700779fe7a09ea7a007cf3d7ab4293c78e41017"
$packageHash = "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9"
$headerHash = "21eb6af105498026ed91a6ea6deb6850c19cbf5302ec3001aee64f6ff5ec4ccd"

function Assert-Hash([string]$Path, [string]$Expected) {
    $file = [IO.File]::OpenRead($Path)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $actual = ([BitConverter]::ToString($sha.ComputeHash($file))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose(); $file.Dispose() }
    if ($actual -ne $Expected) { throw "SHA256 mismatch for $Path" }
}

$Destination = [IO.Path]::GetFullPath($Destination)
$archive = Join-Path (Split-Path -Parent $Destination) "package.zip"
$null = New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/$sdkVersion/microsoft.ai.directml.$sdkVersion.nupkg" -OutFile $archive
}
Assert-Hash $archive $packageHash
$requiredPackageFiles = @(
    "include/DirectML.h",
    "bin/x64-win/DirectML.dll",
    "bin/x64-win/DirectML.Debug.dll",
    "bin/x64-win/DirectML.lib",
    "LICENSE.txt",
    "LICENSE-CODE.txt",
    "ThirdPartyNotices.txt"
)
if (@($requiredPackageFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Destination $_)) }).Count) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    # Repair partial installations as well as a first install. The archive has
    # already passed the pinned hash check; existing package files are refreshed.
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, $Destination, $true)
}
$header = Join-Path $Destination "include/DirectMLX.h"
if (-not (Test-Path -LiteralPath $header)) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/microsoft/DirectML/$headerCommit/Libraries/DirectMLX.h" -OutFile $header
}
Assert-Hash $header $headerHash
foreach ($relative in $requiredPackageFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $Destination $relative))) { throw "SDK file missing: $relative" }
}
Write-Host "DirectML $sdkVersion installed at $Destination"
Write-Host "Optional override: DIRECTML_ROOT=$Destination"
