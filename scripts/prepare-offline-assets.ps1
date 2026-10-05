[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$assetRoot = Join-Path $repoRoot "target\test-assets\mandarin-rvc"
$modelUrl = "https://huggingface.co/maotianchen/maobailing_rvc/resolve/main/maobailing_rvc.pth"
$modelSha256 = "c386a153eace55a15cc50b02d102732fa08c1c6a144184be3424c2188a46867c"
$audioUrl = "https://raw.githubusercontent.com/liangstein/Chinese-speech-to-text/master/1.wav"
$audioSha256 = "30a610eb2027bc2cb59b88da6ad7ab0c2e39ca3e342590ebb455d490b4248e25"

New-Item -ItemType Directory -Force $assetRoot | Out-Null

function Get-Sha256([string]$Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
        } finally {
            $stream.Dispose()
        }
    } finally {
        $algorithm.Dispose()
    }
}

function Get-VerifiedFile([string]$Url, [string]$Path, [string]$Sha256) {
    if ($Force -or -not (Test-Path -LiteralPath $Path -PathType Leaf) -or (Get-Sha256 $Path) -ne $Sha256) {
        Write-Host "downloading $Url"
        $partial = "$Path.part"
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        & curl.exe --fail --location --retry 3 --retry-delay 2 --connect-timeout 20 `
            --speed-time 20 --speed-limit 1024 --max-time 300 --output $partial $Url
        if ($LASTEXITCODE -ne 0) {
            throw "download failed: $Url"
        }
        Move-Item -LiteralPath $partial -Destination $Path -Force
    }
    $actual = Get-Sha256 $Path
    if ($actual -ne $Sha256) {
        throw "SHA-256 mismatch for ${Path}: got $actual, expected $Sha256"
    }
    Write-Host "verified $Path"
}

$audioPath = Join-Path $assetRoot "chinese-input.wav"
$modelPath = Join-Path $assetRoot "maobailing_rvc.pth"
Get-VerifiedFile $audioUrl $audioPath $audioSha256
Get-VerifiedFile $modelUrl $modelPath $modelSha256

$manifest = @{
    id = "maobailing-rvc-v2-40k"
    display_name = "maobailing RVC v2 Chinese female"
    architecture = "rvc-v2"
    version = "v2"
    sample_rate = 40000
    model_file = "maobailing_rvc.pth"
    index_file = $null
    source = "https://huggingface.co/maotianchen/maobailing_rvc"
    source_revision = "main"
    license = "Apache-2.0 (model card; see LICENSE)"
    notes = "Mandarin female voice model for local offline conversion tests."
    feature_model = "hubert_base"
    pitch_model = "rmvpe"
} | ConvertTo-Json -Depth 3
$manifestPath = Join-Path $assetRoot "manifest.json"
[System.IO.File]::WriteAllText(
    $manifestPath,
    $manifest,
    [System.Text.UTF8Encoding]::new($false)
)

$licensePath = Join-Path $assetRoot "LICENSE"
if ($Force -or -not (Test-Path -LiteralPath $licensePath)) {
    & curl.exe --fail --location --retry 3 --output $licensePath `
        "https://huggingface.co/maotianchen/maobailing_rvc/resolve/main/LICENSE"
    if ($LASTEXITCODE -ne 0) {
        throw "could not download model license"
    }
}

$cachePath = Join-Path $assetRoot "maobailing-rvc.vfr"
Write-Host "converting RVC checkpoint to VFRM cache"
& cargo run -p vocal-fast -- model convert-pth $assetRoot $manifestPath $modelPath $cachePath
if ($LASTEXITCODE -ne 0) {
    throw "RVC checkpoint conversion failed"
}

Write-Host "prepared assets in $assetRoot"
Write-Host "input: $audioPath"
Write-Host "model: $cachePath"
Write-Host "run: cargo run -p vocal-fast -- model run-pipeline passthrough $cachePath $audioPath target/mandarin-passthrough.wav"
Write-Host "run: cargo run -p vocal-fast -- model run-pipeline cpu $cachePath $audioPath target/mandarin-cpu.wav"
Write-Host "TensorRT additionally needs a compiled pipeline.json and --features tensorrt-rtx."
