#Requires -Version 7.0
[CmdletBinding()]
param(
    [string]$CacheDirectory = (Join-Path $env:RUNNER_TEMP 'cuda-installer'),
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$cudaRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA/v13.4'
$required = @(
    'bin/nvcc.exe', 'include/cuda_runtime.h', 'include/cufft.h',
    'lib/x64/cudart.lib', 'lib/x64/cufft.lib',
    'bin/x64/cudart64_13.dll', 'bin/x64/cufft64_12.dll'
)

if (-not $ValidateOnly) {
    $null = New-Item -ItemType Directory -Path $CacheDirectory -Force
    $installer = Join-Path $CacheDirectory 'cuda_13.4.0_windows_x86_64.exe'
    if (-not (Test-Path -LiteralPath $installer)) {
        # NVIDIA's 13.4 download archive currently distributes this installer
        # from packages.nvidia.com; third-party action version tables lag it.
        $url = 'https://packages.nvidia.com/prerelease/cuda/13.4.0/local_installers/cuda_13.4.0_windows_x86_64.exe'
        & curl.exe --fail --location --retry 5 --retry-all-errors --output $installer $url
        if ($LASTEXITCODE -ne 0) { throw 'CUDA installer download failed' }
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=NVIDIA Corporation') {
        throw "CUDA installer NVIDIA signature verification failed: $($signature.Status)"
    }
    # Install compiler dependencies and cuFFT headers/import libraries too.
    # No display driver or unrelated inference libraries are installed.
    $arguments = @('-s', 'nvcc_13.4', 'crt_13.4', 'nvvm_13.4', 'cudart_13.4', 'cufft_13.4', 'cufft_dev_13.4')
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 3010)) { throw "CUDA installation failed: exit $($process.ExitCode)" }
}

foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $cudaRoot $relative))) {
        throw "CUDA SDK is missing $relative in $cudaRoot"
    }
}
# The hosted CUDA installer does not consistently copy the top-level notices
# that exist in the full local toolkit. Keep the distributed notice set
# explicit and traceable to NVIDIA's current CUDA EULA page.
$noticeRoot = Join-Path $env:RUNNER_TEMP 'cuda-notices'
$null = New-Item -ItemType Directory -Path $noticeRoot -Force
$eula = Join-Path $noticeRoot 'EULA.html'
& curl.exe --fail --location --retry 5 --retry-all-errors --output $eula 'https://docs.nvidia.com/cuda/eula/index.html'
if ($LASTEXITCODE -ne 0) { throw 'CUDA EULA download failed' }
@(
    'CUDA Toolkit 13.4 hosted-build notice',
    'Runtime files are installed from the NVIDIA CUDA Toolkit 13.4 Windows installer.',
    'License text: https://docs.nvidia.com/cuda/eula/index.html'
) | Set-Content -LiteralPath (Join-Path $noticeRoot 'README') -Encoding utf8
if ($env:GITHUB_ENV) { "CUDA_NOTICE_ROOT=$noticeRoot" | Out-File -LiteralPath $env:GITHUB_ENV -Append -Encoding utf8 }
$env:CUDA_PATH = $cudaRoot
if ($env:GITHUB_ENV) { "CUDA_PATH=$cudaRoot" | Out-File -LiteralPath $env:GITHUB_ENV -Append -Encoding utf8 }
if ($env:GITHUB_PATH) {
    (Join-Path $cudaRoot 'bin') | Out-File -LiteralPath $env:GITHUB_PATH -Append -Encoding utf8
    (Join-Path $cudaRoot 'bin/x64') | Out-File -LiteralPath $env:GITHUB_PATH -Append -Encoding utf8
}
& (Join-Path $cudaRoot 'bin/nvcc.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'CUDA compiler verification failed' }
