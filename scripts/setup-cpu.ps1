param(
    [string]$CacheRoot = (Join-Path $env:LOCALAPPDATA 'vocal-fast/cpu-operators'),
    [int]$Jobs = 8
)
$ErrorActionPreference = 'Stop'
# Standalone oneDNN, THREADPOOL runtime, inference primitives, no GPU or Torch.
& (Join-Path $PSScriptRoot 'setup-cpu-operators.ps1') -CacheRoot $CacheRoot -Jobs $Jobs -SdkOnly
if ($LASTEXITCODE -ne 0) { throw 'CPU operator SDK installation failed' }
$sdk = Join-Path $CacheRoot 'onednn'
if (-not (Test-Path (Join-Path $sdk 'include/oneapi/dnnl/dnnl.hpp'))) { throw 'CPU SDK headers missing' }
Write-Output "CPU operator SDK: $sdk"
Write-Output 'Override: VOCAL_CPU_ONEDNN_ROOT; build: cargo build -p vocal-fast --features cpu-graph'
