param(
    [string]$Pipeline = "target/trt-pipeline/pipeline.json",
    [string]$Model = "target/tsukuyomi-canonical.vfr",
    [string]$Audio = "target/input.f32",
    [string]$Output = "target/nsys-rvc",
    [int]$Warmup = 10,
    [int]$Iterations = 100,
    [string]$Binary = "target/debug/vocal-fast.exe",
    [ValidateSet("default", "cufft", "radix2")]
    [string]$RmvpeFft = "default"
)

$ErrorActionPreference = "Stop"
$nsys = Get-ChildItem "$env:ProgramFiles/NVIDIA Corporation/Nsight Systems*/target-windows-x64/nsys.exe" |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $nsys) {
    throw "Nsight Systems nsys.exe was not found under $env:ProgramFiles/NVIDIA Corporation"
}
$binaryPath = (Resolve-Path $Binary).Path
$pipelinePath = (Resolve-Path $Pipeline).Path
$modelPath = (Resolve-Path $Model).Path
$inputPath = (Resolve-Path $Audio).Path

$trtRoot = if ($env:TRT_RTX_ROOT) {
    $env:TRT_RTX_ROOT
} else {
    "C:/Program Files/NVIDIA Corporation/TensorRT-RTX/1.6.1.120"
}
$trtBin = Join-Path $trtRoot "bin"
$cudaRoot = if ($env:CUDA_PATH) {
    $env:CUDA_PATH
} else {
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4"
}
$cudaBin = Join-Path $cudaRoot "bin"
$cudaBinX64 = Join-Path $cudaBin "x64"
$env:PATH = "$trtBin;$cudaBinX64;$cudaBin;$env:PATH"
$env:VOCAL_TRT_WARMUP = "$Warmup"
$env:VOCAL_TRT_PIPELINE_ITERATIONS = "$Iterations"
$env:VOCAL_TRT_DEBUG_INTERMEDIATES = "0"
$env:VOCAL_TRT_LAYER_PROFILE = "0"
$env:VOCAL_TRT_NATIVE_CUDA_GRAPH = "1"
if ($RmvpeFft -eq "default") {
    Remove-Item Env:VOCAL_TRT_RMVPE_FFT -ErrorAction SilentlyContinue
} else {
    $env:VOCAL_TRT_RMVPE_FFT = $RmvpeFft
}

# Node-level graph tracing keeps device kernels visible after TensorRT-RTX has
# switched from capture to replay. Without it, Nsight often records only the
# first two graph launches while the host NVTX ranges continue for all 100
# iterations, making the kernel summary look like a JIT hotspot.
& $nsys profile --trace=cuda,nvtx --cuda-trace-scope=process-tree --cuda-event-trace=true `
    --cuda-graph-trace=node `
    --sample=none --stats=true --force-overwrite=true `
    --output=$Output $binaryPath model profile-trt-pipeline $pipelinePath $modelPath $inputPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host "Nsight report written with prefix: $Output"
