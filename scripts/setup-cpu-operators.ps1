param(
    [string]$CacheRoot = (Join-Path $env:LOCALAPPDATA 'vocal-fast/cpu-operators'),
    [string]$TorchRoot = 'target/cpu-sdk/torch',
    [int]$Jobs = 8,
    [switch]$SdkOnly,
    [switch]$Int8BenchOnly
)
$ErrorActionPreference = 'Stop'
if ($SdkOnly -and $Int8BenchOnly) { throw 'SdkOnly and Int8BenchOnly are mutually exclusive' }
$repo = Split-Path $PSScriptRoot -Parent
$cache = [IO.Path]::GetFullPath($CacheRoot)
New-Item -ItemType Directory -Path $cache -Force | Out-Null
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & rtk proxy $vswhere -latest -property installationPath
if (-not $vs) { throw 'MSVC installation not found' }
$vcvars = Join-Path $vs.Trim() 'VC/Auxiliary/Build/vcvars64.bat'
$vcEnvironment = & rtk proxy cmd /d /c "`"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed' }
foreach ($entry in $vcEnvironment) {
    if ($entry -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$source = Join-Path $cache 'onednn-3.10.2'
if (-not (Test-Path (Join-Path $source 'CMakeLists.txt'))) {
    & rtk proxy git clone --depth 1 --branch v3.10.2 https://github.com/uxlfoundation/oneDNN.git $source
    if ($LASTEXITCODE -ne 0) { throw 'oneDNN download failed' }
}
$build = Join-Path $cache 'onednn-build'
$install = Join-Path $cache 'onednn'
& rtk proxy cmake -S $source -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$install" `
    -DDNNL_CPU_RUNTIME=THREADPOOL -DDNNL_GPU_RUNTIME=NONE -DONEDNN_BUILD_GRAPH=OFF `
    -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_DOC=OFF `
    -DDNNL_ENABLE_WORKLOAD=INFERENCE -DDNNL_ENABLE_PRIMITIVE_CPU_ISA=AVX2 `
    '-DDNNL_ENABLE_PRIMITIVE=CONVOLUTION;DECONVOLUTION;MATMUL;INNER_PRODUCT;RNN;REORDER;ELTWISE;LAYER_NORMALIZATION;SOFTMAX' `
    -DDNNL_ENABLE_JIT_PROFILING=OFF -DDNNL_ENABLE_ITT_TASKS=OFF
if ($LASTEXITCODE -ne 0) { throw 'oneDNN configure failed' }
& rtk proxy cmake --build $build --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'oneDNN build failed' }
& rtk proxy cmake --install $build
if ($LASTEXITCODE -ne 0) { throw 'oneDNN install failed' }
if (-not $SdkOnly) {
    $benchmarkBuild = Join-Path $cache 'benchmark-build'
    $providerArgs = @('-DVOCAL_BENCH_LEGACY_PROVIDERS=OFF')
    if (-not $Int8BenchOnly) {
        $torchSdk = (Resolve-Path (Join-Path $repo $TorchRoot)).Path
        $providerArgs = @('-DVOCAL_BENCH_LEGACY_PROVIDERS=ON', "-DTORCH_SDK=$torchSdk")
    }
    & rtk proxy cmake -S (Join-Path $repo 'scripts/cpu-operators') -B $benchmarkBuild -G Ninja `
        -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$install" @providerArgs
    if ($LASTEXITCODE -ne 0) { throw 'operator benchmark configure failed' }
    & rtk proxy cmake --build $benchmarkBuild --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'operator benchmark build failed' }
    Write-Output "Benchmark libraries: $benchmarkBuild"
}
Write-Output "oneDNN SDK: $install"
