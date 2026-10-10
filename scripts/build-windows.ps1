#Requires -Version 7.0
# Build a Windows x64 installer or an embedded-frontend debug application.
# The default TensorRt distribution also includes DirectML as a GPU fallback.
[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Profile = "Release",
    [ValidateSet("TensorRt", "DirectMl", "Cpu")]
    [string]$Backend = "TensorRt",
    [string]$TensorRtRoot,
    [string]$CudaRoot,
    [string]$DirectMlRoot,
    [string]$CpuRoot,
    [string]$TargetDirectory,
    [switch]$NoBundle,
    [switch]$Launch,
    [ValidateRange(0, 65535)]
    [int]$InspectPort = 0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not $IsWindows -or -not [Environment]::Is64BitProcess) {
    throw "Run this script with 64-bit PowerShell 7 on Windows."
}
if ($InspectPort -ne 0 -and ($Profile -ne "Debug" -or -not $Launch)) {
    throw "InspectPort requires -Profile Debug -Launch."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$appRoot = Join-Path $repoRoot "apps/tauri"
$defaultTarget = Join-Path $appRoot "src-tauri/target"
$selectedTarget = if ($TargetDirectory) { $TargetDirectory } elseif ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { $defaultTarget }
$cargoTarget = [IO.Path]::GetFullPath($selectedTarget, $repoRoot)
$outputDir = Join-Path $cargoTarget $Profile.ToLowerInvariant()
$reportDir = if ($cargoTarget -eq $defaultTarget) {
    Join-Path $repoRoot "target/windows-build/$Profile-$Backend"
} else {
    Join-Path $cargoTarget "windows-build/$Profile-$Backend"
}
$null = New-Item -ItemType Directory -Path $reportDir -Force

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments 2>&1 |
        Tee-Object -FilePath (Join-Path $reportDir "commands.log") -Append | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$Program exited with code $LASTEXITCODE"
    }
}

function Resolve-SdkRoot([string]$Explicit, [string[]]$Variables, [string]$Fallback) {
    if ($Explicit) { return (Resolve-Path -LiteralPath $Explicit).Path }
    foreach ($variable in $Variables) {
        $value = [Environment]::GetEnvironmentVariable($variable)
        if ($value) { return (Resolve-Path -LiteralPath $value).Path }
    }
    return (Resolve-Path -LiteralPath $Fallback).Path
}

function Copy-RuntimeFile([string]$Source, [string]$Destination) {
    if ((Test-Path -LiteralPath $Destination) -and
        (Get-FileHash -LiteralPath $Destination).Hash -eq (Get-FileHash -LiteralPath $Source).Hash) {
        return
    }
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

[string[]]$features = @(switch ($Backend) {
    "TensorRt" { @("tensorrt-rtx", "directml", "cpu-graph") }
    "DirectMl" { @("directml", "cpu-graph") }
    "Cpu" { @("cpu-graph") }
})
$includeDirectMl = $Backend -in @("TensorRt", "DirectMl")

$initialDirectory = Get-Location
Start-Transcript -LiteralPath (Join-Path $reportDir "build.log") -Force | Out-Null
try {
    # Initialize the compiler, SDK and redistributable discovery for nvcc and
    # Tauri's dynamic CRT bundler, without changing the user's global environment.
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
        $vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -ne 0 -or -not $vsRoot) {
            throw "Visual Studio C++ build tools and the Windows SDK are required."
        }
        & (Join-Path $vsRoot "Common7/Tools/Launch-VsDevShell.ps1") -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
    }
    $env:CARGO_TARGET_DIR = $cargoTarget
    $env:CARGO_INCREMENTAL = "0"
    $runtimeFiles = @()
    $noticeFiles = @()
    $cpuNoticeFiles = @()
    $trtNoticeFiles = @()
    $cudaNoticeFiles = @()
    $resources = [ordered]@{}
    if ($features -contains "cpu-graph") {
        $defaultCpuRoot = Join-Path $env:LOCALAPPDATA "vocal-fast/cpu-operators/onednn"
        if (-not $CpuRoot -and -not $env:VOCAL_CPU_ONEDNN_ROOT -and -not $env:ONEDNN_ROOT -and -not (Test-Path (Join-Path $defaultCpuRoot "include/oneapi/dnnl/dnnl.hpp"))) {
            Push-Location $repoRoot
            try { & (Join-Path $PSScriptRoot "setup-cpu.ps1") } finally { Pop-Location }
        }
        $CpuRoot = Resolve-SdkRoot $CpuRoot @("VOCAL_CPU_ONEDNN_ROOT", "ONEDNN_ROOT") $defaultCpuRoot
        $env:VOCAL_CPU_ONEDNN_ROOT = $CpuRoot
        foreach ($relative in @("include/oneapi/dnnl/dnnl.hpp", "lib/dnnl.lib", "bin/dnnl.dll", "share/doc/dnnl/LICENSE", "share/doc/dnnl/THIRD-PARTY-PROGRAMS")) {
            if (-not (Test-Path -LiteralPath (Join-Path $CpuRoot $relative))) {
                throw "CPU operator SDK file missing: $relative in $CpuRoot"
            }
        }
        $runtimeFiles += Get-Item -LiteralPath (Join-Path $CpuRoot "bin/dnnl.dll")
        foreach ($name in @('LICENSE', 'THIRD-PARTY-PROGRAMS')) {
            $cpuNoticeFiles += Get-Item -LiteralPath (Join-Path $CpuRoot "share/doc/dnnl/$name")
        }
        Write-Host "CPU operator SDK: $CpuRoot"
    }
    if ($Backend -eq "TensorRt") {
        $TensorRtRoot = Resolve-SdkRoot $TensorRtRoot @("TENSORRT_RTX_ROOT", "TRT_RTX_ROOT", "TENSORRT_ROOT", "TRT_ROOT") "C:/Program Files/NVIDIA Corporation/TensorRT-RTX/1.6.1.120"
        $CudaRoot = Resolve-SdkRoot $CudaRoot @("CUDA_PATH") "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4"
        $env:TENSORRT_RTX_ROOT = $TensorRtRoot
        $env:CUDA_PATH = $CudaRoot
        $env:VOCAL_TRT_COPY_DLLS = "1"
        foreach ($relative in @("doc/Acknowledgements.txt", "doc/README.txt")) {
            if (-not (Test-Path -LiteralPath (Join-Path $TensorRtRoot $relative))) {
                throw "TensorRT-RTX notice file missing: $relative in $TensorRtRoot"
            }
            $trtNoticeFiles += Get-Item -LiteralPath (Join-Path $TensorRtRoot $relative)
        }
        foreach ($name in @("EULA.txt", "LICENSE", "README")) {
            if (-not (Test-Path -LiteralPath (Join-Path $CudaRoot $name))) {
                throw "CUDA notice file missing: $name in $CudaRoot"
            }
            $cudaNoticeFiles += Get-Item -LiteralPath (Join-Path $CudaRoot $name)
        }
        # Never bundle all DLLs left in a previous target directory. These are
        # the linked runtime dependencies; weights and plans stay user-managed.
        foreach ($name in @("tensorrt_rtx_1_6.dll", "tensorrt_onnxparser_rtx_1_6.dll")) {
            $runtimeFiles += Get-Item -LiteralPath (Join-Path $TensorRtRoot "bin/$name")
        }
        foreach ($pattern in @("cudart64_*.dll", "cufft64_*.dll")) {
            $matches = @(Get-ChildItem -LiteralPath (Join-Path $CudaRoot "bin/x64") -Filter $pattern -File)
            if ($matches.Count -ne 1) {
                throw "Expected exactly one $pattern in CUDA bin/x64; found $($matches.Count)."
            }
            $runtimeFiles += $matches
        }
        Write-Host "TensorRT SDK: $TensorRtRoot"
        Write-Host "CUDA SDK: $CudaRoot"
    }
    if ($includeDirectMl) {
        $defaultDirectMlRoot = Join-Path $repoRoot "target/directml-sdk/package"
        if (-not $DirectMlRoot -and -not $env:DIRECTML_ROOT) {
            # Provision only the pinned, hash-checked local SDK. Explicit SDK
            # overrides are validated below rather than silently replaced.
            $required = @("include/DirectML.h", "include/DirectMLX.h", "bin/x64-win/DirectML.lib", "bin/x64-win/DirectML.dll", "bin/x64-win/DirectML.Debug.dll", "LICENSE.txt", "LICENSE-CODE.txt", "ThirdPartyNotices.txt")
            if (@($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $defaultDirectMlRoot $_)) }).Count) {
                & (Join-Path $PSScriptRoot "setup-directml.ps1") -Destination $defaultDirectMlRoot
            }
        }
        $DirectMlRoot = Resolve-SdkRoot $DirectMlRoot @("DIRECTML_ROOT") (Join-Path $repoRoot "target/directml-sdk/package")
        $env:DIRECTML_ROOT = $DirectMlRoot
        foreach ($relative in @("include/DirectML.h", "include/DirectMLX.h", "bin/x64-win/DirectML.lib")) {
            if (-not (Test-Path -LiteralPath (Join-Path $DirectMlRoot $relative))) {
                throw "DirectML SDK file missing: $relative in $DirectMlRoot"
            }
        }
        $runtimeFiles += Get-Item -LiteralPath (Join-Path $DirectMlRoot "bin/x64-win/DirectML.dll")
        if ($Profile -eq "Debug") {
            $runtimeFiles += Get-Item -LiteralPath (Join-Path $DirectMlRoot "bin/x64-win/DirectML.Debug.dll")
        }
        foreach ($name in @("LICENSE.txt", "LICENSE-CODE.txt", "ThirdPartyNotices.txt")) {
            $noticeFiles += Get-Item -LiteralPath (Join-Path $DirectMlRoot $name)
        }
        Write-Host "DirectML SDK: $DirectMlRoot"
    }

    # Keep an explicit resource allowlist; old target DLLs must not leak into
    # installers. System D3D12/DXGI and GPU drivers are supplied by Windows.
    $stageDir = Join-Path $cargoTarget "windows-runtime/$Profile-$Backend"
    $null = New-Item -ItemType Directory -Path $stageDir -Force
    $tauriRoot = Join-Path $appRoot "src-tauri"
    if ([IO.Path]::GetPathRoot($stageDir) -ne [IO.Path]::GetPathRoot($tauriRoot)) {
        # Tauri resource normalization drops Windows drive prefixes. A relative
        # junction keeps cross-drive staging valid without copying it back.
        $stageHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData(
            [Text.Encoding]::UTF8.GetBytes($stageDir.ToUpperInvariant())
        )).Substring(0, 16)
        $linkPath = Join-Path $defaultTarget "windows-runtime-external/$Profile-$Backend-$stageHash"
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $linkPath) -Force
        if (Test-Path -LiteralPath $linkPath) {
            $link = Get-Item -LiteralPath $linkPath
            if ($link.LinkType -ne "Junction" -or $link.Target -ne $stageDir) {
                throw "Runtime resource junction has an unexpected target: $linkPath"
            }
        } else {
            $null = New-Item -ItemType Junction -Path $linkPath -Target $stageDir
        }
        $stageResourcePath = [IO.Path]::GetRelativePath($tauriRoot, $linkPath).Replace('\', '/')
    } else {
        $stageResourcePath = [IO.Path]::GetRelativePath($tauriRoot, $stageDir).Replace('\', '/')
    }
    foreach ($file in $runtimeFiles) {
        Copy-RuntimeFile $file.FullName (Join-Path $stageDir $file.Name)
        $resources["$stageResourcePath/$($file.Name)"] = $file.Name
    }
    foreach ($file in $noticeFiles) {
        $relative = "licenses/directml/$($file.Name)"
        Copy-RuntimeFile $file.FullName (Join-Path $stageDir $relative)
        $resources["$stageResourcePath/$relative"] = $relative
    }
    foreach ($file in $cpuNoticeFiles) {
        $relative = "licenses/cpu/$($file.Name)"
        Copy-RuntimeFile $file.FullName (Join-Path $stageDir $relative)
        $resources["$stageResourcePath/$relative"] = $relative
    }
    foreach ($file in $trtNoticeFiles) {
        $relative = "licenses/tensorrt-rtx/$($file.Name)"
        Copy-RuntimeFile $file.FullName (Join-Path $stageDir $relative)
        $resources["$stageResourcePath/$relative"] = $relative
    }
    foreach ($file in $cudaNoticeFiles) {
        $relative = "licenses/cuda/$($file.Name)"
        Copy-RuntimeFile $file.FullName (Join-Path $stageDir $relative)
        $resources["$stageResourcePath/$relative"] = $relative
    }

    $overridePath = Join-Path $reportDir "tauri.runtime.conf.json"
    @{ bundle = @{ resources = $resources } } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $overridePath -Encoding utf8NoBOM
    Set-Location -LiteralPath $appRoot
    if (-not (Test-Path "node_modules/@tauri-apps/cli/tauri.js")) {
        Invoke-Checked npm.cmd @("ci")
    }
    $buildArgs = @("node_modules/@tauri-apps/cli/tauri.js", "build", "--ci", "--config", $overridePath)
    if ($Profile -eq "Debug") { $buildArgs += "--debug" }
    # Default Cargo builds include DML. Select the distribution's exact matrix
    # explicitly so the CPU package still builds without either GPU SDK.
    if ($features) { $buildArgs += @("--features", ($features -join ",")) }
    if ($NoBundle) { $buildArgs += "--no-bundle" } else { $buildArgs += @("--bundles", "nsis") }
    $buildArgs += @("--", "--locked", "--no-default-features")
    Invoke-Checked node $buildArgs

    $binaryPath = Join-Path $outputDir "vocal-fast-tauri.exe"
    if (-not (Test-Path -LiteralPath $binaryPath)) { throw "Build did not produce $binaryPath" }
    # Also refresh loose binaries when Cargo reused its cached build script.
    foreach ($file in $runtimeFiles) {
        Copy-RuntimeFile $file.FullName (Join-Path $outputDir $file.Name)
    }
    foreach ($file in $noticeFiles) {
        Copy-RuntimeFile $file.FullName (Join-Path $outputDir "licenses/directml/$($file.Name)")
    }
    foreach ($file in $cpuNoticeFiles) {
        Copy-RuntimeFile $file.FullName (Join-Path $outputDir "licenses/cpu/$($file.Name)")
    }
    foreach ($file in $trtNoticeFiles) {
        Copy-RuntimeFile $file.FullName (Join-Path $outputDir "licenses/tensorrt-rtx/$($file.Name)")
    }
    foreach ($file in $cudaNoticeFiles) {
        Copy-RuntimeFile $file.FullName (Join-Path $outputDir "licenses/cuda/$($file.Name)")
    }
    $artifacts = @((Get-Item -LiteralPath $binaryPath))
    if (-not $NoBundle) {
        $installers = @(Get-ChildItem -LiteralPath (Join-Path $outputDir "bundle/nsis") -Filter '*-setup.exe' -File)
        if (-not $installers) { throw "NSIS did not produce an installer." }
        $artifacts += $installers
    }
    $manifest = [ordered]@{
        profile = $Profile
        backend = $Backend
        features = @($features)
        targetDirectory = $cargoTarget
        builtAt = [DateTimeOffset]::UtcNow.ToString("o")
        artifacts = @($artifacts | ForEach-Object {
            @{ path = $_.FullName; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
        })
        runtimeFiles = @($runtimeFiles | ForEach-Object {
            @{ name = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
        })
        licenseFiles = @($noticeFiles + $cpuNoticeFiles + $trtNoticeFiles + $cudaNoticeFiles | ForEach-Object {
            @{ name = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
        })
        resourceDestinations = @($resources.Values)
    }
    $manifest | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $reportDir "artifacts.json") -Encoding utf8NoBOM
    $artifacts | Select-Object FullName, Length | Format-Table -AutoSize

    if ($Launch) {
        $env:RUST_BACKTRACE = "1"
        if ($InspectPort -ne 0) {
            $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = "--remote-debugging-address=127.0.0.1 --remote-debugging-port=$InspectPort"
        }
        $process = Start-Process -FilePath $binaryPath -WorkingDirectory $outputDir -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $reportDir "stdout.log") `
            -RedirectStandardError (Join-Path $reportDir "stderr.log")
        $deadline = [DateTime]::UtcNow.AddSeconds(45)
        do {
            Start-Sleep -Milliseconds 250
            $process.Refresh()
            if ($process.HasExited) { throw "GUI exited with code $($process.ExitCode); see $reportDir/stderr.log" }
        } while (($process.MainWindowHandle -eq 0 -or -not $process.MainWindowTitle) -and [DateTime]::UtcNow -lt $deadline)
        if ($process.MainWindowHandle -eq 0 -or -not $process.MainWindowTitle) {
            throw "GUI did not show a titled window within 45 seconds (PID $($process.Id))."
        }
        @{ pid = $process.Id; binary = $binaryPath; windowTitle = $process.MainWindowTitle; responding = $process.Responding } |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportDir "instance.json") -Encoding utf8NoBOM
        Write-Host "GUI running: PID $($process.Id), window '$($process.MainWindowTitle)'"
    }
} finally {
    Set-Location $initialDirectory
    Stop-Transcript | Out-Null
}
