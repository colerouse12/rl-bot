[CmdletBinding()]
param(
    [ValidateRange(1, 32)][int]$Jobs = 1,
    [string]$TorchPrefix = '',
    [string]$BuildDirectory = 'build'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'bootstrap-training.ps1')
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
$pythonBase = & $pythonExe -c 'import sys; print(sys.base_prefix)'
if ($LASTEXITCODE -ne 0) { throw 'Python base lookup failed.' }
if (-not $TorchPrefix) {
    $TorchPrefix = & $pythonExe -c 'import torch; assert torch.version.cuda is None, "CPU Torch required"; print(torch.utils.cmake_prefix_path)'
    if ($LASTEXITCODE -ne 0) { throw 'CPU Torch lookup failed. Install requirements-training.txt into the Python 3.11 .venv, or supply -TorchPrefix for a CPU LibTorch distribution.' }
}
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCommand) { $cmakeExe = $cmakeCommand.Source } else {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio 2022 Build Tools with the C++ workload and CMake tools.' }
    $vsInstall = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $cmakeExe = Join-Path $vsInstall 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
    if (-not (Test-Path -LiteralPath $cmakeExe)) { throw 'CMake is missing. Add Microsoft.VisualStudio.Component.VC.CMake.Project.' }
}
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory = Join-Path $projectRoot $BuildDirectory }
& $cmakeExe -S $projectRoot -B $BuildDirectory -G 'Visual Studio 17 2022' -A x64 `
    "-DCMAKE_PREFIX_PATH=$TorchPrefix" "-DPython_EXECUTABLE=$pythonExe" "-DPython_ROOT_DIR=$pythonBase"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& $cmakeExe --build $BuildDirectory --config Release --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Training build failed.' }
$ctestExe = Join-Path (Split-Path -Parent $cmakeExe) 'ctest.exe'
& $ctestExe --test-dir $BuildDirectory -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Native build checks failed.' }
Write-Host "Trainer: $(Join-Path $BuildDirectory 'bin/Release/rl-bot-train.exe')"
