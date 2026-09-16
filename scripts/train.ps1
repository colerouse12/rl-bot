[CmdletBinding()]
param(
    [string]$Config = 'configs/1v1-cpu.json',
    [string]$CheckpointDirectory = '',
    [string]$MeshDirectory = '',
    [ValidateRange(0, 2147483647)][int]$MaxIterations = 0,
    [ValidateRange(1, 2147483647)][Nullable[int]]$MaxSeconds = $null,
    [ValidateRange(1, 9223372036854775807)][Nullable[long]]$TargetTimesteps = $null,
    [switch]$Resume,
    [switch]$CheckEnvironment,
    [switch]$ContractCheck,
    [switch]$ValidateConfig
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainerExe = Join-Path $projectRoot 'build/bin/Release/rl-bot-train.exe'
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $trainerExe)) { throw 'Build the trainer with scripts/build-training.ps1 first.' }
if (-not $MeshDirectory) {
    $MeshDirectory = Join-Path $projectRoot 'collision_meshes'
    if (-not (Test-Path -LiteralPath (Join-Path $MeshDirectory 'soccar'))) {
        $MeshDirectory = Join-Path $projectRoot '.venv/Lib/site-packages/rlgym/rocket_league/sim/collision_meshes'
    }
}
$pythonBase = & $pythonExe -c 'import sys; print(sys.base_prefix)'
if ($LASTEXITCODE -ne 0) { throw 'Could not locate the Python runtime.' }
$trainerArgs = @('--config', $Config, '--mesh-dir', $MeshDirectory)
if ($CheckpointDirectory) { $trainerArgs += @('--checkpoint-dir', $CheckpointDirectory) }
if ($PSBoundParameters.ContainsKey('MaxIterations')) { $trainerArgs += @('--max-iterations', [string]$MaxIterations) }
if ($PSBoundParameters.ContainsKey('MaxSeconds')) { $trainerArgs += @('--max-seconds', [string]$MaxSeconds) }
if ($PSBoundParameters.ContainsKey('TargetTimesteps')) { $trainerArgs += @('--target-timesteps', [string]$TargetTimesteps) }
if ($Resume) { $trainerArgs += '--resume' }
if ($CheckEnvironment) { $trainerArgs += '--check-environment' }
if ($ContractCheck) { $trainerArgs += '--contract-check' }
if ($ValidateConfig) { $trainerArgs += '--validate-config' }
$previousPythonHome = $env:PYTHONHOME
$previousPythonPath = $env:PYTHONPATH
$previousPath = $env:PATH
Push-Location $projectRoot
try {
    $env:PYTHONHOME = $pythonBase
    $env:PYTHONPATH = Join-Path $projectRoot '.venv/Lib/site-packages'
    $env:PATH = "$pythonBase;$env:PATH"
    & $trainerExe @trainerArgs
    if ($LASTEXITCODE -ne 0) { throw "Trainer failed with exit code $LASTEXITCODE." }
} finally {
    $env:PYTHONHOME = $previousPythonHome
    $env:PYTHONPATH = $previousPythonPath
    $env:PATH = $previousPath
    Pop-Location
}
