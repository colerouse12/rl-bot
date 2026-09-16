[CmdletBinding()]
param(
    [string]$BuildToolsVersion = '17.14.41',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Component IDs: https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools?view=vs-2022
$components = @(
    'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
    'Microsoft.VisualStudio.Component.VC.CMake.Project',
    'Microsoft.VisualStudio.Component.Windows11SDK.26100'
)
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'

function Find-TrainingToolchain {
    if (-not (Test-Path -LiteralPath $vswhere)) { return $null }
    $query = @('-latest', '-products', '*', '-version', '[17.0,18.0)', '-requires') + $components + @('-property', 'installationPath')
    $installation = & $vswhere @query
    if (-not $installation) { return $null }
    $cmake = Join-Path $installation 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    $vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path -LiteralPath $cmake) -or -not (Test-Path -LiteralPath $vcvars)) { return $null }
    return [pscustomobject]@{
        VisualStudio = $installation
        CMake = $cmake
        VcVars64 = $vcvars
    }
}

$toolchain = Find-TrainingToolchain
if (-not $toolchain) {
    if ($CheckOnly) { throw 'Visual Studio 2022 C++ Build Tools, CMake, or Windows 11 SDK 26100 is missing. Run scripts/setup-toolchain.ps1 to install them.' }
    $winget = Get-Command winget.exe -ErrorAction Stop
    $installerArguments = '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools'
    foreach ($component in $components) { $installerArguments += " --add $component" }
    & $winget.Source install --id Microsoft.VisualStudio.2022.BuildTools --exact --version $BuildToolsVersion --source winget --accept-source-agreements --accept-package-agreements --disable-interactivity --override $installerArguments
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) {
        throw "Visual Studio installation failed (exit $LASTEXITCODE). Review the installer output. Windows administrator approval may be required."
    }
    $toolchain = Find-TrainingToolchain
    if (-not $toolchain) {
        throw 'The required components are still unavailable. Use Visual Studio Installer to add MSVC x64/x86, C++ CMake tools, and Windows 11 SDK 26100, then rerun this check. Restart Windows only if the installer requested it.'
    }
}

$toolchain
& $toolchain.CMake --version
if ($LASTEXITCODE -ne 0) { throw 'CMake did not start successfully.' }
