[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'dependencies/gigalearn.json') -Raw | ConvertFrom-Json
$sourceRoot = Join-Path $projectRoot 'artifacts/gigalearn'
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Create the Python 3.11 .venv from requirements-lock.txt first.' }
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    & git clone --no-checkout $manifest.repository $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw 'GigaLearn clone failed.' }
    & git -C $sourceRoot checkout --detach $manifest.commit
    if ($LASTEXITCODE -ne 0) { throw 'GigaLearn checkout failed.' }
}
$actualCommit = & git -C $sourceRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $manifest.commit) { throw 'GigaLearn checkout has the wrong revision. Preserve it and use the pinned source.' }
foreach ($tree in $manifest.trees.PSObject.Properties) {
    $actualTree = & git -C $sourceRoot rev-parse ($manifest.commit + ':' + $tree.Name)
    if ($LASTEXITCODE -ne 0 -or $actualTree -ne $tree.Value) { throw "Unexpected dependency tree: $($tree.Name)" }
}
& $pythonExe (Join-Path $PSScriptRoot 'patch_gigalearn.py') --source $sourceRoot
if ($LASTEXITCODE -ne 0) { throw 'GigaLearn integration patch failed.' }
Write-Host "Pinned source ready: $actualCommit"
