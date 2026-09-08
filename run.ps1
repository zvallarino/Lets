# Use a working project environment, or the tested local Codex Python 3.12
# runtime with the existing project's Python 3.12 packages. No environment edits.
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$projectPackages = Join-Path $PSScriptRoot '.venv\Lib\site-packages'
$pipelineScript = Join-Path $PSScriptRoot 'run_pipeline.py'
$pipelineArguments = @($args)
if ($pipelineArguments.Count -eq 0) {
    $pipelineArguments = @((Join-Path $PSScriptRoot 'pictures\TIFF'), '-o', (Join-Path $PSScriptRoot 'output\review'))
}
$usable = $false
if (Test-Path -LiteralPath $projectPython) {
    & $projectPython -c 'import sys' 2>$null
    $usable = ($LASTEXITCODE -eq 0)
}
if ($usable) {
    & $projectPython $pipelineScript @pipelineArguments
} else {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (!(Test-Path -LiteralPath $bundledPython) -or !(Test-Path -LiteralPath $projectPackages)) {
        throw 'No working Python environment. Install Python 3.12 and create a new venv using requirements.txt.'
    }
    Write-Host 'Using bundled Python 3.12 with existing project packages; the old venv launcher is unavailable.'
    $bootstrap = 'import sys,runpy; from pathlib import Path; assert sys.version_info[:2]==(3,12), "Fallback requires Python 3.12"; sys.path.insert(0,sys.argv.pop(1)); script=sys.argv.pop(1); sys.path.insert(0,str(Path(script).parent)); sys.argv[0]=script; runpy.run_path(script,run_name="__main__")'
    & $bundledPython -c $bootstrap $projectPackages $pipelineScript @pipelineArguments
}
exit $LASTEXITCODE
