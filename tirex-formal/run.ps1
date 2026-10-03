# Runs the project with its own .venv. Extra arguments go to the program, e.g. .\run.ps1 --quick
Set-Location $PSScriptRoot
$env:TORCHDYNAMO_DISABLE = "1"
$env:PYTHONPATH = "src"
$python = ".\.venv\Scripts\python.exe"
# A .venv copied from another PC points at that PC's Python; rebuild it if it does not start.
$healthy = $false
if (Test-Path $python) {
    try { & $python -c "import torch, tirex2" *> $null; $healthy = ($LASTEXITCODE -eq 0) } catch { $healthy = $false }
}
if (-not $healthy) {
    & .\setup.ps1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $python -m tirex_formal @args
exit $LASTEXITCODE
