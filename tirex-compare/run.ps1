# Compares the three projects. They must sit next to this folder and have been run.
# Extra arguments go to compare.py, e.g. .\run.ps1 --projects D:\work
Set-Location $PSScriptRoot
$python = ".\.venv\Scripts\python.exe"
$healthy = $false
if (Test-Path $python) {
    try { & $python -c "import numpy, scipy, matplotlib" *> $null; $healthy = ($LASTEXITCODE -eq 0) } catch { $healthy = $false }
}
if (-not $healthy) {
    & .\setup.ps1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $python compare.py @args
exit $LASTEXITCODE
