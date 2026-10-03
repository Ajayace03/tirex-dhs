# Creates .venv in this folder with numpy, scipy, and matplotlib. Needs Python 3.11+ once.
Set-Location $PSScriptRoot
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv .venv } else { & python -m venv .venv }
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Error "Could not create .venv. Install Python 3.11+ from python.org first."
    exit 1
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Environment ready. Run .\run.ps1"
