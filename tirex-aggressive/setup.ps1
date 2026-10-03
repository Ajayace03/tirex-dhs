# Creates .venv in this folder and installs the pinned packages.
# Needs Python 3.11 or newer (the "py" launcher or "python" on PATH) and internet access once.
Set-Location $PSScriptRoot
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv .venv } else { & python -m venv .venv }
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Error "Could not create .venv. Install Python 3.11+ from python.org first."
    exit 1
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Exact versions failed to install; installing the top-level requirements instead."
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "Environment ready. Run .\run.ps1"
