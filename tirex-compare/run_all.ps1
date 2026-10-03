# Runs the three projects one after another (never in parallel, so the timings are clean),
# then the comparison. Expect about three hours on an 8-core CPU.
Set-Location $PSScriptRoot
foreach ($project in @("tirex-base", "tirex-aggressive-gather", "tirex-formal-selector")) {
    Write-Host "=== $project"
    & (Join-Path $PSScriptRoot "..\$project\run.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& .\run.ps1
exit $LASTEXITCODE
