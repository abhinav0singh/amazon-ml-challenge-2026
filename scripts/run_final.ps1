<#
run_final.ps1 -- the ONE command for the final full-scale run (Windows).

    powershell -ExecutionPolicy Bypass -File scripts\run_final.ps1 -Data D:\amlc\student_resource\dataset

-Data is the folder that holds train\ and test\ (keep it OUTSIDE OneDrive).

Resumable: if it stops for ANY reason (crash, reboot, closed window), run the
exact same command again. Finished stages are reused; nothing is recomputed.
See docs/RUN_FINAL.md.
#>
param(
    [Parameter(Mandatory = $true)][string]$Data,
    [int]$Workers = 3
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"

function Check($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $what (exit code $LASTEXITCODE). Do NOT improvise -- send work\final_run.log to Abhinav." -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path (Join-Path $Data "train\train_source1.tsv")) -or
    -not (Test-Path (Join-Path $Data "test\test_source1.tsv"))) {
    Write-Host "Wrong -Data folder: '$Data' must contain train\ and test\." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $py)) {
    Write-Host "Creating .venv with Python 3.12 (first time only) ..."
    py -3.12 -m venv .venv
    Check "create venv -- is Python 3.12 installed? try: py -3.12 --version"
    & $py -m pip install --upgrade pip
    Check "pip upgrade"
    & $py -m pip install -r requirements.txt
    Check "pip install -r requirements.txt"
}
& $py -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
Check "the venv must be Python 3.12"
& $py src\metric.py
Check "metric self-test"
& $py -m pytest -q tests
Check "unit tests"

& $py -u src\run_pipeline.py --data $Data --out output --work work --loco --block-workers $Workers --log work\final_run.log
Check "pipeline"
& $py scripts\make_handoff.py
Check "handoff bundle"
Write-Host "ALL DONE. Send the zip in handoff\ to Abhinav. Do not upload it yourself." -ForegroundColor Green
