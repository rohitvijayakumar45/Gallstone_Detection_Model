# PowerShell equivalent of regenerate_all_numbers.sh
# Run from repo root:  powershell -ExecutionPolicy Bypass -File scripts\regenerate_all_numbers.ps1

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step([string]$name, [scriptblock]$cmd) {
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "== $name"
    Write-Host ("-" * 78)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $cmd
    $sw.Stop()
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> FAIL ($LASTEXITCODE) in $($sw.Elapsed.TotalSeconds) s" -ForegroundColor Red
    } else {
        Write-Host "  -> OK in $($sw.Elapsed.TotalSeconds) s" -ForegroundColor Green
    }
}

Step "unit tests"       { python -m pytest tests/ -q --ignore=tests/test_smoke.py }
Step "manifest"         { python scripts/generate_manifest.py }
Step "baseline"         { python scripts/comprehensive_evaluation.py --split test --out runs/final_eval/comprehensive_evaluation.json }
Step "shadow tuning"    { python scripts/tune_shadow_threshold.py --split val --out runs/final_eval/shadow_threshold_tuning.json }
Step "calibration"      { python scripts/fit_calibration.py --split val --out-dir production_models --report runs/final_eval/calibration.json }
Step "conformal"        { python scripts/fit_conformal.py --calibration-split val --test-split test --alpha 0.05 --out-dir production_models --report runs/final_eval/conformal.json }

Write-Host ""
Write-Host "DONE. See runs/final_eval/ and production_models/"
