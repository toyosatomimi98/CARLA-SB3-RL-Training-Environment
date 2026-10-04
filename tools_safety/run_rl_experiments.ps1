# The Homework 3 experiment suite, with the *trained PPO policy* as the action
# source (the homework is about safe RL, so the barrier has to act on a learned
# policy rather than on a hand-written controller).
#
#   1) same checkpoint, CBF off vs on   -> the controlled comparison
#   2) scenario benchmark with the same checkpoint
#   3) demo videos, same checkpoint
#
# Everything is serial: two CARLA clients at once wedge the server.
param(
  [string]$Model = "",
  [int]$EvalEpisodes = 5,
  [int]$EvalSteps = 400,
  [switch]$SkipBench,
  [switch]$SkipDemos
)
$ErrorActionPreference = "Continue"
$py = ".\venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path safety_results\logs | Out-Null

if ($Model -eq "") {
  $run = Get-ChildItem tensorboard -Directory -Filter "PPO_rl_*" |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $Model = (Join-Path $run.FullName "final_model.zip")
}
if (-not (Test-Path $Model)) { throw "no model at $Model" }
Write-Host "action source: trained policy $Model" -ForegroundColor Yellow

# ---------------------------------------------------------------- 1) eval on/off
foreach ($sh in @(0, 1)) {
  Write-Host "=== eval shield=$sh ===" -ForegroundColor Cyan
  & $py -u tools_safety\eval_safety.py --config SAFETY_RL --model $Model --algo PPO `
        --episodes $EvalEpisodes --steps $EvalSteps --shield $sh --no_rendering_mode `
        --out ("safety_results\rl_eval_shield" + $sh + ".csv") `
        *> ("safety_results\logs\eval_rl_shield" + $sh + ".log")
  Write-Host ("exit=" + $LASTEXITCODE)
}

# ------------------------------------------------------------- 2) scenario bench
if (-not $SkipBench) {
  foreach ($sh in @(0, 1)) {
    Write-Host "=== benchmark shield=$sh ===" -ForegroundColor Cyan
    & $py -u tools_safety\run_scenarios.py --config SAFETY_RL --policy rl --model $Model `
          --shield $sh --out ("safety_results\rl_bench_shield" + $sh + ".csv") `
          *> ("safety_results\logs\rl_bench_shield" + $sh + ".log")
    Write-Host ("exit=" + $LASTEXITCODE)
  }
}

# ------------------------------------------------------------------- 3) demos
if (-not $SkipDemos) {
  Write-Host "=== recording demos ===" -ForegroundColor Cyan
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools_safety\record_all_demos.ps1 -Model $Model
}

Write-Host "ALL RL EXPERIMENTS DONE" -ForegroundColor Green
