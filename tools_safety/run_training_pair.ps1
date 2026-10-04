# Train the same PPO agent with and without the CBF shield (port of the repo's
# run_experiments.py idea, but headless and safety-focused).
$ErrorActionPreference = "Continue"
$py = ".\venv\Scripts\python.exe"
Write-Host "=== 1/2 training WITHOUT shield ===" -ForegroundColor Cyan
& $py -u train_safety.py --config SAFETY_PPO_UNSHIELDED --total_timesteps 15000 --tag unshielded --num_checkpoints 3 2>&1 |
    Tee-Object -FilePath tmp_train_unshielded.log
Write-Host "=== 2/2 training WITH shield ===" -ForegroundColor Cyan
& $py -u train_safety.py --config SAFETY_PPO --total_timesteps 15000 --tag shielded --num_checkpoints 3 2>&1 |
    Tee-Object -FilePath tmp_train_shielded.log
Write-Host "=== done ===" -ForegroundColor Green
