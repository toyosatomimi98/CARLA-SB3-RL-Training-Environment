# Re-run every CARLA experiment after the "set_simulate_physics" port fix.
# CARLA only accepts one synchronous client at a time, so everything is serial.
$ErrorActionPreference = "Continue"
$py = ".\venv\Scripts\python.exe"
$log = "safety_results\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null

Write-Host "=== [1/4] aggressive-policy shield comparison ===" -ForegroundColor Cyan
& $py -u tools_safety\shield_rollout.py *> "$log\step1_shield_rollout.log"

Write-Host "=== [2/4] train PPO unshielded / shielded (12k steps each) ===" -ForegroundColor Cyan
& $py -u train_safety.py --config SAFETY_PPO_UNSHIELDED --total_timesteps 12000 --tag unshielded --num_checkpoints 3 *> "$log\step2a_train_unshielded.log"
& $py -u train_safety.py --config SAFETY_PPO --total_timesteps 12000 --tag shielded --num_checkpoints 3 *> "$log\step2b_train_shielded.log"

Write-Host "=== [3/4] evaluate the unshielded policy with shield OFF / ON ===" -ForegroundColor Cyan
$model = (Get-ChildItem tensorboard\PPO_unshielded_*\final_model.zip | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
Write-Host "model: $model"
& $py -u tools_safety\eval_safety.py --model "$model" --algo PPO --shield 0 --episodes 3 --steps 500 --out safety_results\eval_off.csv *> "$log\step3a_eval_off.log"
& $py -u tools_safety\eval_safety.py --model "$model" --algo PPO --shield 1 --episodes 3 --steps 500 --out safety_results\eval_on.csv  *> "$log\step3b_eval_on.log"

Write-Host "=== [4/4] demo video ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --seconds 26 *> "$log\step4_video.log"

Write-Host "=== figures ===" -ForegroundColor Cyan
& $py tools_safety\make_figures.py *> "$log\step5_figures.log"
Write-Host "ALL DONE" -ForegroundColor Green
