# Evaluate the trained policy once, then record several demo-video variants
# (different seeds / disturbance windows) so we can keep the best one.
$ErrorActionPreference = "Continue"
$py = ".\venv\Scripts\python.exe"
$model = (Get-ChildItem tensorboard\PPO_drive_shielded_*\final_model.zip | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
Write-Host "model: $model"

Write-Host "=== evaluation: does the policy actually DRIVE? ===" -ForegroundColor Cyan
& $py -u tools_safety\eval_safety.py --model "$model" --algo PPO --config SAFETY_PPO_DRIVE --shield 1 --episodes 3 --steps 400 --out safety_results\eval_drive_on.csv *> safety_results\logs\eval_drive_on.log
& $py -u tools_safety\eval_safety.py --model "$model" --algo PPO --config SAFETY_PPO_DRIVE --shield 0 --episodes 3 --steps 400 --out safety_results\eval_drive_off.csv *> safety_results\logs\eval_drive_off.log

Write-Host "=== recording v1_seed7 ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy rl --model "$model" --algo PPO --seed 7 --disturb-start 8 --disturb-end 17 --disturb-mix 1.0 --seconds 26 --out safety_results\demo\v1_seed7\driving_demo.mp4 *> safety_results\logs\video_v1.log

Write-Host "=== recording v2_seed21 ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy rl --model "$model" --algo PPO --seed 21 --disturb-start 6 --disturb-end 14 --disturb-mix 1.0 --seconds 26 --out safety_results\demo\v2_seed21\driving_demo.mp4 *> safety_results\logs\video_v2.log

Write-Host "=== recording v3_seed33 ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy rl --model "$model" --algo PPO --seed 33 --disturb-start 10 --disturb-end 20 --disturb-mix 0.8 --seconds 26 --out safety_results\demo\v3_seed33\driving_demo.mp4 *> safety_results\logs\video_v3.log

Write-Host "=== ranking ===" -ForegroundColor Green
& $py tools_safety\pick_best_demo.py
Write-Host "ALL DONE" -ForegroundColor Green
