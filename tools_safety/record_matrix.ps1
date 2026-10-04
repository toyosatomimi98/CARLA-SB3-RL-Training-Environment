# Record several demo variants and rank them (see pick_best_demo.py).
$ErrorActionPreference = "Continue"
$py = ".\venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path safety_results\logs | Out-Null

$roll = (Get-ChildItem tensorboard\PPO_roll_*\final_model.zip -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
Write-Host "roll policy: $roll"

Write-Host "=== A: reference controller, disturbance 8-17 s ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy reference --seed 7 --disturb-start 8 --disturb-end 17 --seconds 26 --out safety_results\demo\A_ref_seed7\driving_demo.mp4 *> safety_results\logs\vid_A.log

Write-Host "=== B: reference controller, disturbance 6-16 s ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy reference --seed 21 --disturb-start 6 --disturb-end 16 --seconds 26 --out safety_results\demo\B_ref_seed21\driving_demo.mp4 *> safety_results\logs\vid_B.log

Write-Host "=== C: reference controller, disturbance 10-20 s ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy reference --seed 33 --disturb-start 10 --disturb-end 20 --seconds 26 --out safety_results\demo\C_ref_seed33\driving_demo.mp4 *> safety_results\logs\vid_C.log

if ($roll) {
  Write-Host "=== D: trained RL policy (throttle-floor variant) ===" -ForegroundColor Cyan
  & $py -u tools_safety\make_demo_video.py --policy rl --model "$roll" --algo PPO --seed 5 --disturb-start 9 --disturb-end 18 --seconds 26 --out safety_results\demo\D_rl_roll\driving_demo.mp4 *> safety_results\logs\vid_D.log
}

Write-Host "=== ranking ===" -ForegroundColor Green
& $py tools_safety\pick_best_demo.py
Write-Host "DONE" -ForegroundColor Green
