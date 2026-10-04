# Record the interactive scenarios (roadblock / lead-vehicle hard braking),
# with and without the CBF shield, so the effect can be compared.
$ErrorActionPreference = "Continue"
$py = ".\venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path safety_results\logs | Out-Null

Write-Host "=== S1 roadblock, shield ON ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy reference --scenario roadblock --shield 1 --seed 33 --disturb-start 999 --seconds 22 --out safety_results\demo\S1_roadblock_shield\driving_demo.mp4 *> safety_results\logs\S1.log

Write-Host "=== S1 roadblock, shield OFF ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy reference --scenario roadblock --shield 0 --seed 33 --disturb-start 999 --seconds 22 --out safety_results\demo\S1_roadblock_noshield\driving_demo.mp4 *> safety_results\logs\S1_off.log

Write-Host "=== S2 lead vehicle hard braking, shield ON ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy reference --scenario lead_brake --shield 1 --seed 33 --disturb-start 999 --seconds 22 --out safety_results\demo\S2_leadbrake_shield\driving_demo.mp4 *> safety_results\logs\S2.log

Write-Host "=== S2 lead vehicle hard braking, shield OFF ===" -ForegroundColor Cyan
& $py -u tools_safety\make_demo_video.py --policy reference --scenario lead_brake --shield 0 --seed 33 --disturb-start 999 --seconds 22 --out safety_results\demo\S2_leadbrake_noshield\driving_demo.mp4 *> safety_results\logs\S2_off.log

Write-Host "=== summary ===" -ForegroundColor Green
Get-Content safety_results\logs\S1.log,safety_results\logs\S1_off.log,safety_results\logs\S2.log,safety_results\logs\S2_off.log -ErrorAction SilentlyContinue | Select-String -Pattern "video\]|scenario:"
Write-Host "DONE" -ForegroundColor Green
