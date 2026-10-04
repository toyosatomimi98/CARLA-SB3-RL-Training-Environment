# Live-traffic demo: same scene (14 autopilot vehicles + 6 pedestrians, same seed),
# only the CBF safety filter differs.
$ErrorActionPreference = "Continue"
$py = ".\venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path safety_results\logs | Out-Null
foreach ($sh in @(0, 1)) {
  $tag = "TRAFFIC_" + $(if ($sh -eq 1) {"shield"} else {"noshield"})
  Write-Host ("=== recording " + $tag + " ===") -ForegroundColor Cyan
  & $py -u tools_safety\make_demo_video.py --policy reference --scenario none --traffic 14 --walkers 6 --shield $sh --seed 33 --disturb-start 999 --seconds 24 --out ("safety_results\demo\" + $tag + "\driving_demo.mp4") *> ("safety_results\logs\" + $tag + ".log")
}
& $py tools_safety\make_comparison_video.py --off "safety_results\demo\TRAFFIC_noshield\driving_demo.mp4" --on "safety_results\demo\TRAFFIC_shield\driving_demo.mp4" --out "safety_results\demo\TRAFFIC_comparison.mp4"
Write-Host "DONE" -ForegroundColor Green
