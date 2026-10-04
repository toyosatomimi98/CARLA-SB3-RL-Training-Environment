# Record shield OFF / ON for the two blocking scenarios and build the
# side-by-side comparison videos (same seed, same scenario, only the filter differs).
$ErrorActionPreference = "Continue"
$py = ".\venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path safety_results\logs | Out-Null

foreach ($scn in @("roadblock", "lead_brake")) {
  foreach ($sh in @(0, 1)) {
    $tag = "SCN_" + $scn + $(if ($sh -eq 1) {"_shield"} else {"_noshield"})
    Write-Host ("=== recording " + $tag + " ===") -ForegroundColor Cyan
    & $py -u tools_safety\make_demo_video.py --policy reference --scenario $scn --shield $sh --seed 33 --disturb-start 999 --seconds 24 --out ("safety_results\demo\" + $tag + "\driving_demo.mp4") *> ("safety_results\logs\" + $tag + ".log")
  }
  & $py tools_safety\make_comparison_video.py --off ("safety_results\demo\SCN_" + $scn + "_noshield\driving_demo.mp4") --on ("safety_results\demo\SCN_" + $scn + "_shield\driving_demo.mp4") --out ("safety_results\demo\SCN_" + $scn + "_comparison.mp4")
}
Write-Host "DONE" -ForegroundColor Green
