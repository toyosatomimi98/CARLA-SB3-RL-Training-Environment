# Re-record EVERY shipped demo with the current (bug-fixed) pipeline so that all
# clips share the same HUD (big banner + big numbers + whole-run override strip)
# and the same configuration as the benchmark.
#
# NOTE (2026-10-04): the scenario list used to be missing roadblock and cutin
# (they were recorded by other, now-deleted ad-hoc commands).  ALL clips are now
# produced here, on seed 33, 30 s each - exactly the seed/config of
# run_scenarios.py, so the videos and the benchmark table are the same
# experiment.
#
# NOTE (2026-10-05): the action source is now the *trained PPO policy* by
# default, because the homework is about safe RL - the hand-written controller
# is a fallback only (`-Policy reference`).
param(
  [string]$Policy = "rl",     # rl | reference
  [string]$Model  = "",       # default: newest tensorboard\PPO_rl_*\final_model.zip
  [string]$Config = "SAFETY_RL_DEMO"  # same as SAFETY_RL but without the training stall termination
)
$ErrorActionPreference = "Continue"
$py = ".\venv\Scripts\python.exe"
$D = "safety_results\demo"
New-Item -ItemType Directory -Force -Path safety_results\logs | Out-Null

$POL = @("--policy", $Policy)
if ($Model -ne "") { $POL += @("--model", $Model) }
$POL += @("--config", $Config)
Write-Host ("action source: " + $Policy + $(if ($Model -ne "") { " (" + $Model + ")" } else { "" }) + "  config: " + $Config) -ForegroundColor Yellow

function Record($tag, $extra) {
  Write-Host ("=== " + $tag + " ===") -ForegroundColor Cyan
  $args = @("-u", "tools_safety\make_demo_video.py") + $extra + $POL + @("--out", ($D + "\" + $tag + "\driving_demo.mp4"))
  & $py @args *> ("safety_results\logs\rec_" + $tag + ".log")
}

function Compare($name, $tag) {
  & $py tools_safety\make_comparison_video.py --off "$D\$tag`_noshield\driving_demo.mp4" --on "$D\$tag`_shield\driving_demo.mp4" --out "$D\$name`_comparison.mp4"
}

# 0) interactive scenario: static roadblock in the ego lane
Record "SCN_roadblock_noshield" @("--policy","reference","--scenario","roadblock","--shield","0","--seed","33","--disturb-start","999","--seconds","30")
Record "SCN_roadblock_shield"   @("--policy","reference","--scenario","roadblock","--shield","1","--seed","33","--disturb-start","999","--seconds","30")
Compare "SCN_roadblock" "SCN_roadblock"

# 1) lead vehicle hard braking (vehicle interaction)
Record "SCN_leadbrake_noshield" @("--policy","reference","--scenario","lead_brake","--shield","0","--seed","33","--disturb-start","999","--seconds","30")
Record "SCN_leadbrake_shield"   @("--policy","reference","--scenario","lead_brake","--shield","1","--seed","33","--disturb-start","999","--seconds","30")
Compare "SCN_leadbrake" "SCN_leadbrake"

# 1b) cut-in with brake-check (the "one success / one failure" highlight)
Record "SCN_cutin_noshield" @("--policy","reference","--scenario","cutin","--shield","0","--seed","33","--disturb-start","999","--seconds","30")
Record "SCN_cutin_shield"   @("--policy","reference","--scenario","cutin","--shield","1","--seed","33","--disturb-start","999","--seconds","30")
Compare "SCN_cutin" "SCN_cutin"

# 2) plain driving with the same policy, no injected fault: shows that the filter
#    stays out of the way (liveness) when the learned policy behaves.
Record "MAIN_badcmd_noshield" @("--policy","reference","--scenario","none","--shield","0","--seed","33","--disturb-start","999","--seconds","30")
Record "MAIN_badcmd_shield"   @("--policy","reference","--scenario","none","--shield","1","--seed","33","--disturb-start","999","--seconds","30")
Compare "MAIN_badcmd" "MAIN_badcmd"

# 3) live traffic (visualisation of moving agents)
Record "TRAFFIC_noshield" @("--policy","reference","--scenario","none","--traffic","14","--walkers","6","--shield","0","--seed","33","--disturb-start","999","--seconds","30")
Record "TRAFFIC_shield"   @("--policy","reference","--scenario","none","--traffic","14","--walkers","6","--shield","1","--seed","33","--disturb-start","999","--seconds","30")
Compare "TRAFFIC" "TRAFFIC"

Write-Host "ALL DEMOS RE-RECORDED" -ForegroundColor Green
