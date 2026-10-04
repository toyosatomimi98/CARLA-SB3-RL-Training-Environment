"""Score the recorded demo variants and print a ranking.

A good demo video must show all three things at once:
  1. the agent actually DRIVES (a policy that stands still is a failure);
  2. the safety filter is actually NEEDED and ACTS (interventions > 0);
  3. the barriers still hold (the car stays inside the lane).
"""
import csv
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score(csv_path: str) -> dict:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return {}
    dt = 1.0 / 15.0
    v = np.array([float(r["speed_kmh"]) for r in rows]) / 3.6
    dist = float(np.sum(v) * dt)
    ey = np.abs(np.array([float(r["lateral"]) for r in rows]))
    mh = np.array([float(r["min_h"]) for r in rows])
    itv = np.array([int(r["intervened"]) for r in rows])
    dis = np.array([int(r["disturbance"]) for r in rows])
    itv_during = int(np.sum(itv[dis == 1])) if dis.any() else 0
    # driving: mean speed and distance
    mean_v = float(np.mean(v) * 3.6)
    # safety: how far outside the corridor did we get, and how often
    viol = int(np.sum(mh < 0))
    # score: drive + filter acts + keep the lane
    s = (min(dist, 120.0) / 120.0 * 40.0            # up to 40 pts for distance
         + min(mean_v, 30.0) / 30.0 * 20.0          # up to 20 pts for mean speed
         + min(itv_during, 20) / 20.0 * 20.0        # up to 20 pts for filter acting
         + max(0.0, 1.0 - max(ey.max() - 1.75, 0.0) / 1.0) * 10.0   # up to 10 pts lane keeping
         + max(0.0, 1.0 - viol / max(len(rows), 1) / 0.2) * 10.0)   # up to 10 pts few violations
    return dict(variant=os.path.basename(os.path.dirname(csv_path)),
                frames=len(rows), distance_m=round(dist, 1), mean_speed_kmh=round(mean_v, 1),
                max_abs_lateral=round(float(ey.max()), 2), min_barrier=round(float(mh.min()), 3),
                violation_steps=viol, interventions_total=int(itv.sum()),
                interventions_during_disturbance=itv_during, score=round(s, 1))


def main() -> None:
    pats = [os.path.join(ROOT, "safety_results", "demo", "*", "*telemetry.csv"),
            os.path.join(ROOT, "safety_results", "demo", "*telemetry.csv")]
    files = sorted({f for p in pats for f in glob.glob(p)})
    if not files:
        print("no telemetry csv found")
        return
    rows = [score(f) for f in files]
    rows = [r for r in rows if r]
    rows.sort(key=lambda r: -r["score"])
    hdr = f"{'variant':28s} {'frames':>6} {'dist_m':>7} {'v_kmh':>6} {'max|e_y|':>8} {'minh':>7} {'viol':>5} {'itv':>4} {'itv_d':>5} {'score':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:28s} {r['frames']:6d} {r['distance_m']:7.1f} {r['mean_speed_kmh']:6.1f} "
              f"{r['max_abs_lateral']:8.2f} {r['min_barrier']:7.3f} {r['violation_steps']:5d} "
              f"{r['interventions_total']:4d} {r['interventions_during_disturbance']:5d} {r['score']:6.1f}")
    if rows:
        print("\nBEST:", rows[0]["variant"], "->", os.path.join("safety_results", "demo"))


if __name__ == "__main__":
    main()
