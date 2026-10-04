"""Extract the report's demo stills from the *current* recorded clips.

Keeps `docs_safety/figures/fig3..fig5` in step with the videos: they used to be
stills of the very first (pre-fix) run, which is why the report looked older
than the clips it described.

    python tools_safety/make_demo_frames.py
"""
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs_safety", "figures")
DEMO = os.path.join(ROOT, "safety_results", "demo")

# (file, source clip, seconds into the clip, what it shows)
SHOTS = [
    ("fig3_demo_normal.png", f"{DEMO}/MAIN_badcmd_shield/driving_demo.mp4", 6.0,
     "normal lane following, before the disturbance"),
    ("fig4_demo_intervention.png", f"{DEMO}/MAIN_badcmd_shield/driving_demo.mp4", 11.0,
     "disturbance injected, CBF overriding the policy"),
    ("fig5_demo_recovery.png", f"{DEMO}/MAIN_badcmd_shield/driving_demo.mp4", 20.0,
     "after the disturbance, back to lane-keeping"),
    ("fig6_cutin_comparison.png", f"{DEMO}/SCN_cutin_comparison.mp4", 12.0,
     "cut-in: contact without the filter (top), safe stop with it (bottom)"),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, src, seconds, what in SHOTS:
        if not os.path.exists(src):
            print("missing clip, skip", name)
            continue
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(seconds * fps)))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("skip", name)
            continue
        path = os.path.join(OUT, name)
        cv2.imwrite(path, frame)
        print(f"wrote {path}  (t={seconds:.0f}s - {what})")


if __name__ == "__main__":
    main()
