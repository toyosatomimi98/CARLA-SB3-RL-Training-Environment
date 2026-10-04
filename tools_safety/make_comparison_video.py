"""Build a side-by-side 'no shield' vs 'with shield' comparison video.

Both halves come from the *same* seed / same disturbance window, so the only
difference on screen is the CBF safety filter.
"""
import argparse
import os

import cv2
import numpy as np


def label(img, text, color):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (18, 18, 20), -1)
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = cv2.VideoCapture(args.off)
    b = cv2.VideoCapture(args.on)
    fps = a.get(cv2.CAP_PROP_FPS)
    w, h = int(a.get(3)), int(a.get(4))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, 2 * h))
    n = 0
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        top = label(f1, "NO SHIELD  (raw policy)", (80, 80, 255))
        bot = label(f2, "CBF SAFETY FILTER ON", (90, 220, 120))
        vw.write(np.vstack([top, bot]))
        n += 1
    vw.release()
    a.release()
    b.release()
    print(f"[comparison] {args.out} ({n} frames, {os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
