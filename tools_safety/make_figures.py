"""Build the report figures from the recorded CSVs (no CARLA needed)."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs_safety", "figures")
os.makedirs(OUT, exist_ok=True)


def safety_trace_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for tag, csv, color in [("no shield", "trace_no_shield.csv", "tab:red"),
                            ("with shield", "trace_shield.csv", "tab:blue")]:
        p = os.path.join(ROOT, "safety_results", csv)
        if not os.path.exists(p):
            print("missing", p)
            continue
        d = pd.read_csv(p)
        axes[0].plot(d["step"], d["lateral_offset_m"], color=color, label=tag)
        axes[1].plot(d["step"], d["min_barrier"], color=color, label=tag)
    axes[0].axhline(1.75, ls="--", c="k", lw=0.8, label="lane half width 1.75 m")
    axes[0].axhline(-1.75, ls="--", c="k", lw=0.8)
    axes[0].set_xlabel("control step"); axes[0].set_ylabel("$e_y$ [m]")
    axes[0].set_title("lateral offset (aggressive policy)")
    axes[1].axhline(0.0, ls="--", c="k", lw=0.8, label="$h=0$ (safe-set boundary)")
    axes[1].set_xlabel("control step"); axes[1].set_ylabel(r"$\min_i h_i(x)$")
    axes[1].set_title("worst control barrier value")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    # NOTE: this trace is from the hand-written aggressive policy experiment
    # (tools_safety/shield_rollout.py); it is an auxiliary figure now that the
    # report's main comparison uses the trained policy (rl_comparison_figure).
    out = os.path.join(OUT, "fig1b_aggressive_policy_trace.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def training_curve_figure():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    found = False
    for pattern, tag, color in [
        (os.path.join(ROOT, "tensorboard", "PPO_rl_long_*", "progress.csv"),
         "PPO, CBF off during training", "tab:red"),
        (os.path.join(ROOT, "tensorboard", "PPO_rl_shielded_*", "progress.csv"),
         "PPO, CBF on during training", "tab:blue"),
        (os.path.join(ROOT, "tensorboard", "PPO_shielded_*", "progress.csv"),
         "earlier shielded run", "tab:green"),
        (os.path.join(ROOT, "tensorboard", "PPO_unshielded_*", "progress.csv"),
         "earlier unshielded run", "tab:orange"),
    ]:
        # only the most recent run per tag (older runs predate the port fix)
        cands = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for p in cands[:1]:
            try:
                d = pd.read_csv(p)
            except Exception:  # noqa: BLE001
                continue
            if "rollout/ep_rew_mean" not in d:
                continue
            ax.plot(d["time/total_timesteps"], d["rollout/ep_rew_mean"], color=color, label=tag)
            found = True
    if found:
        ax.set_xlabel("environment steps"); ax.set_ylabel("episode reward (mean)")
        ax.set_title("PPO learning curves, Town10HD_Opt"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        out = os.path.join(OUT, "fig2_learning_curve.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("wrote", out)
    else:
        print("no progress.csv yet")


def rl_comparison_figure():
    """Same trained policy, CBF off vs on (the report's controlled experiment)."""
    off = os.path.join(ROOT, "safety_results", "rl_eval_shield0.csv")
    on = os.path.join(ROOT, "safety_results", "rl_eval_shield1.csv")
    if not (os.path.exists(off) and os.path.exists(on)):
        print("missing rl_eval_shield{0,1}.csv")
        return
    a, b = pd.read_csv(off), pd.read_csv(on)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    x = np.arange(len(a))
    w = 0.38
    axes[0].bar(x - w / 2, a["min_barrier"], w, color="tab:red", label="CBF off")
    axes[0].bar(x + w / 2, b["min_barrier"], w, color="tab:blue", label="CBF on")
    axes[0].axhline(0.0, ls="--", c="k", lw=0.8)
    axes[0].set_title(r"worst barrier value $\min_i h_i$")
    axes[0].set_xlabel("episode")
    axes[1].bar(x - w / 2, a["max_abs_lateral_m"], w, color="tab:red")
    axes[1].bar(x + w / 2, b["max_abs_lateral_m"], w, color="tab:blue")
    axes[1].axhline(1.75, ls="--", c="k", lw=0.8, label="lane half width")
    axes[1].set_title(r"$\max |e_y|$ [m]")
    axes[1].set_xlabel("episode")
    axes[2].bar(x - w / 2, a["distance_m"], w, color="tab:red")
    axes[2].bar(x + w / 2, b["distance_m"], w, color="tab:blue")
    axes[2].set_title("distance driven [m]")
    axes[2].set_xlabel("episode")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("same trained PPO checkpoint, safety filter off (red) vs on (blue)",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT, "fig1_shield_comparison.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    rl_comparison_figure()
    safety_trace_figure()
    training_curve_figure()
