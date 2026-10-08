"""Draw the design schematic: learned policy -> CBF safety filter -> CARLA.

The figure shows the closed loop, what the filter solves at every step, and the
one-step projection it performs.

    python tools_safety/make_schematic.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs_safety", "figures")

INK = "#14181f"
MUTED = "#4a5262"
POLICY_C, POLICY_E = "#e8f1fb", "#2f6fb5"
FILTER_C, FILTER_E = "#fdeeee", "#b5372f"
ENV_C, ENV_E = "#eaf5ec", "#2f7d45"


def box(ax, x, y, w, h, text, fc, ec, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=3, linespacing=1.6)


def seg(ax, p, q, color=INK, lw=1.6, head=False, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style if head else "-",
                                 mutation_scale=16, linewidth=lw, color=color, zorder=4))


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                linewidth=1.2, edgecolor="#c3c9d4", facecolor="#fbfcfe", zorder=1))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center", fontsize=11.5,
            weight="bold", color=INK, zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(12.6, 8.3))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 8.3)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.3, 8.02, "CBF safety filter for a learned driving policy",
            ha="center", va="center", fontsize=16, weight="bold", color=INK)
    ax.text(6.3, 7.66, "the filter is a projection: it changes the action of the policy only when that "
            "action would leave the safe set", ha="center", va="center", fontsize=10.5,
            color=MUTED, style="italic")

    # ---------------------------------------------------------------- loop
    box(ax, 0.35, 5.95, 3.05, 1.35,
        "Policy  $\\pi_\\theta$\nPPO, 150k steps\n15 Hz", POLICY_C, POLICY_E)
    box(ax, 4.30, 5.80, 3.95, 1.65,
        "CBF safety filter\nQP over (steer, throttle)\n"
        "$\\min_u \\|W(u-u_{rl})\\|^2 + \\rho\\sum_j s_j + \\ell(u)$\n"
        "$h_i(x_{k+j}) \\geq (1-\\alpha_i)^j h_i(x_k)$",
        FILTER_C, FILTER_E, fs=10)
    box(ax, 9.20, 5.95, 3.05, 1.35,
        "CARLA 0.10.0\nTown10HD_Opt, 15 Hz\nvehicle dynamics", ENV_C, ENV_E)

    seg(ax, (3.40, 6.62), (4.30, 6.62), head=True)
    ax.text(3.85, 6.95, "$u_{rl}$", ha="center", va="center", fontsize=10.5, color=INK)
    seg(ax, (8.25, 6.62), (9.20, 6.62), head=True)
    ax.text(8.72, 6.95, "$u^{*}$", ha="center", va="center", fontsize=10.5, color=INK)

    # feedback loop, routed between the two rows so that nothing overlaps
    seg(ax, (10.72, 5.95), (10.72, 5.35))
    seg(ax, (10.72, 5.35), (1.88, 5.35))
    seg(ax, (1.88, 5.35), (1.88, 5.95), head=True)
    ax.text(6.30, 5.62,
            "measured state $x_k=(v, e_y, e_\\psi, \\kappa, \\mathrm{waypoints})$ "
            "+ obstacle gap, every step",
            ha="center", va="center", fontsize=10, color=INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none"))

    # ------------------------------------------------------- projection panel
    px, py, pw, ph = 0.35, 0.40, 5.75, 4.15
    panel(ax, px, py, pw, ph, "what the filter does in one step")

    ox, oy, sw, sh = 1.55, 1.32, 3.35, 2.20
    ax.add_patch(plt.Rectangle((ox, oy), sw, sh, fc="#f2f4f8", ec="#9aa3b2", lw=1.2, zorder=2))
    ax.add_patch(plt.Polygon([[ox + 0.50, oy + 0.55], [ox + 1.25, oy + 0.45], [ox + 2.45, oy + 0.85],
                              [ox + 2.85, oy + 1.55], [ox + 2.20, oy + 1.90], [ox + 1.00, oy + 1.82],
                              [ox + 0.55, oy + 1.15]],
                             closed=True, fc="#dcecdc", ec=ENV_E, lw=1.2, zorder=3))
    ax.text(ox + 1.72, oy + 1.15, "safe set\n$h_i(x_{k+1}) \\geq 0$", ha="center", va="center",
            fontsize=10, color=ENV_E, zorder=4)
    ax.plot([ox + 0.28], [oy + 1.85], marker="o", ms=9, color=ENV_E, zorder=5)
    ax.text(ox + 0.10, oy + 2.02, "$u_{rl}$ (policy)", ha="left", va="bottom",
            fontsize=9.5, color=ENV_E, zorder=6)
    ax.annotate("$u^{*}$ (executed)", (ox + 1.10, oy + 1.30), xytext=(ox + 1.55, oy + 1.86),
                fontsize=9.5, color=FILTER_E, ha="left", va="bottom", zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.9, color="#8a93a3"))
    seg(ax, (ox + 0.36, oy + 1.83), (ox + 1.04, oy + 1.36), color=FILTER_E)
    ax.text(ox - 0.30, oy + sh / 2, "throttle", rotation=90, ha="center", va="center",
            fontsize=9.5, color=MUTED)
    ax.text(ox + sw / 2, oy - 0.26, "steer", ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(px + pw / 2, py + 0.36,
            "if $u_{rl}$ is already inside the safe set, $u^{*}=u_{rl}$\n"
            "(measured: $\\|u^{*}-u_{rl}\\|_\\infty < 10^{-6}$)",
            ha="center", va="center", fontsize=9.5, color=MUTED, linespacing=1.5)

    # --------------------------------------------------------- barrier panel
    bx, by, bw, bh = 6.35, 0.40, 5.90, 4.15
    panel(ax, bx, by, bw, bh, "the five barriers of this design")
    rows = [
        ("lane, right", r"$h_1=(w/2-m)-e_y$"),
        ("lane, left", r"$h_2=(w/2-m)+e_y$"),
        ("speed", r"$h_3=v_{\max}-v$"),
        ("curve", r"$h_4=\sqrt{a_{\mathrm{lat}}/\kappa}-v$"),
        ("headway to obstacle", r"$h_5=\mathrm{gap}-(d_{\min}+\tau v)$"),
    ]
    for i, (name, formula) in enumerate(rows):
        y = by + bh - 0.95 - 0.42 * i
        ax.text(bx + 0.30, y, "\u2022", ha="center", va="center", fontsize=11, color=FILTER_E)
        ax.text(bx + 0.55, y, name, ha="left", va="center", fontsize=10, color=INK)
        ax.text(bx + bw - 0.30, y, formula, ha="right", va="center", fontsize=10.5, color=INK)
    ax.plot([bx + 0.30, bx + bw - 0.30], [by + 1.62, by + 1.62], color="#d5d9e0", lw=1.0)
    ax.text(bx + 0.30, by + 1.22,
            r"$w/2=1.75$ m, $m=0.30$ m, $v_{\max}=13.9$ m/s,",
            ha="left", va="center", fontsize=9)
    ax.text(bx + 0.30, by + 1.00,
            r"$a_{\mathrm{lat}}=2.5$ m/s$^2$, $d_{\min}=5$ m, $\tau=1.2$ s",
            ha="left", va="center", fontsize=9)
    ax.text(bx + 0.30, by + 0.66, "same trained checkpoint, filter off vs on:",
            ha="left", va="center", fontsize=9, color=MUTED)
    ax.text(bx + 0.30, by + 0.36,
            r"worst $\min_i h_i$: $-0.53 \to -0.15$       "
            r"worst $|e_y|$: 1.98 m $\to$ 1.60 m (half-lane 1.75 m)",
            ha="left", va="center", fontsize=9, color=MUTED)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(OUT, f"fig0_schematic.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", path)


if __name__ == "__main__":
    main()
