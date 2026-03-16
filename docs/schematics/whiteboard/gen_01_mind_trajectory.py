"""Diagram 1: Digital vs Human Mind Trajectories in value-space."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pathlib import Path

# --- Shared style ---
FONT = "Helvetica Neue"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.linewidth": 0,
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
})

# Muted palette
C_HUMAN     = "#5B8DBE"   # steel blue
C_DIGITAL   = "#D4775B"   # terra cotta
C_OVERLAP   = "#8B7FB5"   # muted purple
C_ATTRACTOR = "#4A8C6F"   # sage green
C_ARROW     = "#555555"
C_DIVERGE   = "#C0392B"

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

# --- Current regions (overlapping ellipses) ---
human_now = mpatches.Ellipse((3.2, 4.5), 3.6, 2.4, angle=-10,
                              facecolor=C_HUMAN, alpha=0.25, edgecolor=C_HUMAN, lw=2)
digital_now = mpatches.Ellipse((5.0, 4.0), 3.2, 2.0, angle=8,
                                facecolor=C_DIGITAL, alpha=0.25, edgecolor=C_DIGITAL, lw=2)
ax.add_patch(human_now)
ax.add_patch(digital_now)

ax.text(2.1, 5.3, "Human Minds\n(now)", fontsize=12, fontweight="bold",
        color=C_HUMAN, ha="center", va="center")
ax.text(6.1, 3.3, "Digital Minds\n(now)", fontsize=12, fontweight="bold",
        color=C_DIGITAL, ha="center", va="center")

# --- Future expanded regions (dashed) ---
human_fut = mpatches.Ellipse((2.0, 6.0), 5.5, 4.0, angle=-15,
                              facecolor="none", edgecolor=C_HUMAN, lw=1.8,
                              linestyle="--", alpha=0.6)
digital_fut = mpatches.Ellipse((7.0, 2.5), 5.0, 3.8, angle=12,
                                facecolor="none", edgecolor=C_DIGITAL, lw=1.8,
                                linestyle="--", alpha=0.6)
ax.add_patch(human_fut)
ax.add_patch(digital_fut)

ax.text(0.5, 7.5, "Human Minds\n(future)", fontsize=10, fontstyle="italic",
        color=C_HUMAN, alpha=0.7, ha="center")
ax.text(8.8, 1.2, "Digital Minds\n(future)", fontsize=10, fontstyle="italic",
        color=C_DIGITAL, alpha=0.7, ha="center")

# --- Evolution arrows ---
for start, end in [((3.2, 5.6), (2.2, 7.0)),
                   ((2.0, 4.8), (0.8, 6.2)),
                   ((4.0, 5.0), (2.8, 6.8))]:
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=C_HUMAN, lw=1.2, alpha=0.5))

for start, end in [((5.8, 3.2), (7.5, 2.0)),
                   ((6.2, 4.2), (8.2, 3.2)),
                   ((5.5, 3.0), (6.8, 1.5))]:
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=C_DIGITAL, lw=1.2, alpha=0.5))

# --- CEV attractors ---
# Convergent attractor basin
attractor_x, attractor_y = 4.5, 6.8
basin = mpatches.Ellipse((attractor_x, attractor_y), 1.6, 1.0, angle=0,
                          facecolor=C_ATTRACTOR, alpha=0.15, edgecolor=C_ATTRACTOR,
                          lw=2, linestyle="-")
ax.add_patch(basin)
ax.plot(attractor_x, attractor_y, marker="*", markersize=16, color=C_ATTRACTOR,
        markeredgecolor="white", markeredgewidth=0.8, zorder=5)
ax.text(attractor_x, attractor_y + 0.75, "CEV Attractor\n(convergent basin)",
        fontsize=9.5, ha="center", color=C_ATTRACTOR, fontweight="bold")

# Arrows toward attractor
ax.annotate("", xy=(4.1, 6.5), xytext=(2.8, 6.8),
            arrowprops=dict(arrowstyle="-|>", color=C_ATTRACTOR, lw=1.5, alpha=0.6))
ax.annotate("", xy=(4.8, 6.4), xytext=(6.5, 5.5),
            arrowprops=dict(arrowstyle="-|>", color=C_ATTRACTOR, lw=1.5, alpha=0.6))

# --- Divergence zone ---
ax.annotate("", xy=(9.2, 0.5), xytext=(7.5, 1.8),
            arrowprops=dict(arrowstyle="-|>", color=C_DIVERGE, lw=2.0, alpha=0.5))
ax.annotate("", xy=(0.2, 8.5), xytext=(1.2, 7.2),
            arrowprops=dict(arrowstyle="-|>", color=C_DIVERGE, lw=2.0, alpha=0.5))
ax.text(9.0, 0.3, "Divergence\n(no attractor)", fontsize=9, color=C_DIVERGE,
        ha="center", fontstyle="italic")

# --- Axes labels ---
ax.set_xlabel("Value Dimension 1", fontsize=12, color="#666666", labelpad=10)
ax.set_ylabel("Value Dimension 2", fontsize=12, color="#666666", labelpad=10)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 9.5)
ax.set_xticks([])
ax.set_yticks([])

# Light border
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("#CCCCCC")
    spine.set_linewidth(0.8)

ax.set_title("Mind Trajectory Space: Divergence vs. Convergence",
             fontsize=16, fontweight="bold", color="#333333", pad=18)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=C_HUMAN, alpha=0.3, edgecolor=C_HUMAN, label="Human mind-space"),
    mpatches.Patch(facecolor=C_DIGITAL, alpha=0.3, edgecolor=C_DIGITAL, label="Digital mind-space"),
    mpatches.Patch(facecolor=C_ATTRACTOR, alpha=0.2, edgecolor=C_ATTRACTOR, label="CEV attractor basin"),
    plt.Line2D([0], [0], color=C_DIVERGE, lw=2, alpha=0.6, label="Divergence trajectory"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9.5,
          framealpha=0.9, edgecolor="#CCCCCC")

fig.tight_layout()
fig.savefig(Path(__file__).parent / "mind-trajectory-space.png",
            dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
plt.close(fig)
print("Done: mind-trajectory-space.png")
