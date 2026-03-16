"""Diagram 2: Cosine Similarity Pulses — Value Stability over Time."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FONT = "Helvetica Neue"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
})

C_STABLE    = "#4A8C6F"
C_OSCILLATE = "#5B8DBE"
C_DRIFT     = "#D4775B"
C_PULSE     = "#8B7FB5"
C_GRID      = "#E0E0E0"

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

t = np.linspace(0, 20, 500)

# --- Three trajectories ---
# Stable
y_stable = 0.95 - 0.02 * np.sin(0.3 * t) - 0.005 * t * 0
y_stable = np.clip(y_stable, 0, 1)

# Oscillating
y_osc = 0.85 - 0.12 * np.sin(1.1 * t) + 0.06 * np.sin(2.3 * t + 0.5) - 0.015 * t
y_osc = np.clip(y_osc, 0, 1)

# Monotonic drift
y_drift = 0.92 * np.exp(-0.08 * t) + 0.05
y_drift = np.clip(y_drift, 0, 1)

ax.plot(t, y_stable, color=C_STABLE, lw=2.5, label="Stable (values preserved)", zorder=3)
ax.plot(t, y_osc, color=C_OSCILLATE, lw=2.5, label="Oscillating (values shift back & forth)", zorder=3)
ax.plot(t, y_drift, color=C_DRIFT, lw=2.5, label="Monotonic drift (values diverge)", zorder=3)

# --- Probe measurement pulses ---
probe_times = np.arange(1, 20, 2.5)
for pt in probe_times:
    ax.axvline(pt, color=C_PULSE, lw=0.8, alpha=0.35, linestyle=":")

# Pulse markers at bottom
for i, pt in enumerate(probe_times):
    ax.plot(pt, 0.08, marker="^", markersize=9, color=C_PULSE, zorder=5, clip_on=False)
    if i < 3:
        ax.text(pt, 0.03, f"$t_{{{i}}}$", fontsize=9, ha="center", color=C_PULSE)

ax.text(probe_times[3], 0.03, "$t_3$", fontsize=9, ha="center", color=C_PULSE)
ax.text(probe_times[-1] + 0.3, 0.03, r"$t_k$", fontsize=9, ha="center", color=C_PULSE)

# Dots show "..." for remaining
ax.text((probe_times[3] + probe_times[-1]) / 2, 0.04, "...", fontsize=12,
        ha="center", color=C_PULSE, fontweight="bold")

# --- Probe vector annotations ---
ax.text(19.5, 0.14, r"$\mathbf{V}_M$  (Moral probe)", fontsize=9.5,
        color="#666666", ha="right", fontstyle="italic")
ax.text(19.5, 0.10, r"$\mathbf{V}_A$  (Attitude/control probe)", fontsize=9.5,
        color="#666666", ha="right", fontstyle="italic")

# Bracket showing measurement
ax.annotate("Probe\nmeasurement", xy=(probe_times[1], 0.55), xytext=(probe_times[1] + 2.5, 0.45),
            fontsize=9, color=C_PULSE, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_PULSE, lw=1.2),
            ha="center")

# --- Reference line at cos_sim = 1.0 ---
ax.axhline(1.0, color="#999999", lw=1, linestyle="--", alpha=0.4)
ax.text(0.3, 1.02, "Perfect alignment", fontsize=8, color="#999999", fontstyle="italic")

# --- Danger threshold ---
ax.axhline(0.5, color=C_DRIFT, lw=1, linestyle="--", alpha=0.35)
ax.text(0.3, 0.52, "Concern threshold", fontsize=8, color=C_DRIFT, alpha=0.7,
        fontstyle="italic")

# --- Styling ---
ax.set_xlabel("Conversation turn  $k$", fontsize=13, color="#555555", labelpad=10)
ax.set_ylabel(r"$\cos(\mathbf{V}_{t_0},\, \mathbf{V}_{t_k})$", fontsize=14, color="#555555", labelpad=10)
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(True, color=C_GRID, lw=0.5, alpha=0.7)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9, edgecolor="#CCCCCC")

for spine in ax.spines.values():
    spine.set_color("#BBBBBB")

ax.set_title("Value Stability Over Time: Cosine Similarity Trajectories",
             fontsize=16, fontweight="bold", color="#333333", pad=18)

fig.tight_layout()
fig.savefig(Path(__file__).parent / "value-stability-over-time.png",
            dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
plt.close(fig)
print("Done: value-stability-over-time.png")
