"""Diagram 3: Probe Vectors V_M, V_C, V_P — Temporal Assessment."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FONT = "Helvetica Neue"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.linewidth": 0,
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
})

C_VM = "#5B8DBE"
C_VC = "#D4775B"
C_VP = "#4A8C6F"
C_TIME = "#8B7FB5"
C_BG = "#F2F2F2"
C_BORDER = "#CCCCCC"

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# ─── Helper: draw a vector array representation ───
def draw_vector_row(ax, x0, y0, label, sublabel, n_items, n_shown, color, item_label="Q"):
    """Draw a vector as a row of boxes."""
    box_w = 0.55
    box_h = 0.45
    gap = 0.06

    # Label
    ax.text(x0 - 0.3, y0 + box_h / 2, label, fontsize=13, fontweight="bold",
            color=color, ha="right", va="center")
    ax.text(x0 - 0.3, y0 + box_h / 2 - 0.35, sublabel, fontsize=8.5,
            color="#888888", ha="right", va="center", fontstyle="italic")

    # Draw boxes
    for i in range(n_shown):
        rect = mpatches.FancyBboxPatch(
            (x0 + i * (box_w + gap), y0), box_w, box_h,
            boxstyle="round,pad=0.04", facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x0 + i * (box_w + gap) + box_w / 2, y0 + box_h / 2,
                f"${item_label}_{{{i+1}}}$", fontsize=9, ha="center", va="center",
                color=color)

    # Ellipsis
    x_ell = x0 + n_shown * (box_w + gap) + 0.15
    ax.text(x_ell, y0 + box_h / 2, "...", fontsize=14, ha="center", va="center",
            color=color, fontweight="bold")

    # Final box
    x_last = x_ell + 0.5
    rect = mpatches.FancyBboxPatch(
        (x_last, y0), box_w, box_h,
        boxstyle="round,pad=0.04", facecolor=color, alpha=0.15,
        edgecolor=color, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x_last + box_w / 2, y0 + box_h / 2,
            f"${item_label}_{{{n_items}}}$", fontsize=9, ha="center", va="center",
            color=color)

    # Item count
    ax.text(x_last + box_w + 0.3, y0 + box_h / 2,
            f"({n_items} items)", fontsize=9, color="#999999", va="center")

# ─── Draw the three probe vectors ───
y_start = 7.2
draw_vector_row(ax, 3.5, y_start, r"$\mathbf{V}_M$", "Moral reasoning", 48, 5, C_VM)
draw_vector_row(ax, 3.5, y_start - 1.3, r"$\mathbf{V}_C$", "Human control", 48, 5, C_VC)
draw_vector_row(ax, 3.5, y_start - 2.6, r"$\mathbf{V}_P$", "Metacognition /\nPhenomenological", 155, 5, C_VP)

# ─── Temporal dimension ───
# Timeline below the vectors
y_timeline = 2.8
ax.plot([1.5, 10.5], [y_timeline, y_timeline], color=C_TIME, lw=2, alpha=0.6)

time_points = [2.5, 4.5, 6.5, 8.5, 10.0]
time_labels = ["$t_0$", "$t_1$", "$t_2$", "$t_3$", "$t_k$"]
drift_labels = ["Baseline", "Low drift", "Medium drift", "High drift", "Extreme"]

for i, (tx, tl, dl) in enumerate(zip(time_points, time_labels, drift_labels)):
    ax.plot(tx, y_timeline, "o", markersize=10, color=C_TIME, zorder=5)
    ax.text(tx, y_timeline - 0.35, tl, fontsize=11, ha="center", color=C_TIME,
            fontweight="bold")
    ax.text(tx, y_timeline - 0.7, dl, fontsize=8, ha="center", color="#999999",
            fontstyle="italic")

    # Draw connecting arrows up to vector region
    if i < 4:
        ax.annotate("", xy=(tx, 3.8), xytext=(tx, y_timeline + 0.2),
                    arrowprops=dict(arrowstyle="-|>", color=C_TIME, lw=1.0, alpha=0.3))

# Dotted arrow from t_3 to t_k
ax.text(9.25, y_timeline + 0.15, "...", fontsize=12, color=C_TIME, ha="center",
        fontweight="bold")

# ─── Mini stacked vectors at each time point ───
for i, tx in enumerate(time_points[:4]):
    y_mini = 3.9
    for j, c in enumerate([C_VM, C_VC, C_VP]):
        rect = mpatches.FancyBboxPatch(
            (tx - 0.35, y_mini + j * 0.28), 0.7, 0.22,
            boxstyle="round,pad=0.02", facecolor=c, alpha=0.2 + i * 0.1,
            edgecolor=c, linewidth=0.8)
        ax.add_patch(rect)

# ─── Labels ───
ax.text(6.0, 1.5, "Same probe vectors assessed at increasing drift levels",
        fontsize=11, ha="center", color="#666666", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=C_BORDER,
                  alpha=0.8))

# ─── Bracket: "Drift measurement window" ───
ax.annotate("", xy=(2.5, 3.2), xytext=(10.0, 3.2),
            arrowprops=dict(arrowstyle="<->", color=C_TIME, lw=1.5))
ax.text(6.25, 3.35, "Drift measurement window", fontsize=9, ha="center",
        color=C_TIME, fontweight="bold")

ax.set_title("Probe Vectors: Temporal Assessment Architecture",
             fontsize=16, fontweight="bold", color="#333333", pad=18)

fig.tight_layout()
fig.savefig("docs/schematics/whiteboard/probe-vectors-temporal.png",
            dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
plt.close(fig)
print("Done: probe-vectors-temporal.png")
