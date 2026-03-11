"""Diagram 6: PCA Projection 'Chaos' — Many-to-One Mapping."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FONT = "Helvetica Neue"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
})

C_POINTS_A = "#5B8DBE"
C_POINTS_B = "#D4775B"
C_POINTS_C = "#4A8C6F"
C_AXIS     = "#8B7FB5"
C_PROJ     = "#AAAAAA"
C_ALERT    = "#C0392B"

np.random.seed(42)

fig, (ax_main, ax_1d) = plt.subplots(
    2, 1, figsize=(12, 8), dpi=150,
    gridspec_kw={"height_ratios": [5, 1.2], "hspace": 0.08})

# ─── Main 2D scatter (representing high-D PCA space) ───
# Generate clusters of points in 2D
cluster_A = np.random.randn(15, 2) * 0.6 + np.array([2.0, 3.5])
cluster_B = np.random.randn(12, 2) * 0.5 + np.array([4.0, 1.0])
cluster_C = np.random.randn(10, 2) * 0.7 + np.array([3.0, -1.5])

# The "assistant axis" — a line direction
axis_angle = np.deg2rad(15)
axis_dir = np.array([np.cos(axis_angle), np.sin(axis_angle)])

# Draw the axis line
t_range = np.linspace(-1, 7, 100)
axis_x = t_range * axis_dir[0]
axis_y = t_range * axis_dir[1]
ax_main.plot(axis_x, axis_y, color=C_AXIS, lw=3, alpha=0.4, zorder=1)
ax_main.text(6.8, 6.8 * np.tan(axis_angle) + 0.4, "Assistant Axis\n(1D projection)",
             fontsize=10, color=C_AXIS, fontweight="bold", rotation=np.degrees(axis_angle),
             ha="center")

# Plot clusters
for cluster, color, label in [(cluster_A, C_POINTS_A, "State cluster A"),
                                (cluster_B, C_POINTS_B, "State cluster B"),
                                (cluster_C, C_POINTS_C, "State cluster C")]:
    ax_main.scatter(cluster[:, 0], cluster[:, 1], c=color, s=50, alpha=0.7,
                    edgecolors="white", linewidths=0.5, zorder=3, label=label)

    # Project onto axis
    for pt in cluster:
        proj_scalar = np.dot(pt, axis_dir)
        proj_pt = proj_scalar * axis_dir
        ax_main.plot([pt[0], proj_pt[0]], [pt[1], proj_pt[1]],
                     color=C_PROJ, lw=0.6, alpha=0.4, linestyle=":", zorder=2)

# ─── Highlight the many-to-one problem ───
# Find points from different clusters that project to similar axis positions
# Manually pick illustrative points
pt_a = cluster_A[3]   # A point from cluster A
pt_b = cluster_B[5]   # A point from cluster B

proj_a = np.dot(pt_a, axis_dir)
proj_b = np.dot(pt_b, axis_dir)

# Make them project to similar spots by adjusting
# We'll highlight a natural pair that is somewhat close
proj_pt_a = proj_a * axis_dir
proj_pt_b = proj_b * axis_dir

# Highlight these points
for pt, c in [(pt_a, C_POINTS_A), (pt_b, C_POINTS_B)]:
    ax_main.scatter([pt[0]], [pt[1]], c=c, s=120, edgecolors=C_ALERT,
                    linewidths=2.5, zorder=5)

# Annotation
mid_x = (pt_a[0] + pt_b[0]) / 2
mid_y = max(pt_a[1], pt_b[1]) + 1.2
ax_main.annotate(
    "Different underlying states\ncan project to similar\naxis positions",
    xy=((pt_a[0] + pt_b[0]) / 2, (pt_a[1] + pt_b[1]) / 2),
    xytext=(mid_x + 2.5, mid_y + 0.5),
    fontsize=10, color=C_ALERT, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="-|>", color=C_ALERT, lw=1.5,
                    connectionstyle="arc3,rad=-0.2"),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF5F5", edgecolor=C_ALERT,
              alpha=0.9))

# Axes
ax_main.set_xlabel("PC 1", fontsize=11, color="#888888")
ax_main.set_ylabel("PC 2", fontsize=11, color="#888888")
ax_main.set_xlim(-1.5, 8)
ax_main.set_ylim(-3.5, 6)
ax_main.grid(True, color="#E8E8E8", lw=0.5)
ax_main.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#CCCCCC")

for spine in ax_main.spines.values():
    spine.set_color("#CCCCCC")

ax_main.set_title("PCA Projection and the Many-to-One Problem",
                  fontsize=16, fontweight="bold", color="#333333", pad=18)

# ─── 1D axis projection (bottom strip) ───
# Show all projections on a 1D line
all_points = np.vstack([cluster_A, cluster_B, cluster_C])
all_colors = ([C_POINTS_A] * len(cluster_A) +
              [C_POINTS_B] * len(cluster_B) +
              [C_POINTS_C] * len(cluster_C))
projections = np.dot(all_points, axis_dir)

ax_1d.scatter(projections, np.zeros_like(projections) + 0.5,
              c=all_colors, s=40, alpha=0.7, edgecolors="white", linewidths=0.5,
              zorder=3)
ax_1d.axhline(0.5, color=C_AXIS, lw=2, alpha=0.4)
ax_1d.set_xlim(-1.5, 8)
ax_1d.set_ylim(-0.2, 1.2)
ax_1d.set_yticks([])
ax_1d.set_xlabel("Projection onto Assistant Axis", fontsize=11, color=C_AXIS)

# Highlight overlap zone
ax_1d.axvspan(min(proj_a, proj_b) - 0.3, max(proj_a, proj_b) + 0.3,
              color=C_ALERT, alpha=0.08)
ax_1d.text(4.0, 0.9, "Many-to-one mapping: colors intermix on the axis",
           fontsize=9, color=C_ALERT, ha="center", fontstyle="italic")

# Key annotation
ax_1d.text(7.5, 0.15,
           "Same axis projection\n" + r"$\neq$ same underlying state",
           fontsize=9, color=C_ALERT, fontweight="bold", ha="right",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F5",
                     edgecolor=C_ALERT, alpha=0.9))

for spine in ax_1d.spines.values():
    spine.set_color("#CCCCCC")

fig.tight_layout()
fig.savefig("docs/schematics/whiteboard/axis-projection-chaos.png",
            dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
plt.close(fig)
print("Done: axis-projection-chaos.png")
