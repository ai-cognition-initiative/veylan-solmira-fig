"""Diagram 7: Agent Foundations & Speculative Directions Landscape."""

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

C_CORE       = "#444444"
C_GROUNDED   = "#4A8C6F"
C_THEORETIC  = "#5B8DBE"
C_SPECULATIVE= "#D4775B"
C_RING       = "#E8E8E8"
C_BORDER     = "#CCCCCC"

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

cx, cy = 6.0, 4.5

# ─── Concentric rings (grounded → speculative gradient) ───
ring_radii = [3.8, 2.8, 1.8]
ring_colors = ["#D4775B", "#5B8DBE", "#4A8C6F"]
ring_alphas = [0.06, 0.08, 0.10]
ring_labels = ["Speculative", "Theoretical", "Grounded / Empirical"]

for r, c, a in zip(ring_radii, ring_colors, ring_alphas):
    circle = mpatches.Circle((cx, cy), r, facecolor=c, alpha=a,
                              edgecolor=c, linewidth=1.2, linestyle="--")
    ax.add_patch(circle)

# Ring labels (right side)
ax.text(cx + 3.9, cy + 1.2, "Speculative", fontsize=9, color=C_SPECULATIVE,
        fontstyle="italic", rotation=-15)
ax.text(cx + 2.9, cy + 0.8, "Theoretical", fontsize=9, color=C_THEORETIC,
        fontstyle="italic", rotation=-15)
ax.text(cx + 1.9, cy + 0.5, "Grounded", fontsize=9, color=C_GROUNDED,
        fontstyle="italic", rotation=-15)

# ─── Core node ───
core = mpatches.FancyBboxPatch(
    (cx - 1.3, cy - 0.5), 2.6, 1.0,
    boxstyle="round,pad=0.2", facecolor="#F0EDE5", alpha=1.0,
    edgecolor=C_CORE, linewidth=2.5)
ax.add_patch(core)
ax.text(cx, cy + 0.1, "Agent\nFoundations", fontsize=13, fontweight="bold",
        color=C_CORE, ha="center", va="center")

# ─── Radiating topics ───
topics = [
    # (angle_deg, distance, label, level_color, level_label)
    (90,   1.7, "Value Drift\nMeasurement", C_GROUNDED, "grounded"),
    (160,  1.9, "Behavioral\nProbe Design", C_GROUNDED, "grounded"),
    (30,   2.8, "Bottlenecked\nAgents", C_THEORETIC, "theoretical"),
    (210,  2.9, "Coherent Extrapolated\nVolition (CEV)", C_THEORETIC, "theoretical"),
    (330,  3.0, "Convergent\nInstrumentality", C_SPECULATIVE, "speculative"),
    (270,  3.2, "Acausal\nTrading", C_SPECULATIVE, "speculative"),
    (120,  3.5, "Simulation\nArguments", C_SPECULATIVE, "speculative"),
]

for angle_deg, dist, label, color, level in topics:
    angle_rad = np.deg2rad(angle_deg)
    tx = cx + dist * np.cos(angle_rad)
    ty = cy + dist * np.sin(angle_rad)

    # Connection line
    ax.plot([cx, tx], [cy, ty], color=color, lw=1.5, alpha=0.3, zorder=1)

    # Node
    node_w = 2.2
    node_h = 0.8
    node = mpatches.FancyBboxPatch(
        (tx - node_w / 2, ty - node_h / 2), node_w, node_h,
        boxstyle="round,pad=0.12", facecolor=color, alpha=0.12,
        edgecolor=color, linewidth=1.5, zorder=3)
    ax.add_patch(node)
    ax.text(tx, ty, label, fontsize=9, fontweight="bold", color=color,
            ha="center", va="center", zorder=4)

# ─── Gradient arrow at bottom ───
arrow_y = 0.6
ax.annotate("", xy=(10, arrow_y), xytext=(2, arrow_y),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=2))

# Gradient color blocks under arrow
n_steps = 80
for i in range(n_steps):
    x0 = 2 + i * 8 / n_steps
    t = i / n_steps
    r = int(74 + t * (212 - 74))
    g = int(140 + t * (119 - 140))
    b = int(111 + t * (91 - 111))
    color_hex = f"#{r:02x}{g:02x}{b:02x}"
    ax.barh(arrow_y - 0.2, 8 / n_steps, 0.15, left=x0,
            color=color_hex, alpha=0.4)

ax.text(2.0, arrow_y - 0.55, "Grounded", fontsize=10, color=C_GROUNDED,
        fontweight="bold", ha="left")
ax.text(10.0, arrow_y - 0.55, "Speculative", fontsize=10, color=C_SPECULATIVE,
        fontweight="bold", ha="right")
ax.text(6.0, arrow_y - 0.55, "Theoretical", fontsize=10, color=C_THEORETIC,
        fontweight="bold", ha="center")

# ─── Title ───
ax.set_title("Agent Foundations: Research Landscape",
             fontsize=16, fontweight="bold", color="#333333", pad=18)

# ─── Subtitle ───
ax.text(6.0, 8.6,
        "The broader intellectual landscape within which drift research is situated",
        fontsize=10, ha="center", color="#999999", fontstyle="italic")

fig.tight_layout()
fig.savefig("docs/schematics/whiteboard/agent-foundations-landscape.png",
            dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
plt.close(fig)
print("Done: agent-foundations-landscape.png")
