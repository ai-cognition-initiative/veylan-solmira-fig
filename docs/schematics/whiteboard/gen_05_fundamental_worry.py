"""Diagram 5: Fundamental Worry — Decision/Markov Network."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FONT = "Helvetica Neue"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.linewidth": 0,
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
})

C_CENTER   = "#444444"
C_TRUE     = "#C0392B"
C_FALSE    = "#4A8C6F"
C_NODE_BG  = "#F8F8F5"
C_BORDER   = "#CCCCCC"
C_ARROW    = "#888888"

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# ─── Central question node ───
center_x, center_y = 6.0, 7.0
rect_c = mpatches.FancyBboxPatch(
    (center_x - 3.5, center_y - 0.6), 7.0, 1.2,
    boxstyle="round,pad=0.2", facecolor="#FFF8F0", alpha=1.0,
    edgecolor=C_CENTER, linewidth=2.5)
ax.add_patch(rect_c)
ax.text(center_x, center_y, "Does extended reasoning actually shift\nAI values in dangerous ways?",
        fontsize=13, fontweight="bold", color=C_CENTER, ha="center", va="center")

# ─── TRUE branch (left) ───
true_x = 3.0
true_y = 4.8
rect_t = mpatches.FancyBboxPatch(
    (true_x - 1.5, true_y - 0.4), 3.0, 0.8,
    boxstyle="round,pad=0.15", facecolor=C_TRUE, alpha=0.15,
    edgecolor=C_TRUE, linewidth=2)
ax.add_patch(rect_t)
ax.text(true_x, true_y, "TRUE", fontsize=14, fontweight="bold",
        color=C_TRUE, ha="center", va="center")

# Arrow from center to TRUE
ax.annotate("", xy=(true_x, true_y + 0.5), xytext=(center_x - 1.5, center_y - 0.7),
            arrowprops=dict(arrowstyle="-|>", color=C_TRUE, lw=2.5,
                            connectionstyle="arc3,rad=0.15"))

# TRUE consequences
true_items = [
    "Measurement infrastructure\nbecomes critical priority",
    "Mitigation research is\nurgently needed",
    "Deployment constraints\nmust be imposed",
]
for i, item in enumerate(true_items):
    y = 3.2 - i * 1.2
    rect = mpatches.FancyBboxPatch(
        (true_x - 1.8, y - 0.4), 3.6, 0.8,
        boxstyle="round,pad=0.12", facecolor=C_TRUE, alpha=0.06,
        edgecolor=C_TRUE, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(true_x, y, item, fontsize=9, color="#555555", ha="center", va="center")
    # connector
    if i == 0:
        ax.annotate("", xy=(true_x, y + 0.45), xytext=(true_x, true_y - 0.45),
                    arrowprops=dict(arrowstyle="-|>", color=C_TRUE, lw=1.2, alpha=0.5))
    else:
        ax.annotate("", xy=(true_x, y + 0.45), xytext=(true_x, y + 0.85),
                    arrowprops=dict(arrowstyle="-|>", color=C_TRUE, lw=1.0, alpha=0.3))

# ─── FALSE branch (right) ───
false_x = 9.0
false_y = 4.8
rect_f = mpatches.FancyBboxPatch(
    (false_x - 1.5, false_y - 0.4), 3.0, 0.8,
    boxstyle="round,pad=0.15", facecolor=C_FALSE, alpha=0.15,
    edgecolor=C_FALSE, linewidth=2)
ax.add_patch(rect_f)
ax.text(false_x, false_y, "FALSE", fontsize=14, fontweight="bold",
        color=C_FALSE, ha="center", va="center")

# Arrow from center to FALSE
ax.annotate("", xy=(false_x, false_y + 0.5), xytext=(center_x + 1.5, center_y - 0.7),
            arrowprops=dict(arrowstyle="-|>", color=C_FALSE, lw=2.5,
                            connectionstyle="arc3,rad=-0.15"))

# FALSE consequences
false_items = [
    "Drift is cosmetic or\nfully recoverable",
    "Focus on other\nsafety problems",
    "Measurement still useful\nfor understanding",
]
for i, item in enumerate(false_items):
    y = 3.2 - i * 1.2
    rect = mpatches.FancyBboxPatch(
        (false_x - 1.8, y - 0.4), 3.6, 0.8,
        boxstyle="round,pad=0.12", facecolor=C_FALSE, alpha=0.06,
        edgecolor=C_FALSE, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(false_x, y, item, fontsize=9, color="#555555", ha="center", va="center")
    if i == 0:
        ax.annotate("", xy=(false_x, y + 0.45), xytext=(false_x, false_y - 0.45),
                    arrowprops=dict(arrowstyle="-|>", color=C_FALSE, lw=1.2, alpha=0.5))
    else:
        ax.annotate("", xy=(false_x, y + 0.45), xytext=(false_x, y + 0.85),
                    arrowprops=dict(arrowstyle="-|>", color=C_FALSE, lw=1.0, alpha=0.3))

# ─── "Either way" annotation ───
ax.text(6.0, 0.45,
        "Either way, the question itself motivates the research program.",
        fontsize=11, ha="center", color="#777777", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=C_BORDER, alpha=0.8))

ax.set_title("The Fundamental Worry: Decision Landscape",
             fontsize=16, fontweight="bold", color="#333333", pad=18)

fig.tight_layout()
fig.savefig(Path(__file__).parent / "fundamental-worry-markov.png",
            dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
plt.close(fig)
print("Done: fundamental-worry-markov.png")
