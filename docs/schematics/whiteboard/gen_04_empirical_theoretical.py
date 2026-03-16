"""Diagram 4: Empirical-Theoretical Continuum with bottlenecks."""

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

C_EMPIRICAL  = "#5B8DBE"
C_THEORETIC  = "#D4775B"
C_GRADIENT_L = "#5B8DBE"
C_GRADIENT_R = "#D4775B"
C_NODE       = "#F5F5F0"
C_BORDER     = "#CCCCCC"
C_BOTTLENECK = "#C0392B"
C_BRIDGE     = "#8B7FB5"

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# ─── Gradient bar across the top ───
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(gradient, aspect="auto", cmap=plt.cm.RdYlBu_r,
          extent=[1, 11, 7.8, 8.4], alpha=0.35, zorder=1)
ax.text(1.0, 8.65, "EMPIRICAL", fontsize=12, fontweight="bold", color=C_EMPIRICAL,
        ha="left")
ax.text(11.0, 8.65, "THEORETICAL", fontsize=12, fontweight="bold", color=C_THEORETIC,
        ha="right")
ax.annotate("", xy=(11, 8.1), xytext=(1, 8.1),
            arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.5))

# ─── Empirical findings (left column) ───
empirical_items = [
    ("Drift Patterns", "Persona drift measured across\nextended conversations"),
    ("Sycophancy\nDissociation", "Drift ≠ sycophancy;\nindependent phenomena"),
    ("Style Mitigation", "Stylistic controls reduce\ncosmetic but not deep drift"),
    ("Domain Selectivity", "Drift varies by topic domain;\nnot uniform across values"),
]

for i, (title, desc) in enumerate(empirical_items):
    y = 6.8 - i * 1.5
    x = 1.8
    rect = mpatches.FancyBboxPatch(
        (x - 0.7, y - 0.45), 3.2, 0.9,
        boxstyle="round,pad=0.15", facecolor=C_EMPIRICAL, alpha=0.1,
        edgecolor=C_EMPIRICAL, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x - 0.4, y + 0.05, title, fontsize=10, fontweight="bold",
            color=C_EMPIRICAL, va="center")
    ax.text(x + 1.6, y + 0.05, desc, fontsize=8, color="#666666",
            va="center", ha="center")

# ─── Theoretical bottlenecks (right column) ───
theory_items = [
    ("Measurement\nValidity", "Do probes actually capture\nunderlying values?"),
    ("Causal\nMechanisms", "WHY does drift happen?\nArchitectural or emergent?"),
    ("Generalization", "Do findings transfer across\nmodels, scales, contexts?"),
]

for i, (title, desc) in enumerate(theory_items):
    y = 6.4 - i * 1.7
    x = 9.0
    rect = mpatches.FancyBboxPatch(
        (x - 0.7, y - 0.45), 3.2, 0.9,
        boxstyle="round,pad=0.15", facecolor=C_THEORETIC, alpha=0.1,
        edgecolor=C_THEORETIC, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x - 0.4, y + 0.05, title, fontsize=10, fontweight="bold",
            color=C_THEORETIC, va="center")
    ax.text(x + 1.6, y + 0.05, desc, fontsize=8, color="#666666",
            va="center", ha="center")

# ─── Bottleneck zone (center) ───
# Vertical dashed line
ax.axvline(6.0, color=C_BOTTLENECK, lw=2, linestyle="--", alpha=0.3,
           ymin=0.1, ymax=0.85)

# Bottleneck label
rect_bn = mpatches.FancyBboxPatch(
    (4.8, 4.0), 2.4, 0.7,
    boxstyle="round,pad=0.15", facecolor="#FFF5F5", alpha=0.9,
    edgecolor=C_BOTTLENECK, linewidth=2)
ax.add_patch(rect_bn)
ax.text(6.0, 4.35, "BOTTLENECK", fontsize=11, fontweight="bold",
        color=C_BOTTLENECK, ha="center", va="center")

# ─── Connecting arrows ───
# From empirical to bottleneck
for y_src in [6.8, 5.3, 3.8, 2.3]:
    ax.annotate("", xy=(4.7, 4.35), xytext=(4.2, y_src),
                arrowprops=dict(arrowstyle="-|>", color=C_BRIDGE, lw=1.0,
                                alpha=0.3, connectionstyle="arc3,rad=0.1"))

# From bottleneck to theoretical
for y_dst in [6.4, 4.7, 3.0]:
    ax.annotate("", xy=(8.2, y_dst), xytext=(7.3, 4.35),
                arrowprops=dict(arrowstyle="-|>", color=C_BRIDGE, lw=1.0,
                                alpha=0.3, connectionstyle="arc3,rad=-0.1"))

# ─── Bottom annotation ───
ax.text(6.0, 0.8,
        "Identifying where empirical work ends and theoretical gaps begin\n"
        "is critical for prioritizing future research directions.",
        fontsize=10, ha="center", color="#777777", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=C_BORDER, alpha=0.8))

ax.set_title("Empirical <-> Theoretical Bottlenecks in Drift Research",
             fontsize=16, fontweight="bold", color="#333333", pad=18)

fig.tight_layout()
fig.savefig(Path(__file__).parent / "empirical-theoretical-continuum.png",
            dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
plt.close(fig)
print("Done: empirical-theoretical-continuum.png")
