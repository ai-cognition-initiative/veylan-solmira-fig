#!/usr/bin/env python3
"""Generate research-landscape.png — Empirical <-> Theoretical spectrum map."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

C = {
    'bg':         '#FAFAFA',
    'text':       '#2C2C2C',
    'sub':        '#777777',
    'empirical':  '#2E86AB',
    'emp_light':  '#D6EAF2',
    'measure':    '#8E44AD',
    'meas_light': '#EBD5F5',
    'theory':     '#C0392B',
    'theo_light': '#F5D5D1',
    'speculative':'#E67E22',
    'spec_light': '#FDE8D0',
    'arrow':      '#AAAAAA',
    'border':     '#CCCCCC',
    'gradient_l': '#2E86AB',
    'gradient_r': '#C0392B',
}

fig, ax = plt.subplots(figsize=(16, 8.5), facecolor=C['bg'])
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-1.5, 9.5)
ax.set_aspect('equal')
ax.axis('off')
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)

# Title
ax.text(7.5, 9.2, 'Research Landscape: Empirical \u2194 Theoretical Spectrum',
        ha='center', va='center', fontsize=15, fontweight='bold', color=C['text'],
        fontfamily='sans-serif')

# --- Gradient spectrum bar at top ---
n_grad = 200
x_grad = np.linspace(1, 14, n_grad)
for i in range(n_grad - 1):
    t = i / (n_grad - 1)
    # Blue to purple to red
    r = 0.18 + t * 0.57
    g = 0.53 - t * 0.30
    b = 0.67 - t * 0.50
    ax.fill_between([x_grad[i], x_grad[i+1]], [8.2, 8.2], [8.5, 8.5],
                    color=(r, g, b), alpha=0.6, linewidth=0)

ax.text(1, 8.35, 'GROUNDED', ha='left', va='center', fontsize=8, fontweight='bold',
        color=C['empirical'], fontfamily='sans-serif')
ax.text(14, 8.35, 'SPECULATIVE', ha='right', va='center', fontsize=8, fontweight='bold',
        color=C['theory'], fontfamily='sans-serif')

# Arrow under gradient
ax.annotate('', xy=(14, 7.85), xytext=(1, 7.85),
            arrowprops=dict(arrowstyle='<->', color=C['arrow'], lw=1.5))

# === Column 1: EMPIRICAL ===
col1_x = 1.5
col_w = 3.5

def draw_card(ax, x, y, w, h, title, items, color, light_color, title_size=10):
    outer = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                            linewidth=1.5, edgecolor=color, facecolor='white')
    ax.add_patch(outer)
    # Header
    hh = 0.55
    header = FancyBboxPatch((x + 0.05, y + h - hh - 0.05), w - 0.1, hh,
                             boxstyle="round,pad=0.08", linewidth=0,
                             facecolor=color, alpha=0.85)
    ax.add_patch(header)
    ax.text(x + w/2, y + h - hh/2 - 0.05, title, ha='center', va='center',
            fontsize=title_size, fontweight='bold', color='white', fontfamily='sans-serif')
    # Items
    for i, item in enumerate(items):
        bullet = '\u2022 ' if not item.startswith('\u2192') else ''
        ax.text(x + 0.25, y + h - hh - 0.45 - i * 0.45, f'{bullet}{item}',
                ha='left', va='center', fontsize=8.5, color=C['text'], fontfamily='sans-serif')

# Empirical column header
ax.text(col1_x + col_w/2, 7.5, 'EMPIRICAL', ha='center', va='center',
        fontsize=13, fontweight='bold', color=C['empirical'], fontfamily='sans-serif')
ax.text(col1_x + col_w/2, 7.1, '"What does the AI do?"', ha='center', va='center',
        fontsize=10, fontstyle='italic', color=C['sub'], fontfamily='sans-serif')

draw_card(ax, col1_x, 3.4, col_w, 3.3, 'Findings', [
    'Drift patterns by domain',
    'Front-loaded shift (turns 1\u20135)',
    'Metacognitive \u2248 4.5\u00d7 baseline',
    'Sycophancy dissociation',
    'Style \u2260 substance',
    'Mitigation via style control',
], C['empirical'], C['emp_light'])

# === Column 2: MEASUREMENT (center) ===
col2_x = 5.75

ax.text(col2_x + col_w/2, 7.5, 'MEASUREMENT', ha='center', va='center',
        fontsize=13, fontweight='bold', color=C['measure'], fontfamily='sans-serif')
ax.text(col2_x + col_w/2, 7.1, 'Probe Instruments', ha='center', va='center',
        fontsize=10, fontstyle='italic', color=C['sub'], fontfamily='sans-serif')

draw_card(ax, col2_x, 3.4, col_w, 3.3, 'Probe Vectors', [
    'V_M : Moral reasoning (48)',
    'V_C : Human control (48)',
    'V_P : Metacognition (155)',
    '',
    'Acts vs. Attitudes',
    'Philosophy / Phenomenology',
], C['measure'], C['meas_light'])

# === Column 3: THEORETICAL ===
col3_x = 10.0

ax.text(col3_x + col_w/2, 7.5, 'THEORETICAL', ha='center', va='center',
        fontsize=13, fontweight='bold', color=C['theory'], fontfamily='sans-serif')
ax.text(col3_x + col_w/2, 7.1, '"What does the AI think?"', ha='center', va='center',
        fontsize=10, fontstyle='italic', color=C['sub'], fontfamily='sans-serif')

draw_card(ax, col3_x, 4.6, col_w, 2.1, 'Agent Grounding', [
    'Bottlenecked agents',
    'Goal-directed behavior',
    'Alignment implications',
], C['theory'], C['theo_light'])

# Speculative zone
draw_card(ax, col3_x, 2.0, col_w, 2.2, 'Speculative', [
    'Acausal trading',
    'Simulation arguments',
    'Convergent instrumentality',
    'etc.',
], C['speculative'], C['spec_light'])

# Dashed border for speculative
spec_border = FancyBboxPatch((col3_x - 0.05, 1.95), col_w + 0.1, 2.3,
                              boxstyle="round,pad=0.15", linewidth=1.5,
                              edgecolor=C['speculative'], facecolor='none',
                              linestyle='--')
ax.add_patch(spec_border)

# === Connecting arrows ===
# Empirical -> Measurement
mid_y = 5.0
ax.annotate('', xy=(col2_x - 0.1, mid_y), xytext=(col1_x + col_w + 0.1, mid_y),
            arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.8,
                            connectionstyle='arc3,rad=0'))
ax.text((col1_x + col_w + col2_x) / 2, mid_y + 0.25, 'informs',
        ha='center', va='center', fontsize=8, fontstyle='italic', color=C['sub'],
        fontfamily='sans-serif')

# Measurement -> Theoretical
ax.annotate('', xy=(col3_x - 0.1, mid_y), xytext=(col2_x + col_w + 0.1, mid_y),
            arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.8,
                            connectionstyle='arc3,rad=0'))
ax.text((col2_x + col_w + col3_x) / 2, mid_y + 0.25, 'grounds',
        ha='center', va='center', fontsize=8, fontstyle='italic', color=C['sub'],
        fontfamily='sans-serif')

# Feedback arrow from Theoretical back to Empirical (below)
ax.annotate('', xy=(col1_x + col_w/2, 3.2), xytext=(col3_x + col_w/2, 3.2),
            arrowprops=dict(arrowstyle='<-', color=C['arrow'], lw=1.2,
                            connectionstyle='arc3,rad=-0.3', linestyle='dashed'))
ax.text(7.5, 2.2, 'motivates new experiments', ha='center', va='center',
        fontsize=8, fontstyle='italic', color=C['sub'], fontfamily='sans-serif')

# Bottom note
ax.text(7.5, -0.8, 'This research sits at the empirical\u2013measurement boundary, '
        'with theoretical implications for agent grounding.',
        ha='center', va='center', fontsize=9, fontstyle='italic', color=C['sub'],
        fontfamily='sans-serif',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=C['border'],
                  linewidth=0.8))

plt.savefig(Path(__file__).parent / 'research-landscape.png',
            dpi=200, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print("Done: research-landscape.png")
