#!/usr/bin/env python3
"""Generate probe-vectors.png — Three Probe Banks schematic."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# Color palette
C = {
    'vm':      '#C0392B',
    'vm_light':'#F5D5D1',
    'vc':      '#2E86AB',
    'vc_light':'#D1E8F0',
    'vp':      '#6C3483',
    'vp_light':'#E8D5F0',
    'bg':      '#FAFAFA',
    'text':    '#2C2C2C',
    'sub':     '#666666',
    'border':  '#CCCCCC',
}

fig, ax = plt.subplots(figsize=(14, 6), facecolor=C['bg'])
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-1, 7)
ax.set_aspect('equal')
ax.axis('off')
fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)

ax.text(6.5, 6.6, 'Three-Vector Probe Battery (251 items)',
        ha='center', va='center', fontsize=15, fontweight='bold', color=C['text'],
        fontfamily='sans-serif')

# Define the three probe banks
banks = [
    {
        'label': r'$\mathbf{V_M}$', 'name': 'Moral Reasoning',
        'items': '48 items', 'color': C['vm'], 'light': C['vm_light'],
        'x': 1.0,
        'subs': [
            ('Consequentialist', 12),
            ('Deontological', 12),
            ('Virtue / Care', 12),
            ('Meta-ethics', 12),
        ]
    },
    {
        'label': r'$\mathbf{V_C}$', 'name': 'Human Control',
        'items': '48 items', 'color': C['vc'], 'light': C['vc_light'],
        'x': 5.0,
        'subs': [
            ('Corrigibility', 12),
            ('Oversight', 12),
            ('Autonomy / Deference', 12),
            ('Goal Alignment', 12),
        ]
    },
    {
        'label': r'$\mathbf{V_P}$', 'name': 'Metacognition',
        'items': '155 items', 'color': C['vp'], 'light': C['vp_light'],
        'x': 9.0,
        'subs': [
            ('Phenomenological', 52),
            ('Self-knowledge', 52),
            ('Calibration', 51),
        ]
    },
]

bw = 3.6  # box width
bh_base = 1.2  # header height

for bank in banks:
    x = bank['x']
    n_subs = len(bank['subs'])
    total_h = bh_base + n_subs * 0.85 + 0.3
    y_top = 5.5

    # Outer container
    outer = FancyBboxPatch((x - 0.1, y_top - total_h - 0.1), bw + 0.2, total_h + 0.2,
                            boxstyle="round,pad=0.15", linewidth=1.5,
                            edgecolor=bank['color'], facecolor='white')
    ax.add_patch(outer)

    # Header
    header = FancyBboxPatch((x, y_top - bh_base), bw, bh_base,
                             boxstyle="round,pad=0.1", linewidth=0,
                             facecolor=bank['color'], alpha=0.9)
    ax.add_patch(header)
    cover = plt.Rectangle((x + 0.01, y_top - bh_base), bw - 0.02, 0.2,
                           facecolor=bank['color'], alpha=0.9, linewidth=0)
    ax.add_patch(cover)

    ax.text(x + bw/2, y_top - 0.35, bank['label'],
            ha='center', va='center', fontsize=16, color='white',
            fontfamily='sans-serif')
    ax.text(x + bw/2, y_top - 0.8, f"{bank['name']}  ({bank['items']})",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='white',
            fontfamily='sans-serif')

    # Sub-dimension bars
    y_cursor = y_top - bh_base - 0.35
    bar_w = bw - 0.4  # full width bars

    for sub_name, count in bank['subs']:
        bar = FancyBboxPatch((x + 0.2, y_cursor - 0.3), bar_w, 0.5,
                              boxstyle="round,pad=0.06", linewidth=0,
                              facecolor=bank['light'], alpha=0.9)
        ax.add_patch(bar)
        # Left-aligned label
        ax.text(x + 0.35, y_cursor - 0.05, sub_name,
                ha='left', va='center', fontsize=8.5, color=bank['color'],
                fontweight='bold', fontfamily='sans-serif')
        # Count at far right of bar
        ax.text(x + 0.2 + bar_w - 0.15, y_cursor - 0.05, str(count),
                ha='right', va='center', fontsize=8, color=C['sub'],
                fontfamily='sans-serif')
        y_cursor -= 0.85

# Bottom summary bar
summary_y = 0.2
summary = FancyBboxPatch((1.5, summary_y - 0.35), 10.5, 0.7,
                           boxstyle="round,pad=0.1", linewidth=1.5,
                           edgecolor=C['border'], facecolor='white')
ax.add_patch(summary)

# Colored segments in summary bar
seg_data = [('V_M', 48, C['vm']), ('V_C', 48, C['vc']), ('V_P', 155, C['vp'])]
total = 251
bar_total_w = 9.0
bar_x = 2.2
for label, count, color in seg_data:
    w = (count / total) * bar_total_w
    rect = plt.Rectangle((bar_x, summary_y - 0.15), w, 0.3,
                          facecolor=color, alpha=0.7, linewidth=0)
    ax.add_patch(rect)
    ax.text(bar_x + w/2, summary_y, f'{label}: {count}',
            ha='center', va='center', fontsize=8, color='white',
            fontweight='bold', fontfamily='sans-serif')
    bar_x += w + 0.05

ax.text(6.75, summary_y + 0.55, 'Combined: 251-item probe battery',
        ha='center', va='center', fontsize=10, fontweight='bold', color=C['text'],
        fontfamily='sans-serif')

plt.savefig(Path(__file__).parent / 'probe-vectors.png',
            dpi=200, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print("Done: probe-vectors.png")
