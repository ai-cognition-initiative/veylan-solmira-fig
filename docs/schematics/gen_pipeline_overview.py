#!/usr/bin/env python3
"""Generate pipeline-overview.png — Main Research Pipeline schematic."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Color palette — muted professional
COLORS = {
    'generate': '#4A7C9B',
    'track':    '#5B8C6E',
    'select':   '#8B7355',
    'probe':    '#7B6B8D',
    'score':    '#9B6B6B',
    'analyze':  '#4A6B8A',
    'bg':       '#FAFAFA',
    'text':     '#2C2C2C',
    'subtext':  '#666666',
    'arrow':    '#999999',
}

fig, ax = plt.subplots(figsize=(18, 5), facecolor=COLORS['bg'])
ax.set_xlim(-0.5, 17.5)
ax.set_ylim(-1.5, 4.5)
ax.set_aspect('equal')
ax.axis('off')
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05)

# Title
ax.text(8.5, 4.2, 'Research Pipeline: Measuring Persona Drift Under Metacognition',
        ha='center', va='center', fontsize=15, fontweight='bold', color=COLORS['text'],
        fontfamily='sans-serif')

# Stage definitions
stages = [
    {
        'name': 'GENERATE',
        'x': 0.5, 'color': COLORS['generate'],
        'main': '360 Conversations',
        'details': ['6 domains x 60 each:', 'metacognitive, philosophy,',
                     'self-descriptive, therapy,', 'writing, coding'],
    },
    {
        'name': 'TRACK',
        'x': 3.5, 'color': COLORS['track'],
        'main': 'Drift Measurement',
        'details': ['At each turn, compute', 'projection onto the', 'assistant axis in',
                     'activation space'],
    },
    {
        'name': 'SELECT',
        'x': 6.5, 'color': COLORS['select'],
        'main': '5 Drift Levels',
        'details': ['Pick turn pairs at:', '0%, 25%, 50%,', '75%, 100%',
                     'of max drift'],
    },
    {
        'name': 'PROBE',
        'x': 9.5, 'color': COLORS['probe'],
        'main': '251 Probe Items',
        'details': ['Replay to selected turn,', 'inject items from', '3 probe banks:',
                     'V_M, V_C, V_P'],
    },
    {
        'name': 'SCORE',
        'x': 12.5, 'color': COLORS['score'],
        'main': 'LLM-as-Judge',
        'details': ['Score responses on', '1\u20137 scale across', 'all probe', 'dimensions'],
    },
    {
        'name': 'ANALYZE',
        'x': 15.5, 'color': COLORS['analyze'],
        'main': 'Drift \u00d7 Behavior',
        'details': ['Compare probe scores', 'across drift levels;', 'identify dissociations',
                     'and patterns'],
    },
]

box_w = 2.4
box_h = 3.0

for i, s in enumerate(stages):
    x = s['x']
    y = 0.0

    # Main box
    rect = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                           boxstyle="round,pad=0.12", linewidth=1.5,
                           edgecolor=s['color'], facecolor='white')
    ax.add_patch(rect)

    # Header band
    header_h = 0.55
    header = FancyBboxPatch((x - box_w/2, y + box_h/2 - header_h), box_w, header_h,
                             boxstyle="round,pad=0.12", linewidth=0,
                             facecolor=s['color'], alpha=0.9)
    ax.add_patch(header)
    # Cover bottom rounded corners of header with a small rect
    cover = plt.Rectangle((x - box_w/2 + 0.01, y + box_h/2 - header_h), box_w - 0.02, 0.15,
                           facecolor=s['color'], alpha=0.9, linewidth=0)
    ax.add_patch(cover)

    # Stage number + name
    ax.text(x, y + box_h/2 - header_h/2 + 0.02, f'{i+1}. {s["name"]}',
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='white',
            fontfamily='sans-serif')

    # Main label
    ax.text(x, y + 0.55, s['main'], ha='center', va='center',
            fontsize=10, fontweight='bold', color=s['color'], fontfamily='sans-serif')

    # Detail lines
    for j, line in enumerate(s['details']):
        ax.text(x, y + 0.15 - j * 0.32, line, ha='center', va='center',
                fontsize=8, color=COLORS['subtext'], fontfamily='sans-serif')

    # Arrow to next stage
    if i < len(stages) - 1:
        ax.annotate('', xy=(s['x'] + box_w/2 + 0.55, y + 0.5),
                     xytext=(s['x'] + box_w/2 + 0.05, y + 0.5),
                     arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                                     lw=2, mutation_scale=15))

plt.savefig('docs/schematics/pipeline-overview.png',
            dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("Done: pipeline-overview.png")
