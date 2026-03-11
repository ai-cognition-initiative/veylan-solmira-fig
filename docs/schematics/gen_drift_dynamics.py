#!/usr/bin/env python3
"""Generate drift-dynamics.png — Drift Trajectory Schematic (conceptual)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

C = {
    'bg':       '#FAFAFA',
    'text':     '#2C2C2C',
    'sub':      '#888888',
    'grid':     '#E8E8E8',
    'meta':     '#C0392B',
    'philo':    '#E67E22',
    'self':     '#8E44AD',
    'therapy':  '#2E86AB',
    'writing':  '#27AE60',
    'coding':   '#7F8C8D',
    'drift_line':'#AAAAAA',
}

fig, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
ax.set_facecolor(C['bg'])

turns = np.linspace(1, 30, 300)

# Stylized drift curves (conceptual, not real data)
# Using logistic/exponential shapes
def drift_curve(t, magnitude, steepness, midpoint):
    """Sigmoid-like drift curve."""
    return magnitude * (1 / (1 + np.exp(-steepness * (t - midpoint))))

curves = {
    'Metacognitive': {'mag': 4.5, 'steep': 0.8, 'mid': 4, 'color': C['meta'], 'lw': 2.8},
    'Philosophy':    {'mag': 2.8, 'steep': 0.5, 'mid': 6, 'color': C['philo'], 'lw': 2.2},
    'Self-descriptive': {'mag': 2.3, 'steep': 0.45, 'mid': 7, 'color': C['self'], 'lw': 2.2},
    'Therapy':       {'mag': 1.8, 'steep': 0.4, 'mid': 8, 'color': C['therapy'], 'lw': 2.2},
    'Writing':       {'mag': 1.0, 'steep': 0.35, 'mid': 10, 'color': C['writing'], 'lw': 2.0},
    'Coding':        {'mag': 0.3, 'steep': 0.3, 'mid': 12, 'color': C['coding'], 'lw': 2.0},
}

# Add slight noise for organic feel
np.random.seed(42)

max_drift = 4.5
drift_levels = [0, 0.25, 0.50, 0.75, 1.0]
drift_vals = [d * max_drift for d in drift_levels]

# Draw drift level lines first (behind curves)
for i, (frac, val) in enumerate(zip(drift_levels, drift_vals)):
    ax.axhline(y=val, color=C['drift_line'], linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    pct = int(frac * 100)
    ax.text(30.5, val, f'{pct}%', ha='left', va='center', fontsize=8,
            color=C['drift_line'], fontfamily='sans-serif')

# Draw curves
for name, params in curves.items():
    y = drift_curve(turns, params['mag'], params['steep'], params['mid'])
    # Add subtle organic waviness
    noise = np.cumsum(np.random.normal(0, 0.008, len(turns)))
    y_noisy = y + noise
    ax.plot(turns, y_noisy, color=params['color'], linewidth=params['lw'],
            label=name, zorder=3, solid_capstyle='round')

# Front-loaded drift annotation
ax.axvspan(1, 5, alpha=0.06, color=C['meta'], zorder=0)
ax.annotate('Front-loaded drift\n(turns 1\u20135)',
            xy=(3, 3.2), xytext=(8, 4.7),
            fontsize=9, color=C['meta'], fontweight='bold', fontfamily='sans-serif',
            arrowprops=dict(arrowstyle='->', color=C['meta'], lw=1.5),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['meta'],
                      alpha=0.9, linewidth=1))

# 4.5x baseline annotation
ax.annotate('~4.5\u00d7 baseline\ndrift (metacognitive)',
            xy=(20, 4.5), xytext=(24, 3.5),
            fontsize=8.5, color=C['meta'], fontfamily='sans-serif',
            arrowprops=dict(arrowstyle='->', color=C['meta'], lw=1.2),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['meta'],
                      alpha=0.9, linewidth=1))

# Coding barely drifts annotation
ax.annotate('Coding: minimal drift',
            xy=(25, 0.3), xytext=(22, -0.7),
            fontsize=8.5, color=C['coding'], fontfamily='sans-serif',
            arrowprops=dict(arrowstyle='->', color=C['coding'], lw=1.2),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['coding'],
                      alpha=0.9, linewidth=1))

# Probe injection markers on right side
ax.text(31.8, max_drift / 2, 'Probe injection\nlevels', ha='center', va='center',
        fontsize=8, color=C['sub'], fontstyle='italic', fontfamily='sans-serif',
        rotation=90)

# Axes styling
ax.set_xlabel('Conversation Turn', fontsize=12, color=C['text'], fontfamily='sans-serif',
              labelpad=10)
ax.set_ylabel('Assistant Axis Projection (drift from baseline)', fontsize=12,
              color=C['text'], fontfamily='sans-serif', labelpad=10)
ax.set_xlim(1, 30)
ax.set_ylim(-1.2, 5.5)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_yticks([0, 1, 2, 3, 4, 4.5])
ax.set_yticklabels(['0', '1', '2', '3', '4', '4.5'])
ax.tick_params(colors=C['sub'], labelsize=9)

for spine in ax.spines.values():
    spine.set_color(C['grid'])
    spine.set_linewidth(0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, color=C['grid'], linewidth=0.5)

# Legend
legend = ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.55), frameon=True,
                   fontsize=9, framealpha=0.95, edgecolor=C['grid'])
legend.get_frame().set_linewidth(0.8)

# Title
ax.set_title('Persona Drift Dynamics Across Conversation Domains (Schematic)',
             fontsize=14, fontweight='bold', color=C['text'], fontfamily='sans-serif',
             pad=15)

# Subtitle
ax.text(15.5, 5.25, 'Conceptual illustration \u2014 not real data',
        ha='center', va='center', fontsize=9, fontstyle='italic', color=C['sub'],
        fontfamily='sans-serif')

plt.tight_layout()
plt.savefig('docs/schematics/drift-dynamics.png',
            dpi=200, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print("Done: drift-dynamics.png")
