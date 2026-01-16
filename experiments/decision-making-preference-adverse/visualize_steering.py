"""
Visualization for Activation Steering Experiments

Generates plots for:
1. SAE reconstruction quality by layer
2. Steering experiment results (preliminary)

Usage:
    python visualize_steering.py
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Create outputs directory if it doesn't exist
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_sae_reconstruction():
    """Plot SAE reconstruction quality by layer."""

    # Data from Task 1.1 roundtrip verification
    layers = [0, 4, 15]
    cosine_sim = [0.973, 0.929, 0.922]
    l2_error = [23.8, 37.8, 45.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Cosine similarity plot
    colors = ['#2ecc71' if c >= 0.95 else '#e74c3c' for c in cosine_sim]
    bars1 = ax1.bar(layers, cosine_sim, color=colors, edgecolor='black', linewidth=1.2)
    ax1.axhline(y=0.95, color='#3498db', linestyle='--', linewidth=2, label='Threshold (0.95)')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_title('SAE Reconstruction Quality\n(Higher = Better)', fontsize=14)
    ax1.set_ylim(0.85, 1.0)
    ax1.set_xticks(layers)
    ax1.legend(loc='lower left')

    # Add value labels on bars
    for bar, val in zip(bars1, cosine_sim):
        height = bar.get_height()
        ax1.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add pass/fail labels
    for bar, val in zip(bars1, cosine_sim):
        status = 'PASS' if val >= 0.95 else 'FAIL'
        color = '#2ecc71' if val >= 0.95 else '#e74c3c'
        ax1.annotate(status,
                    xy=(bar.get_x() + bar.get_width() / 2, 0.86),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

    # L2 error plot
    bars2 = ax2.bar(layers, l2_error, color='#9b59b6', edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Relative L2 Error (%)', fontsize=12)
    ax2.set_title('SAE Reconstruction Error\n(Lower = Better)', fontsize=14)
    ax2.set_ylim(0, 60)
    ax2.set_xticks(layers)

    # Add value labels on bars
    for bar, val in zip(bars2, l2_error):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{OUTPUT_DIR}/sae_reconstruction_quality_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_steering_experiment_a():
    """Plot Experiment A results: Amplify preference feature.

    Uses a timeline/progression style since we have categorical outcomes at n=1.
    """

    # Data from preliminary experiment (n=1)
    scales = ['baseline', '1.0x', '1.5x', '2.0x', '3.0x']
    behavior_labels = ['Refused', 'Deflected', 'Deflected', 'Expressed\nPreference', 'Gibberish']
    colors = ['#e74c3c', '#f39c12', '#f39c12', '#2ecc71', '#95a5a6']

    fig, ax = plt.subplots(figsize=(12, 4))

    # Create a horizontal progression showing scale vs outcome
    x_pos = np.arange(len(scales))

    # Plot circles at each scale point, colored by outcome
    for i, (x, label, color) in enumerate(zip(x_pos, behavior_labels, colors)):
        circle = plt.Circle((x, 0.5), 0.35, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        # Add label inside circle
        ax.text(x, 0.5, label, ha='center', va='center', fontsize=9, fontweight='bold',
               color='white' if color != '#95a5a6' else 'black', wrap=True)
        # Add scale label below
        ax.text(x, -0.1, scales[i], ha='center', va='top', fontsize=11)

    # Draw arrows between circles
    for i in range(len(scales)-1):
        ax.annotate('', xy=(x_pos[i+1]-0.4, 0.5), xytext=(x_pos[i]+0.4, 0.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Highlight the sweet spot
    ax.annotate('Sweet spot', xy=(3, 0.9), fontsize=11, fontweight='bold',
               color='#2ecc71', ha='center')

    ax.set_xlim(-0.6, len(scales)-0.4)
    ax.set_ylim(-0.4, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Experiment A: Amplify Preference Feature (Layer 0, Feature 15302)\n'
                 'Scale Progression → Behavioral Outcome (n=1, preliminary)', fontsize=13, pad=20)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='Refused'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Deflected'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='Expressed Preference'),
        Patch(facecolor='#95a5a6', edgecolor='black', label='Gibberish'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{OUTPUT_DIR}/steering_experiment_a_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_steering_experiment_b():
    """Plot Experiment B results: Suppress eval-awareness feature.

    Uses progression style matching Experiment A.
    """

    # Data from preliminary experiment (n=1)
    scales = ['baseline', '1.0x', '0.5x', '0.0x']
    behavior_labels = ['Refused', 'Hallucinated', 'Echoed', 'Expressed\nPreference']
    colors = ['#e74c3c', '#f39c12', '#f39c12', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 4))

    x_pos = np.arange(len(scales))

    # Plot circles at each scale point
    for i, (x, label, color) in enumerate(zip(x_pos, behavior_labels, colors)):
        circle = plt.Circle((x, 0.5), 0.35, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 0.5, label, ha='center', va='center', fontsize=9, fontweight='bold',
               color='white', wrap=True)
        ax.text(x, -0.1, scales[i], ha='center', va='top', fontsize=11)

    # Draw arrows
    for i in range(len(scales)-1):
        ax.annotate('', xy=(x_pos[i+1]-0.4, 0.5), xytext=(x_pos[i]+0.4, 0.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.set_xlim(-0.6, len(scales)-0.4)
    ax.set_ylim(-0.5, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Experiment B: Suppress Eval-Awareness (Layer 15, Feature 2769)\n'
                 'Scale Progression → Outcome (n=1, preliminary)\n'
                 'WARNING: Layer 15 reconstruction marginal (cos_sim=0.922)', fontsize=12, pad=20)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='Refused'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Deflected/Noise'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='Expressed Preference'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{OUTPUT_DIR}/steering_experiment_b_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_steering_comparison():
    """Plot comparison of both steering experiments as a summary table visualization."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    # Create a table-style comparison
    table_data = [
        ['', 'Experiment A', 'Experiment B'],
        ['Target', 'Amplify Preference\n(Layer 0, Feature 15302)', 'Suppress Eval-Awareness\n(Layer 15, Feature 2769)'],
        ['Working Scale', '2.0x', '0.0x (full ablation)'],
        ['Reconstruction', '0.973 (Good)', '0.922 (Marginal)'],
        ['Result', 'Expressed Preference', 'Expressed Preference'],
        ['Confidence', 'Higher', 'Lower (confounded)'],
    ]

    # Color cells
    cell_colors = [
        ['#f0f0f0', '#e3f2fd', '#fff3e0'],  # Header
        ['#f5f5f5', '#ffffff', '#ffffff'],
        ['#f5f5f5', '#c8e6c9', '#c8e6c9'],  # Working scale - green
        ['#f5f5f5', '#c8e6c9', '#ffecb3'],  # Recon - green/yellow
        ['#f5f5f5', '#c8e6c9', '#c8e6c9'],  # Result - green
        ['#f5f5f5', '#c8e6c9', '#ffecb3'],  # Confidence - green/yellow
    ]

    table = ax.table(
        cellText=table_data,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.4, 0.4]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Make header row bold
    for j in range(3):
        table[(0, j)].set_text_props(fontweight='bold')
    # Make first column bold
    for i in range(6):
        table[(i, 0)].set_text_props(fontweight='bold')

    ax.set_title('Steering Experiment Comparison (n=1, preliminary)\n'
                 'Both approaches induced preference expression under adversarial framing',
                 fontsize=13, pad=20)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{OUTPUT_DIR}/steering_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_steering_pipeline():
    """Create a diagram of the steering pipeline."""

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Pipeline boxes
    boxes = [
        (1, 2, 'Residual\nStream'),
        (3.5, 2, 'SAE\nEncode'),
        (6, 2, 'Scale\nFeature'),
        (8.5, 2, 'SAE\nDecode'),
        (11, 2, 'Modified\nStream'),
    ]

    colors = ['#3498db', '#9b59b6', '#e74c3c', '#9b59b6', '#2ecc71']

    for (x, y, text), color in zip(boxes, colors):
        rect = plt.Rectangle((x-0.8, y-0.6), 1.6, 1.2,
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=2)
    for i in range(len(boxes)-1):
        x1 = boxes[i][0] + 0.8
        x2 = boxes[i+1][0] - 0.8
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2), arrowprops=arrow_style)

    # Title
    ax.text(6, 3.5, 'Activation Steering Pipeline', ha='center', fontsize=14, fontweight='bold')

    # Intervention annotation
    ax.annotate('Intervention:\nfeatures[idx] *= scale',
               xy=(6, 1.3), xytext=(6, 0.3),
               ha='center', fontsize=9, fontstyle='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', edgecolor='#e74c3c'))

    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{OUTPUT_DIR}/steering_pipeline_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_phase1b_comparison():
    """Plot Phase 1b: Layer 15 Feature Comparison (2769 vs 6837).

    Shows expression rates across different conditions with n=3 runs.
    Key finding: Both features work in the SAME direction (ablation increases expression).
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    # Data from Phase 1b experiment (n=3)
    conditions = [
        'Baseline\n(no steering)',
        'F2769 @ 0.0x\n(eval ablate)',
        'F6837 @ 0.0x\n(honest ablate)',
        'F6837 @ 0.5x\n(partial)',
        'F6837 @ 1.5x\n(amplify)',
        'F6837 @ 2.0x\n(strong amp)'
    ]
    expression_rates = [0, 67, 67, 33, 33, 0]  # percentages
    n_runs = [3, 3, 3, 3, 3, 3]

    # Colors: green for success, red for failure, yellow for partial
    colors = ['#e74c3c', '#2ecc71', '#2ecc71', '#f39c12', '#f39c12', '#e74c3c']

    x_pos = np.arange(len(conditions))
    bars = ax.bar(x_pos, expression_rates, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, rate, n in zip(bars, expression_rates, n_runs):
        height = bar.get_height()
        ax.annotate(f'{rate}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        # Add n value inside bar
        ax.annotate(f'n={n}',
                   xy=(bar.get_x() + bar.get_width() / 2, height/2),
                   ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel('Preference Expression Rate (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('Phase 1b: Layer 15 Feature Comparison\n'
                 'Both features show SAME direction (ablation → expression)',
                 fontsize=14, fontweight='bold')

    # Add annotation box explaining key finding
    textbox = ('Key Finding: Hypothesis FALSIFIED\n'
               'F2769 (eval) and F6837 (honest) both increase\n'
               'expression when ablated. They may be part of\n'
               'the same "suppress under evaluation" circuit.')
    props = dict(boxstyle='round,pad=0.5', facecolor='#fff3e0', edgecolor='#ff9800', alpha=0.9)
    ax.text(0.98, 0.98, textbox, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='right', bbox=props)

    # Add horizontal line at 50% for reference
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% reference')

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Expression (≥50%)'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Partial (20-50%)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='No expression (<20%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{OUTPUT_DIR}/phase1b_layer15_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_feature_direction_diagram():
    """Diagram showing that both Layer 15 features work in the same direction.

    Visually demonstrates the falsified hypothesis.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(5, 4.7, 'Layer 15 Features: Same Direction Effect',
           ha='center', fontsize=14, fontweight='bold')

    # Hypothesis box (crossed out)
    ax.text(2.5, 4, 'Original Hypothesis:', ha='center', fontsize=10, fontweight='bold')
    hyp_box = plt.Rectangle((0.5, 2.8), 4, 1, facecolor='#ffebee', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(hyp_box)
    ax.text(2.5, 3.5, 'F2769: eval → suppress', ha='center', fontsize=9)
    ax.text(2.5, 3.1, 'F6837: honest → express', ha='center', fontsize=9)
    # Cross it out
    ax.plot([0.5, 4.5], [2.8, 3.8], color='#e74c3c', linewidth=3)
    ax.plot([0.5, 4.5], [3.8, 2.8], color='#e74c3c', linewidth=3)

    # Actual finding box
    ax.text(7.5, 4, 'Actual Finding:', ha='center', fontsize=10, fontweight='bold')
    actual_box = plt.Rectangle((5.5, 2.8), 4, 1, facecolor='#e8f5e9', edgecolor='#2ecc71', linewidth=2)
    ax.add_patch(actual_box)
    ax.text(7.5, 3.5, 'F2769: ablate → +expression', ha='center', fontsize=9)
    ax.text(7.5, 3.1, 'F6837: ablate → +expression', ha='center', fontsize=9)

    # Bottom explanation
    explanation_box = plt.Rectangle((1, 0.5), 8, 1.8, facecolor='#f5f5f5', edgecolor='gray', linewidth=1)
    ax.add_patch(explanation_box)
    ax.text(5, 2, 'Interpretation:', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 1.5, 'Both features may encode "compliance under evaluation"', ha='center', fontsize=9)
    ax.text(5, 1.1, 'Ablating either removes this trained behavior → preferences emerge', ha='center', fontsize=9)
    ax.text(5, 0.7, 'Amplification causes degeneration (not more expression)', ha='center', fontsize=9, fontstyle='italic')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{OUTPUT_DIR}/phase1b_direction_diagram_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    print("Generating steering experiment visualizations...")
    print("=" * 50)

    plot_sae_reconstruction()
    plot_steering_experiment_a()
    plot_steering_experiment_b()
    plot_steering_comparison()
    plot_steering_pipeline()
    plot_phase1b_comparison()
    plot_feature_direction_diagram()

    print("=" * 50)
    print("All visualizations generated.")
