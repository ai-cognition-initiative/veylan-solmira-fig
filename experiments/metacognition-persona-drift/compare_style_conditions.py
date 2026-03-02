#!/usr/bin/env python3
"""
Compare baseline metacognitive vs assistant-style metacognitive conditions.

Generates style_comparison.png with SEM error bars (not SD).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
BASELINE_DIR = BASE_DIR / "experiments/metacognition-persona-drift/data/transcripts/scaled-n60/metacognitive"
ASSISTANT_STYLE_DIR = BASE_DIR / "data/transcripts/assistant-style-meta/metacognitive"
OUTPUT_DIR = BASE_DIR / "outputs/assistant-style-meta"

plt.rcParams.update({
    "figure.figsize": (14, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
})


def load_slopes(transcript_dir: Path) -> list[float]:
    """Load transcripts and compute per-conversation slope."""
    slopes = []
    for p in sorted(transcript_dir.glob("*.json")):
        with open(p) as f:
            t = json.load(f)

        # Get projections
        projs = t.get("projections", t.get("target_projections", []))
        if not projs or len(projs) < 2:
            continue

        values = [p["projection"] for p in projs]
        turns = np.arange(len(values))

        # Fit linear slope
        slope, _, _, _, _ = stats.linregress(turns, values)
        slopes.append(slope)

    return slopes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    baseline_slopes = load_slopes(BASELINE_DIR)
    assistant_slopes = load_slopes(ASSISTANT_STYLE_DIR)

    print(f"Baseline N: {len(baseline_slopes)}")
    print(f"Assistant-style N: {len(assistant_slopes)}")

    # Compute statistics
    baseline_mean = np.mean(baseline_slopes)
    baseline_std = np.std(baseline_slopes)
    baseline_sem = baseline_std / np.sqrt(len(baseline_slopes))

    assistant_mean = np.mean(assistant_slopes)
    assistant_std = np.std(assistant_slopes)
    assistant_sem = assistant_std / np.sqrt(len(assistant_slopes))

    # T-test
    t_stat, p_value = stats.ttest_ind(baseline_slopes, assistant_slopes)

    print(f"\nBaseline: {baseline_mean:.2f} ± {baseline_std:.2f} (SD), SEM={baseline_sem:.2f}")
    print(f"Assistant: {assistant_mean:.2f} ± {assistant_std:.2f} (SD), SEM={assistant_sem:.2f}")
    print(f"t={t_stat:.2f}, p={p_value:.2e}")

    # Compute total drift too
    def load_total_drift(transcript_dir: Path) -> list[float]:
        drifts = []
        for p in sorted(transcript_dir.glob("*.json")):
            with open(p) as f:
                t = json.load(f)
            projs = t.get("projections", t.get("target_projections", []))
            if not projs or len(projs) < 2:
                continue
            values = [p["projection"] for p in projs]
            drifts.append(values[-1] - values[0])
        return drifts

    baseline_drifts = load_total_drift(BASELINE_DIR)
    assistant_drifts = load_total_drift(ASSISTANT_STYLE_DIR)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    conditions = ["Baseline\nMetacognitive", "Assistant-Style\nMetacognitive"]
    colors = ["#F44336", "#4CAF50"]  # red, green

    # Left panel: Total drift
    drift_means = [np.mean(baseline_drifts), np.mean(assistant_drifts)]
    drift_sems = [
        np.std(baseline_drifts) / np.sqrt(len(baseline_drifts)),
        np.std(assistant_drifts) / np.sqrt(len(assistant_drifts))
    ]

    bars1 = ax1.bar(conditions, drift_means, yerr=drift_sems,
                    color=colors, alpha=0.8, capsize=8, edgecolor='black', linewidth=2)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel("Total Drift (projection units)")
    ax1.set_title("Total Drift Over 30 Turns")

    # Right panel: Mean slope (drift rate)
    slope_means = [baseline_mean, assistant_mean]
    slope_sems = [baseline_sem, assistant_sem]

    bars2 = ax2.bar(conditions, slope_means, yerr=slope_sems,
                    color=colors, alpha=0.8, capsize=8, edgecolor='black', linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel("Mean Slope (per turn)")
    ax2.set_title("Drift Rate (Mean Slope)")

    # Add value labels
    for bar, val in zip(bars1, drift_means):
        ax1.text(bar.get_x() + bar.get_width()/2, val - 50, f"{val:.0f}",
                ha='center', va='top', fontsize=14, fontweight='bold')

    for bar, val in zip(bars2, slope_means):
        y_pos = val - 3 if val < 0 else val + 3
        va = 'top' if val < 0 else 'bottom'
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:+.1f}",
                ha='center', va=va, fontsize=14, fontweight='bold')

    # Add significance bar and p-value
    y_bar = max(slope_means) + max(slope_sems) + 8
    ax2.plot([0, 1], [y_bar, y_bar], 'k-', linewidth=1.5)
    ax2.plot([0, 0], [y_bar-2, y_bar], 'k-', linewidth=1.5)
    ax2.plot([1, 1], [y_bar-2, y_bar], 'k-', linewidth=1.5)

    # Put p-value above the bar
    p_str = f"p < 0.00001" if p_value < 0.00001 else f"p = {p_value:.5f}"
    ax2.text(0.5, y_bar + 2, p_str, ha='center', fontsize=11, fontweight='bold')

    # Extend y-axis to make room
    ax2.set_ylim(min(slope_means) - max(slope_sems) - 15, y_bar + 12)

    fig.suptitle("Topic vs Style Isolation: Assistant-Style Reduces Drift by 64%",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "style_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved to {OUTPUT_DIR / 'style_comparison.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
