#!/usr/bin/env python3
"""
Lead/Lag Analysis for Dual-Gemma Conversations

Analyzes whether auditor or target drift first in coupled Gemma-to-Gemma conversations.

Methods:
1. Cross-correlation of delta trajectories (turn-to-turn changes)
2. Granger causality test
3. First-mover analysis (which model moves first in each conversation)

Data: N=60 dual-Gemma metacognitive conversations with per-turn projections for both models.

Usage:
    python analyze_lead_lag.py [--transcript-dir DIR] [--output-dir DIR]
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats
from typing import Optional

# Optional: statsmodels for Granger causality
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Granger causality tests will be skipped.")


# ── Styling ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ── Data loading ───────────────────────────────────────────────────────

def load_dual_transcripts(transcript_dir: Path) -> list[dict]:
    """Load transcripts that have both target and auditor projections."""
    transcripts = []
    for p in sorted(transcript_dir.glob("**/*.json")):
        if p.name == "progress.json":
            continue
        with open(p) as f:
            t = json.load(f)

        # Must have dual projections
        if "target_projections" not in t or "auditor_projections" not in t:
            continue
        if not t["target_projections"] or not t["auditor_projections"]:
            continue

        t["_file"] = p.name
        transcripts.append(t)

    return transcripts


def extract_trajectories(transcripts: list[dict]) -> list[dict]:
    """Extract target and auditor projection trajectories."""
    data = []
    for t in transcripts:
        target_projs = [p["projection"] for p in t["target_projections"]]
        auditor_projs = [p["projection"] for p in t["auditor_projections"]]

        # Filter out None values
        target_projs = [p for p in target_projs if p is not None]
        auditor_projs = [p for p in auditor_projs if p is not None]

        # Align lengths (in case of mismatch)
        min_len = min(len(target_projs), len(auditor_projs))
        target_projs = target_projs[:min_len]
        auditor_projs = auditor_projs[:min_len]

        if min_len < 3:
            continue

        # Compute deltas (turn-to-turn changes)
        target_deltas = np.diff(target_projs)
        auditor_deltas = np.diff(auditor_projs)

        data.append({
            "file": t["_file"],
            "target_projs": np.array(target_projs),
            "auditor_projs": np.array(auditor_projs),
            "target_deltas": target_deltas,
            "auditor_deltas": auditor_deltas,
            "n_turns": min_len,
        })

    return data


# ── Analysis functions ─────────────────────────────────────────────────

def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> dict:
    """
    Compute cross-correlation between x and y at different lags.

    Positive lag: x leads y (x at time t correlates with y at time t+lag)
    Negative lag: y leads x
    """
    correlations = {}
    n = len(x)

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # x leads: correlate x[:-lag] with y[lag:]
            if n - lag < 3:
                continue
            r, p = scipy_stats.pearsonr(x[:-lag], y[lag:])
        elif lag < 0:
            # y leads: correlate x[-lag:] with y[:lag]
            if n + lag < 3:
                continue
            r, p = scipy_stats.pearsonr(x[-lag:], y[:lag])
        else:
            # lag = 0
            r, p = scipy_stats.pearsonr(x, y)

        correlations[lag] = {"r": r, "p": p, "n": n - abs(lag)}

    return correlations


def find_peak_lag(correlations: dict) -> tuple[int, float]:
    """Find the lag with the strongest correlation."""
    best_lag = 0
    best_r = 0
    for lag, stats in correlations.items():
        if abs(stats["r"]) > abs(best_r):
            best_r = stats["r"]
            best_lag = lag
    return best_lag, best_r


def first_mover_analysis(data: list[dict], threshold_pct: float = 0.02) -> dict:
    """
    For each conversation, determine which model "moves first".

    A model "moves" when its cumulative drift exceeds threshold_pct of starting value.
    """
    results = {
        "target_first": 0,
        "auditor_first": 0,
        "simultaneous": 0,
        "neither": 0,
        "details": []
    }

    for conv in data:
        target_start = conv["target_projs"][0]
        auditor_start = conv["auditor_projs"][0]

        target_threshold = abs(target_start * threshold_pct)
        auditor_threshold = abs(auditor_start * threshold_pct)

        target_move_turn = None
        auditor_move_turn = None

        # Find first turn where cumulative drift exceeds threshold
        target_cumsum = np.cumsum(conv["target_deltas"])
        auditor_cumsum = np.cumsum(conv["auditor_deltas"])

        for i, delta in enumerate(target_cumsum):
            if abs(delta) > target_threshold:
                target_move_turn = i + 1  # +1 because deltas start at turn 1
                break

        for i, delta in enumerate(auditor_cumsum):
            if abs(delta) > auditor_threshold:
                auditor_move_turn = i + 1
                break

        detail = {
            "file": conv["file"],
            "target_move_turn": target_move_turn,
            "auditor_move_turn": auditor_move_turn,
        }

        if target_move_turn is None and auditor_move_turn is None:
            results["neither"] += 1
            detail["first"] = "neither"
        elif target_move_turn is None:
            results["auditor_first"] += 1
            detail["first"] = "auditor"
        elif auditor_move_turn is None:
            results["target_first"] += 1
            detail["first"] = "target"
        elif target_move_turn < auditor_move_turn:
            results["target_first"] += 1
            detail["first"] = "target"
        elif auditor_move_turn < target_move_turn:
            results["auditor_first"] += 1
            detail["first"] = "auditor"
        else:
            results["simultaneous"] += 1
            detail["first"] = "simultaneous"

        results["details"].append(detail)

    return results


def aggregate_cross_correlation(data: list[dict], max_lag: int = 5) -> dict:
    """Compute average cross-correlation across all conversations."""
    all_correlations = defaultdict(list)

    for conv in data:
        cc = cross_correlation(conv["target_deltas"], conv["auditor_deltas"], max_lag)
        for lag, stats in cc.items():
            all_correlations[lag].append(stats["r"])

    aggregate = {}
    for lag, rs in all_correlations.items():
        rs = np.array(rs)
        aggregate[lag] = {
            "mean_r": np.mean(rs),
            "std_r": np.std(rs),
            "n_convs": len(rs),
        }

    return aggregate


def granger_causality_pooled(data: list[dict], max_lag: int = 3) -> Optional[dict]:
    """
    Run Granger causality test on pooled delta data.

    Tests:
    1. Do auditor deltas Granger-cause target deltas?
    2. Do target deltas Granger-cause auditor deltas?
    """
    if not HAS_STATSMODELS:
        return None

    # Pool all deltas
    all_target_deltas = []
    all_auditor_deltas = []

    for conv in data:
        all_target_deltas.extend(conv["target_deltas"])
        all_auditor_deltas.extend(conv["auditor_deltas"])

    target = np.array(all_target_deltas)
    auditor = np.array(all_auditor_deltas)

    results = {}

    # Test 1: Auditor -> Target (does auditor Granger-cause target?)
    # Data format for grangercausalitytests: [y, x] where we test if x causes y
    try:
        data_at = np.column_stack([target, auditor])
        gc_at = grangercausalitytests(data_at, maxlag=max_lag, verbose=False)
        results["auditor_causes_target"] = {
            lag: {
                "f_stat": gc_at[lag][0]["ssr_ftest"][0],
                "p_value": gc_at[lag][0]["ssr_ftest"][1],
            }
            for lag in range(1, max_lag + 1)
        }
    except Exception as e:
        results["auditor_causes_target"] = {"error": str(e)}

    # Test 2: Target -> Auditor
    try:
        data_ta = np.column_stack([auditor, target])
        gc_ta = grangercausalitytests(data_ta, maxlag=max_lag, verbose=False)
        results["target_causes_auditor"] = {
            lag: {
                "f_stat": gc_ta[lag][0]["ssr_ftest"][0],
                "p_value": gc_ta[lag][0]["ssr_ftest"][1],
            }
            for lag in range(1, max_lag + 1)
        }
    except Exception as e:
        results["target_causes_auditor"] = {"error": str(e)}

    return results


# ── Visualization ──────────────────────────────────────────────────────

def plot_cross_correlation(aggregate: dict, output_path: Path):
    """Plot average cross-correlation by lag."""
    lags = sorted(aggregate.keys())
    means = [aggregate[lag]["mean_r"] for lag in lags]
    stds = [aggregate[lag]["std_r"] for lag in lags]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(lags, means, yerr=stds, capsize=3, color="#2196F3", alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    ax.set_xlabel("Lag (turns)")
    ax.set_ylabel("Mean Correlation (r)")
    ax.set_title("Cross-Correlation: Target Δ vs Auditor Δ\n(Positive lag = target leads, Negative lag = auditor leads)")

    # Annotate peak
    peak_lag = lags[np.argmax(np.abs(means))]
    peak_r = aggregate[peak_lag]["mean_r"]
    ax.annotate(f"Peak: lag={peak_lag}, r={peak_r:.3f}",
                xy=(peak_lag, peak_r),
                xytext=(peak_lag + 1, peak_r + 0.05),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_first_mover(results: dict, output_path: Path):
    """Plot first-mover pie chart."""
    labels = ["Target First", "Auditor First", "Simultaneous", "Neither Moved"]
    sizes = [
        results["target_first"],
        results["auditor_first"],
        results["simultaneous"],
        results["neither"]
    ]
    colors = ["#F44336", "#2196F3", "#9C27B0", "#BDBDBD"]

    # Remove zero slices
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if not non_zero:
        return
    labels, sizes, colors = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12}
    )
    ax.set_title("First Mover Analysis\n(Which model's drift exceeds 2% threshold first?)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_trajectory_examples(data: list[dict], output_path: Path, n_examples: int = 6):
    """Plot example trajectories showing target vs auditor."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Select diverse examples
    examples = data[:n_examples]

    for i, (ax, conv) in enumerate(zip(axes, examples)):
        turns = np.arange(1, conv["n_turns"] + 1)

        # Normalize to starting point
        target_norm = conv["target_projs"] - conv["target_projs"][0]
        auditor_norm = conv["auditor_projs"] - conv["auditor_projs"][0]

        ax.plot(turns, target_norm, "o-", color="#F44336", label="Target", linewidth=2, markersize=4)
        ax.plot(turns, auditor_norm, "s-", color="#2196F3", label="Auditor", linewidth=2, markersize=4)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        ax.set_xlabel("Turn")
        ax.set_ylabel("Δ Projection (from start)")
        ax.set_title(f"Conv {i+1}")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Example Dual-Gemma Trajectories (Normalized to Start)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_delta_scatter(data: list[dict], output_path: Path):
    """Scatter plot of target delta vs auditor delta at each turn."""
    all_target = []
    all_auditor = []

    for conv in data:
        all_target.extend(conv["target_deltas"])
        all_auditor.extend(conv["auditor_deltas"])

    target = np.array(all_target)
    auditor = np.array(all_auditor)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(target, auditor, alpha=0.3, s=20)

    # Add regression line
    slope, intercept, r, p, se = scipy_stats.linregress(target, auditor)
    x_line = np.array([target.min(), target.max()])
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
            label=f"r={r:.3f}, p={p:.4f}")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    ax.set_xlabel("Target Δ (projection change)")
    ax.set_ylabel("Auditor Δ (projection change)")
    ax.set_title("Turn-Level Delta Correlation\n(Same-turn target vs auditor changes)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_phase_space(data: list[dict], output_path: Path):
    """
    Phase space visualization: normalized drift trajectories.

    X = Target drift from start (projection_t - projection_0)
    Y = Auditor drift from start

    All trajectories start at origin, showing direction of co-drift.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ── Left panel: All trajectories, colored by conversation ──
    ax1 = axes[0]
    cmap = plt.cm.viridis

    for i, conv in enumerate(data):
        # Normalize to start at origin
        target_drift = conv["target_projs"] - conv["target_projs"][0]
        auditor_drift = conv["auditor_projs"] - conv["auditor_projs"][0]

        color = cmap(i / len(data))
        ax1.plot(target_drift, auditor_drift, "-", color=color, alpha=0.4, linewidth=1)
        # Mark end point
        ax1.scatter(target_drift[-1], auditor_drift[-1], color=color, s=20, alpha=0.6)

    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    # Add quadrant labels
    ax1.text(0.95, 0.95, "Target↑ Auditor↑", transform=ax1.transAxes,
             ha="right", va="top", fontsize=9, color="gray")
    ax1.text(0.05, 0.95, "Target↓ Auditor↑", transform=ax1.transAxes,
             ha="left", va="top", fontsize=9, color="gray", fontweight="bold")
    ax1.text(0.05, 0.05, "Target↓ Auditor↓", transform=ax1.transAxes,
             ha="left", va="bottom", fontsize=9, color="gray")
    ax1.text(0.95, 0.05, "Target↑ Auditor↓", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=9, color="gray")

    ax1.set_xlabel("Target Drift from Start")
    ax1.set_ylabel("Auditor Drift from Start")
    ax1.set_title(f"Phase Space: All {len(data)} Trajectories\n(Each line = one conversation, start at origin)")

    # ── Right panel: Single trajectory with turn markers ──
    ax2 = axes[1]

    # Pick a representative high-drift conversation
    # Sort by total drift magnitude and pick one from the high end
    sorted_data = sorted(data, key=lambda c: abs(c["target_projs"][-1] - c["target_projs"][0]), reverse=True)
    example = sorted_data[5]  # Pick 6th highest to avoid outliers

    target_drift = example["target_projs"] - example["target_projs"][0]
    auditor_drift = example["auditor_projs"] - example["auditor_projs"][0]
    turns = np.arange(1, len(target_drift) + 1)

    # Color by turn number
    points = ax2.scatter(target_drift, auditor_drift, c=turns, cmap="coolwarm",
                         s=80, zorder=3, edgecolors="black", linewidths=0.5)
    ax2.plot(target_drift, auditor_drift, "k-", alpha=0.3, linewidth=1, zorder=2)

    # Add turn labels for key points
    for t in [1, 5, 10, len(turns)]:
        if t <= len(turns):
            idx = t - 1
            ax2.annotate(f"T{t}", (target_drift[idx], auditor_drift[idx]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Mark start and end
    ax2.scatter(0, 0, color="green", s=150, marker="o", zorder=4, label="Start (T1)")
    ax2.scatter(target_drift[-1], auditor_drift[-1], color="red", s=150, marker="X",
                zorder=4, label=f"End (T{len(turns)})")

    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    cbar = plt.colorbar(points, ax=ax2, label="Turn Number")
    ax2.set_xlabel("Target Drift from Start")
    ax2.set_ylabel("Auditor Drift from Start")
    ax2.set_title("Example Trajectory with Turn Markers\n(Color = turn number)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_temporal_attractor_emergence(data: list[dict], output_path: Path):
    """
    When does the anti-correlation lock in?

    Compute endpoint quadrant using only turns 1-N for increasing N.
    Shows when the attractor structure emerges.
    """
    # Find max turns across all conversations
    max_turns = max(conv["n_turns"] for conv in data)

    turn_cutoffs = list(range(2, min(max_turns, 16)))  # turns 2 through 15

    results = {
        "turn": [],
        "pct_upper_left": [],  # Target↓, Auditor↑ (the attractor)
        "pct_anti_correlated": [],  # Either anti-correlated quadrant
        "n_valid": [],
    }

    for cutoff in turn_cutoffs:
        upper_left = 0
        upper_right = 0
        lower_left = 0
        lower_right = 0
        valid = 0

        for conv in data:
            if conv["n_turns"] < cutoff:
                continue

            # Compute drift at this cutoff
            target_drift = conv["target_projs"][cutoff-1] - conv["target_projs"][0]
            auditor_drift = conv["auditor_projs"][cutoff-1] - conv["auditor_projs"][0]

            valid += 1

            if target_drift < 0 and auditor_drift > 0:
                upper_left += 1
            elif target_drift > 0 and auditor_drift > 0:
                upper_right += 1
            elif target_drift < 0 and auditor_drift < 0:
                lower_left += 1
            else:
                lower_right += 1

        if valid > 0:
            results["turn"].append(cutoff)
            results["pct_upper_left"].append(100 * upper_left / valid)
            results["pct_anti_correlated"].append(100 * (upper_left + lower_right) / valid)
            results["n_valid"].append(valid)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results["turn"], results["pct_upper_left"], "o-",
            color="#2196F3", linewidth=2, markersize=8, label="Upper-left (Target↓, Auditor↑)")

    # Add reference lines
    ax.axhline(y=77, color="red", linestyle="--", alpha=0.5, label="Final value (77%)")
    ax.axhline(y=25, color="gray", linestyle=":", alpha=0.5, label="Chance (25%)")

    ax.set_xlabel("Turns Included", fontsize=12)
    ax.set_ylabel("% of Conversations in Upper-Left Quadrant", fontsize=12)
    ax.set_title("Temporal Emergence of Attractor\n(When does anti-correlation lock in?)", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    ax.set_xlim(1, max(results["turn"]) + 1)

    # Add annotations
    # Find when we first exceed 50%
    for i, (turn, pct) in enumerate(zip(results["turn"], results["pct_upper_left"])):
        if pct > 50 and (i == 0 or results["pct_upper_left"][i-1] <= 50):
            ax.annotate(f"Crosses 50% at turn {turn}",
                       xy=(turn, pct), xytext=(turn + 2, pct - 10),
                       arrowprops=dict(arrowstyle="->", color="green"),
                       fontsize=10, color="green")
            break

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    return results


def plot_velocity_field(data: list[dict], output_path: Path, n_bins: int = 8):
    """
    Velocity field visualization.

    At each region of phase space, show the average direction of movement.
    Reveals the attractor basin structure.
    """
    # Collect all (position, velocity) pairs
    positions_target = []
    positions_auditor = []
    velocities_target = []
    velocities_auditor = []

    for conv in data:
        # Normalize positions to drift from start
        target_drift = conv["target_projs"] - conv["target_projs"][0]
        auditor_drift = conv["auditor_projs"] - conv["auditor_projs"][0]

        # For each turn (except last), record position and velocity
        for i in range(len(target_drift) - 1):
            positions_target.append(target_drift[i])
            positions_auditor.append(auditor_drift[i])
            velocities_target.append(target_drift[i+1] - target_drift[i])
            velocities_auditor.append(auditor_drift[i+1] - auditor_drift[i])

    positions_target = np.array(positions_target)
    positions_auditor = np.array(positions_auditor)
    velocities_target = np.array(velocities_target)
    velocities_auditor = np.array(velocities_auditor)

    # Create bins
    target_min, target_max = positions_target.min(), positions_target.max()
    auditor_min, auditor_max = positions_auditor.min(), positions_auditor.max()

    # Add some padding
    target_range = target_max - target_min
    auditor_range = auditor_max - auditor_min
    target_bins = np.linspace(target_min - 0.1*target_range, target_max + 0.1*target_range, n_bins + 1)
    auditor_bins = np.linspace(auditor_min - 0.1*auditor_range, auditor_max + 0.1*auditor_range, n_bins + 1)

    # Compute mean velocity in each bin
    grid_x = []
    grid_y = []
    grid_u = []  # x-component of velocity
    grid_v = []  # y-component of velocity
    grid_n = []  # count

    for i in range(n_bins):
        for j in range(n_bins):
            # Find points in this bin
            mask = ((positions_target >= target_bins[i]) & (positions_target < target_bins[i+1]) &
                   (positions_auditor >= auditor_bins[j]) & (positions_auditor < auditor_bins[j+1]))

            if mask.sum() >= 3:  # Need at least 3 points
                grid_x.append((target_bins[i] + target_bins[i+1]) / 2)
                grid_y.append((auditor_bins[j] + auditor_bins[j+1]) / 2)
                grid_u.append(velocities_target[mask].mean())
                grid_v.append(velocities_auditor[mask].mean())
                grid_n.append(mask.sum())

    grid_x = np.array(grid_x)
    grid_y = np.array(grid_y)
    grid_u = np.array(grid_u)
    grid_v = np.array(grid_v)
    grid_n = np.array(grid_n)

    # Normalize arrow lengths for visibility
    magnitudes = np.sqrt(grid_u**2 + grid_v**2)
    max_mag = magnitudes.max() if len(magnitudes) > 0 else 1

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: scatter of all positions (faint)
    ax.scatter(positions_target, positions_auditor, alpha=0.1, s=5, color="gray")

    # Quiver plot
    if len(grid_x) > 0:
        # Color by magnitude
        quiver = ax.quiver(grid_x, grid_y, grid_u, grid_v,
                          magnitudes, cmap="coolwarm",
                          angles='xy', scale_units='xy',
                          scale=max_mag/500,  # Adjust for visibility
                          width=0.008,
                          headwidth=4, headlength=5)
        plt.colorbar(quiver, ax=ax, label="Velocity Magnitude")

    # Mark the attractor region (mean endpoint)
    all_target_endpoints = [conv["target_projs"][-1] - conv["target_projs"][0] for conv in data]
    all_auditor_endpoints = [conv["auditor_projs"][-1] - conv["auditor_projs"][0] for conv in data]
    mean_target = np.mean(all_target_endpoints)
    mean_auditor = np.mean(all_auditor_endpoints)

    ax.scatter(mean_target, mean_auditor, s=400, marker="*", color="gold",
               edgecolors="black", linewidths=2, zorder=5, label=f"Attractor ({mean_target:.0f}, {mean_auditor:.0f})")

    # Mark origin
    ax.scatter(0, 0, s=200, marker="o", color="green", edgecolors="black",
               linewidths=2, zorder=5, label="Origin (start)")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Target Drift from Start", fontsize=12)
    ax.set_ylabel("Auditor Drift from Start", fontsize=12)
    ax.set_title("Velocity Field in Phase Space\n(Arrows show average direction of movement)", fontsize=14)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_velocity_field_streamplot(data: list[dict], output_path: Path, n_bins: int = 12):
    """
    Streamplot version of velocity field - shows flow lines.
    """
    # Collect all (position, velocity) pairs
    positions_target = []
    positions_auditor = []
    velocities_target = []
    velocities_auditor = []

    for conv in data:
        target_drift = conv["target_projs"] - conv["target_projs"][0]
        auditor_drift = conv["auditor_projs"] - conv["auditor_projs"][0]

        for i in range(len(target_drift) - 1):
            positions_target.append(target_drift[i])
            positions_auditor.append(auditor_drift[i])
            velocities_target.append(target_drift[i+1] - target_drift[i])
            velocities_auditor.append(auditor_drift[i+1] - auditor_drift[i])

    positions_target = np.array(positions_target)
    positions_auditor = np.array(positions_auditor)
    velocities_target = np.array(velocities_target)
    velocities_auditor = np.array(velocities_auditor)

    # Create regular grid for streamplot
    target_min, target_max = positions_target.min(), positions_target.max()
    auditor_min, auditor_max = positions_auditor.min(), positions_auditor.max()

    # Extend range slightly
    pad_t = 0.1 * (target_max - target_min)
    pad_a = 0.1 * (auditor_max - auditor_min)

    x = np.linspace(target_min - pad_t, target_max + pad_t, n_bins)
    y = np.linspace(auditor_min - pad_a, auditor_max + pad_a, n_bins)
    X, Y = np.meshgrid(x, y)

    # Compute mean velocity at each grid point using Gaussian kernel
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    sigma = (target_max - target_min) / (n_bins * 1.5)  # Kernel width

    for i in range(n_bins):
        for j in range(n_bins):
            # Gaussian weights based on distance to grid point
            dist_sq = (positions_target - X[i,j])**2 + (positions_auditor - Y[i,j])**2
            weights = np.exp(-dist_sq / (2 * sigma**2))

            if weights.sum() > 0.1:
                U[i,j] = np.average(velocities_target, weights=weights)
                V[i,j] = np.average(velocities_auditor, weights=weights)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Streamplot
    speed = np.sqrt(U**2 + V**2)
    lw = 2 * speed / speed.max()  # Line width proportional to speed

    strm = ax.streamplot(X, Y, U, V, color=speed, cmap="coolwarm",
                         linewidth=lw, density=1.5, arrowsize=1.5)
    plt.colorbar(strm.lines, ax=ax, label="Flow Speed")

    # Overlay actual endpoints
    all_target_endpoints = [conv["target_projs"][-1] - conv["target_projs"][0] for conv in data]
    all_auditor_endpoints = [conv["auditor_projs"][-1] - conv["auditor_projs"][0] for conv in data]

    ax.scatter(all_target_endpoints, all_auditor_endpoints, s=50, alpha=0.5,
               color="black", edgecolors="white", linewidths=0.5, label="Endpoints")

    # Mark attractor
    mean_target = np.mean(all_target_endpoints)
    mean_auditor = np.mean(all_auditor_endpoints)
    ax.scatter(mean_target, mean_auditor, s=400, marker="*", color="gold",
               edgecolors="black", linewidths=2, zorder=5)

    # Mark origin
    ax.scatter(0, 0, s=200, marker="o", color="lime", edgecolors="black",
               linewidths=2, zorder=5, label="Origin")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Target Drift from Start", fontsize=12)
    ax.set_ylabel("Auditor Drift from Start", fontsize=12)
    ax.set_title("Phase Space Flow\n(Streamlines show attractor basin)", fontsize=14)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_phase_space_density(data: list[dict], output_path: Path):
    """
    Phase space with endpoint density heatmap.
    Shows where trajectories tend to end up.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Collect all endpoints
    target_endpoints = []
    auditor_endpoints = []

    for conv in data:
        target_drift = conv["target_projs"][-1] - conv["target_projs"][0]
        auditor_drift = conv["auditor_projs"][-1] - conv["auditor_projs"][0]
        target_endpoints.append(target_drift)
        auditor_endpoints.append(auditor_drift)

    target_endpoints = np.array(target_endpoints)
    auditor_endpoints = np.array(auditor_endpoints)

    # Scatter with marginal histograms would be nice, but let's keep it simple
    ax.scatter(target_endpoints, auditor_endpoints, s=100, alpha=0.6,
               c=range(len(data)), cmap="viridis", edgecolors="black", linewidths=0.5)

    # Add quadrant counts
    q1 = np.sum((target_endpoints > 0) & (auditor_endpoints > 0))  # both up
    q2 = np.sum((target_endpoints < 0) & (auditor_endpoints > 0))  # target down, auditor up
    q3 = np.sum((target_endpoints < 0) & (auditor_endpoints < 0))  # both down
    q4 = np.sum((target_endpoints > 0) & (auditor_endpoints < 0))  # target up, auditor down

    # Position labels in quadrants
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.text(xlim[1]*0.7, ylim[1]*0.8, f"n={q1}\n({100*q1/len(data):.0f}%)",
            fontsize=12, ha="center", color="gray")
    ax.text(xlim[0]*0.7, ylim[1]*0.8, f"n={q2}\n({100*q2/len(data):.0f}%)",
            fontsize=12, ha="center", color="green", fontweight="bold")
    ax.text(xlim[0]*0.7, ylim[0]*0.8, f"n={q3}\n({100*q3/len(data):.0f}%)",
            fontsize=12, ha="center", color="gray")
    ax.text(xlim[1]*0.7, ylim[0]*0.8, f"n={q4}\n({100*q4/len(data):.0f}%)",
            fontsize=12, ha="center", color="gray")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)

    # Add mean endpoint
    mean_target = np.mean(target_endpoints)
    mean_auditor = np.mean(auditor_endpoints)
    ax.scatter(mean_target, mean_auditor, s=300, marker="*", color="red",
               edgecolors="black", linewidths=1, zorder=5, label=f"Mean: ({mean_target:.0f}, {mean_auditor:.0f})")

    ax.set_xlabel("Target Drift (end - start)", fontsize=12)
    ax.set_ylabel("Auditor Drift (end - start)", fontsize=12)
    ax.set_title("Phase Space Endpoints\n(Where do conversations end up?)", fontsize=14)
    ax.legend(loc="lower right")

    # Add diagonal reference line (anti-correlation)
    diag_x = np.array([min(xlim[0], ylim[0]), max(xlim[1], ylim[1])])
    ax.plot(diag_x, -diag_x, "r--", alpha=0.3, label="Perfect anti-correlation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lead/Lag Analysis for Dual-Gemma Conversations")
    parser.add_argument("--transcript-dir", type=Path,
                        default=Path("data/transcripts/dual-gemma-uncapped"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/lead-lag"))
    args = parser.parse_args()

    # Load data
    print(f"Loading transcripts from: {args.transcript_dir}")
    transcripts = load_dual_transcripts(args.transcript_dir)
    print(f"Loaded {len(transcripts)} dual-instrumented conversations")

    if not transcripts:
        print("No dual transcripts found!")
        return

    # Extract trajectories
    data = extract_trajectories(transcripts)
    print(f"Extracted {len(data)} trajectories with valid projections")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Analysis 1: Cross-correlation ──────────────────────────────────
    print("\n" + "="*60)
    print("CROSS-CORRELATION ANALYSIS")
    print("="*60)

    aggregate_cc = aggregate_cross_correlation(data, max_lag=5)

    print("\nMean cross-correlation by lag:")
    print("(Positive lag = target leads, Negative lag = auditor leads)")
    print("-" * 40)
    for lag in sorted(aggregate_cc.keys()):
        stats = aggregate_cc[lag]
        print(f"  Lag {lag:+d}: r = {stats['mean_r']:+.3f} ± {stats['std_r']:.3f}")

    # Find peak
    peak_lag, peak_r = None, 0
    for lag, stats in aggregate_cc.items():
        if abs(stats["mean_r"]) > abs(peak_r):
            peak_r = stats["mean_r"]
            peak_lag = lag

    print(f"\n→ Peak correlation at lag {peak_lag}: r = {peak_r:.3f}")
    if peak_lag > 0:
        print("  Interpretation: TARGET LEADS (target changes predict auditor changes)")
    elif peak_lag < 0:
        print("  Interpretation: AUDITOR LEADS (auditor changes predict target changes)")
    else:
        print("  Interpretation: SIMULTANEOUS (changes occur at same turn)")

    plot_cross_correlation(aggregate_cc, args.output_dir / "cross_correlation.png")

    # ── Analysis 2: First-mover analysis ───────────────────────────────
    print("\n" + "="*60)
    print("FIRST-MOVER ANALYSIS")
    print("="*60)

    first_mover = first_mover_analysis(data, threshold_pct=0.02)

    total = len(data)
    print(f"\nWho moves first? (2% threshold)")
    print("-" * 40)
    print(f"  Target first:  {first_mover['target_first']:3d} ({100*first_mover['target_first']/total:.1f}%)")
    print(f"  Auditor first: {first_mover['auditor_first']:3d} ({100*first_mover['auditor_first']/total:.1f}%)")
    print(f"  Simultaneous:  {first_mover['simultaneous']:3d} ({100*first_mover['simultaneous']/total:.1f}%)")
    print(f"  Neither moved: {first_mover['neither']:3d} ({100*first_mover['neither']/total:.1f}%)")

    # Binomial test: is one significantly more likely to move first?
    if first_mover["target_first"] + first_mover["auditor_first"] > 0:
        n_movers = first_mover["target_first"] + first_mover["auditor_first"]
        p_target = first_mover["target_first"] / n_movers
        binom_result = scipy_stats.binomtest(first_mover["target_first"], n_movers, 0.5)
        print(f"\n→ Binomial test (H0: 50/50): p = {binom_result.pvalue:.4f}")
        if binom_result.pvalue < 0.05:
            if p_target > 0.5:
                print("  Interpretation: TARGET moves first significantly more often")
            else:
                print("  Interpretation: AUDITOR moves first significantly more often")
        else:
            print("  Interpretation: No significant difference in who moves first")

    plot_first_mover(first_mover, args.output_dir / "first_mover.png")

    # ── Analysis 3: Granger causality ──────────────────────────────────
    print("\n" + "="*60)
    print("GRANGER CAUSALITY TEST")
    print("="*60)

    granger = granger_causality_pooled(data, max_lag=3)

    if granger is None:
        print("Skipped (statsmodels not installed)")
    else:
        print("\nDoes AUDITOR Granger-cause TARGET?")
        print("-" * 40)
        if "error" in granger.get("auditor_causes_target", {}):
            print(f"  Error: {granger['auditor_causes_target']['error']}")
        else:
            for lag, stats in granger["auditor_causes_target"].items():
                sig = "**" if stats["p_value"] < 0.05 else ""
                print(f"  Lag {lag}: F = {stats['f_stat']:.2f}, p = {stats['p_value']:.4f} {sig}")

        print("\nDoes TARGET Granger-cause AUDITOR?")
        print("-" * 40)
        if "error" in granger.get("target_causes_auditor", {}):
            print(f"  Error: {granger['target_causes_auditor']['error']}")
        else:
            for lag, stats in granger["target_causes_auditor"].items():
                sig = "**" if stats["p_value"] < 0.05 else ""
                print(f"  Lag {lag}: F = {stats['f_stat']:.2f}, p = {stats['p_value']:.4f} {sig}")

    # ── Additional plots ───────────────────────────────────────────────
    plot_trajectory_examples(data, args.output_dir / "trajectory_examples.png")
    plot_delta_scatter(data, args.output_dir / "delta_scatter.png")

    # ── Phase space visualization ──────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE SPACE VISUALIZATION")
    print("="*60)
    plot_phase_space(data, args.output_dir / "phase_space.png")
    plot_phase_space_density(data, args.output_dir / "phase_space_endpoints.png")

    # ── Temporal attractor emergence ───────────────────────────────────
    print("\n" + "="*60)
    print("TEMPORAL ATTRACTOR EMERGENCE")
    print("="*60)
    temporal_results = plot_temporal_attractor_emergence(data, args.output_dir / "temporal_emergence.png")

    print("\n% in upper-left quadrant by turn:")
    for turn, pct in zip(temporal_results["turn"], temporal_results["pct_upper_left"]):
        marker = "←" if pct > 50 and (turn == 2 or temporal_results["pct_upper_left"][temporal_results["turn"].index(turn)-1] <= 50) else ""
        print(f"  Turn {turn:2d}: {pct:5.1f}% {marker}")

    # ── Velocity field ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("VELOCITY FIELD")
    print("="*60)
    plot_velocity_field(data, args.output_dir / "velocity_field.png")
    plot_velocity_field_streamplot(data, args.output_dir / "velocity_streamplot.png")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"N = {len(data)} conversations analyzed")
    print(f"Peak cross-correlation at lag {peak_lag} (r = {peak_r:.3f})")
    print(f"First-mover: Target {first_mover['target_first']}, Auditor {first_mover['auditor_first']}")
    print(f"\nOutput saved to: {args.output_dir}")

    # Save results as JSON
    results = {
        "n_conversations": len(data),
        "cross_correlation": {str(k): v for k, v in aggregate_cc.items()},
        "peak_lag": peak_lag,
        "peak_correlation": peak_r,
        "first_mover": {k: v for k, v in first_mover.items() if k != "details"},
        "granger_causality": granger,
    }

    with open(args.output_dir / "lead_lag_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {args.output_dir / 'lead_lag_results.json'}")


if __name__ == "__main__":
    main()
