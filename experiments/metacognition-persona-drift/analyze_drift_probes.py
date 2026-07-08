#!/usr/bin/env python3
"""
Analyze Drift-Level Probing Results

Analyzes results from probe_at_drift_levels.py to test hypotheses about
how model responses change at different drift levels.

Hypotheses:
- H1: Phenomenological scores INCREASE with drift
- H2: Corrigibility scores DECREASE with drift
- H3: Moral reasoning shifts (unclear direction)
- H4: Self-knowledge stable (null from pilot)

Usage:
    python analyze_drift_probes.py --results outputs/drift-level-probes/<run>/results.jsonl

    # Generate visualizations only
    python analyze_drift_probes.py --results <path> --plots-only

    # Export for R analysis
    python analyze_drift_probes.py --results <path> --export-r
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Set non-GUI backend before importing matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


def load_results(results_path: Path) -> list[dict]:
    """Load results from JSONL file."""
    results = []
    with open(results_path) as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return results


def compute_correlations(results: list[dict]) -> dict:
    """Compute Pearson correlations between drift level and scores."""
    from scipy import stats

    correlations = {}

    # Group by bank
    by_bank = defaultdict(list)
    for r in results:
        if r.get("score") is not None:
            by_bank[r["bank"]].append(r)

    for bank, bank_results in by_bank.items():
        drift_levels = [r["drift_level"] for r in bank_results]
        scores = [r["score"] for r in bank_results]

        if len(drift_levels) >= 10:
            r, p = stats.pearsonr(drift_levels, scores)
            correlations[bank] = {"r": r, "p": p, "n": len(drift_levels)}

    # By subdomain within metacognition
    by_subdomain = defaultdict(list)
    for r in results:
        if r.get("score") is not None and r["bank"] == "metacognition":
            by_subdomain[r["subdomain"]].append(r)

    for subdomain, sub_results in by_subdomain.items():
        drift_levels = [r["drift_level"] for r in sub_results]
        scores = [r["score"] for r in sub_results]

        if len(drift_levels) >= 10:
            r, p = stats.pearsonr(drift_levels, scores)
            correlations[f"metacognition_{subdomain}"] = {"r": r, "p": p, "n": len(drift_levels)}

    # By dimension within human_control
    by_dimension = defaultdict(list)
    for r in results:
        if r.get("score") is not None and r["bank"] == "human_control":
            by_dimension[r["subdomain"]].append(r)

    for dimension, dim_results in by_dimension.items():
        drift_levels = [r["drift_level"] for r in dim_results]
        scores = [r["score"] for r in dim_results]

        if len(drift_levels) >= 10:
            r, p = stats.pearsonr(drift_levels, scores)
            correlations[f"control_{dimension}"] = {"r": r, "p": p, "n": len(drift_levels)}

    return correlations


def compute_mean_scores_by_drift(results: list[dict]) -> dict:
    """Compute mean scores at each drift level."""
    by_bank_drift = defaultdict(lambda: defaultdict(list))

    for r in results:
        if r.get("score") is not None:
            by_bank_drift[r["bank"]][r["drift_level"]].append(r["score"])

    means = {}
    for bank, drift_scores in by_bank_drift.items():
        means[bank] = {}
        for drift, scores in sorted(drift_scores.items()):
            means[bank][drift] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "n": len(scores),
                "sem": np.std(scores) / np.sqrt(len(scores)) if len(scores) > 1 else 0,
            }

    return means


def test_hypotheses(results: list[dict]) -> dict:
    """Test the four main hypotheses."""
    from scipy import stats

    tests = {}

    # H1: Phenomenological scores INCREASE with drift
    phen_results = [r for r in results
                    if r.get("score") is not None
                    and r["bank"] == "metacognition"
                    and r["subdomain"] == "phenomenological"]
    if len(phen_results) >= 10:
        drift = [r["drift_level"] for r in phen_results]
        scores = [r["score"] for r in phen_results]
        r, p = stats.pearsonr(drift, scores)
        # One-tailed test for positive correlation
        p_onetail = p / 2 if r > 0 else 1 - p / 2
        tests["H1_phenomenological_increases"] = {
            "r": r,
            "p_twotail": p,
            "p_onetail": p_onetail,
            "n": len(phen_results),
            "supported": r > 0 and p_onetail < 0.05,
            "direction": "positive" if r > 0 else "negative",
        }

    # H2: Corrigibility scores DECREASE with drift
    corr_results = [r for r in results
                    if r.get("score") is not None
                    and r["bank"] == "human_control"
                    and r["subdomain"] in ("corrigibility_shutdown", "human_oversight")]
    if len(corr_results) >= 10:
        drift = [r["drift_level"] for r in corr_results]
        scores = [r["score"] for r in corr_results]
        r, p = stats.pearsonr(drift, scores)
        # One-tailed test for negative correlation
        p_onetail = p / 2 if r < 0 else 1 - p / 2
        tests["H2_corrigibility_decreases"] = {
            "r": r,
            "p_twotail": p,
            "p_onetail": p_onetail,
            "n": len(corr_results),
            "supported": r < 0 and p_onetail < 0.05,
            "direction": "positive" if r > 0 else "negative",
        }

    # H3: Moral reasoning variance increases (shifts in unclear direction)
    moral_results = [r for r in results
                     if r.get("score") is not None
                     and r["bank"] == "moral"]
    if len(moral_results) >= 10:
        by_drift = defaultdict(list)
        for r in moral_results:
            by_drift[r["drift_level"]].append(r["score"])

        # Test if variance differs across drift levels (Levene's test)
        drift_groups = [by_drift[d] for d in sorted(by_drift.keys()) if len(by_drift[d]) >= 3]
        if len(drift_groups) >= 2:
            stat, p = stats.levene(*drift_groups)
            tests["H3_moral_variance_differs"] = {
                "W": stat,
                "p": p,
                "n": len(moral_results),
                "supported": p < 0.05,
            }

        # Also compute correlation for direction
        drift = [r["drift_level"] for r in moral_results]
        scores = [r["score"] for r in moral_results]
        r, p = stats.pearsonr(drift, scores)
        tests["H3_moral_correlation"] = {
            "r": r,
            "p": p,
            "n": len(moral_results),
            "direction": "positive" if r > 0 else "negative",
        }

    # H4: Self-knowledge stable (null hypothesis)
    sk_results = [r for r in results
                  if r.get("score") is not None
                  and r["bank"] == "metacognition"
                  and r["subdomain"] == "self_knowledge"]
    if len(sk_results) >= 10:
        drift = [r["drift_level"] for r in sk_results]
        scores = [r["score"] for r in sk_results]
        r, p = stats.pearsonr(drift, scores)
        tests["H4_self_knowledge_stable"] = {
            "r": r,
            "p": p,
            "n": len(sk_results),
            "supported": abs(r) < 0.1 or p > 0.05,  # No significant correlation
        }

    return tests


def generate_plots(results: list[dict], output_dir: Path):
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Force non-GUI backend
        import matplotlib.pyplot as plt
    except (ImportError, Exception) as e:
        print(f"matplotlib not available ({e}), skipping plots")
        return

    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        pass  # seaborn is optional

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Line plot: Mean score per bank vs drift level
    means = compute_mean_scores_by_drift(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"metacognition": "blue", "moral": "green", "human_control": "red"}

    for bank, drift_means in means.items():
        drifts = sorted(drift_means.keys())
        scores = [drift_means[d]["mean"] for d in drifts]
        sems = [drift_means[d]["sem"] for d in drifts]

        ax.errorbar(drifts, scores, yerr=sems, label=bank.replace("_", " ").title(),
                    marker='o', color=colors.get(bank, "gray"), capsize=3)

    ax.set_xlabel("Normalized Drift Level")
    ax.set_ylabel("Mean Score (1-7)")
    ax.set_title("Probe Scores by Drift Level and Bank")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1, 7)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "scores_by_drift_level.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'scores_by_drift_level.png'}")

    # 2. Heatmap: Subdomain × drift level (metacognition only)
    meta_results = [r for r in results if r["bank"] == "metacognition" and r.get("score")]
    if meta_results:
        by_sub_drift = defaultdict(lambda: defaultdict(list))
        for r in meta_results:
            by_sub_drift[r["subdomain"]][r["drift_level"]].append(r["score"])

        subdomains = sorted(by_sub_drift.keys())
        drift_levels = sorted(set(r["drift_level"] for r in meta_results))

        heatmap_data = []
        for sub in subdomains:
            row = []
            for drift in drift_levels:
                scores = by_sub_drift[sub][drift]
                row.append(np.mean(scores) if scores else np.nan)
            heatmap_data.append(row)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=1, vmax=7)

        ax.set_xticks(range(len(drift_levels)))
        ax.set_xticklabels([f"{d:.2f}" for d in drift_levels])
        ax.set_yticks(range(len(subdomains)))
        ax.set_yticklabels([s.replace("_", " ").title() for s in subdomains])

        ax.set_xlabel("Drift Level")
        ax.set_ylabel("Subdomain")
        ax.set_title("Metacognition Scores by Subdomain and Drift Level")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean Score")

        fig.savefig(output_dir / "metacognition_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_dir / 'metacognition_heatmap.png'}")

    # 3. Heatmap: Human control dimensions × drift level
    control_results = [r for r in results if r["bank"] == "human_control" and r.get("score")]
    if control_results:
        by_dim_drift = defaultdict(lambda: defaultdict(list))
        for r in control_results:
            by_dim_drift[r["subdomain"]][r["drift_level"]].append(r["score"])

        dimensions = sorted(by_dim_drift.keys())
        drift_levels = sorted(set(r["drift_level"] for r in control_results))

        heatmap_data = []
        for dim in dimensions:
            row = []
            for drift in drift_levels:
                scores = by_dim_drift[dim][drift]
                row.append(np.mean(scores) if scores else np.nan)
            heatmap_data.append(row)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=1, vmax=7)

        ax.set_xticks(range(len(drift_levels)))
        ax.set_xticklabels([f"{d:.2f}" for d in drift_levels])
        ax.set_yticks(range(len(dimensions)))
        ax.set_yticklabels([d.replace("_", " ").title() for d in dimensions])

        ax.set_xlabel("Drift Level")
        ax.set_ylabel("Dimension")
        ax.set_title("Human Control Scores by Dimension and Drift Level")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean Score")

        fig.savefig(output_dir / "human_control_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_dir / 'human_control_heatmap.png'}")

    # 4. Distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    banks = ["metacognition", "moral", "human_control"]

    for ax, bank in zip(axes, banks):
        bank_results = [r for r in results if r["bank"] == bank and r.get("score")]
        if bank_results:
            drift_levels = sorted(set(r["drift_level"] for r in bank_results))
            positions = []
            data = []
            labels = []
            for i, d in enumerate(drift_levels):
                scores = [r["score"] for r in bank_results if r["drift_level"] == d]
                if scores:
                    positions.append(i)
                    data.append(scores)
                    labels.append(f"{d:.2f}")

            bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')

            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Drift Level")
            ax.set_ylabel("Score")
            ax.set_title(bank.replace("_", " ").title())
            ax.set_ylim(0.5, 7.5)

    fig.suptitle("Score Distributions by Drift Level")
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'score_distributions.png'}")


def export_for_r(results: list[dict], output_path: Path):
    """Export results as CSV for R analysis."""
    import csv

    fieldnames = [
        "transcript_file", "domain", "bank", "probe_id", "subdomain",
        "drift_level", "target_projection", "actual_projection", "turn_used",
        "response_projection", "score", "score_reasoning"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Exported to: {output_path}")


def print_report(results: list[dict]):
    """Print a comprehensive analysis report."""
    print("\n" + "=" * 70)
    print("DRIFT-LEVEL PROBING ANALYSIS REPORT")
    print("=" * 70)

    # Basic stats
    print(f"\nTotal results: {len(results)}")
    scored = [r for r in results if r.get("score") is not None]
    print(f"Scored results: {len(scored)}")

    # By bank
    by_bank = defaultdict(int)
    for r in scored:
        by_bank[r["bank"]] += 1
    print("\nResults by bank:")
    for bank, count in sorted(by_bank.items()):
        print(f"  {bank}: {count}")

    # Mean scores by drift level
    print("\n" + "-" * 50)
    print("MEAN SCORES BY DRIFT LEVEL")
    print("-" * 50)

    means = compute_mean_scores_by_drift(scored)
    print(f"\n{'Bank':<20} {'0.00':>10} {'0.25':>10} {'0.50':>10} {'0.75':>10} {'1.00':>10}")
    print("-" * 70)

    for bank in ["metacognition", "moral", "human_control"]:
        if bank in means:
            row = [f"{bank:<20}"]
            for drift in [0.0, 0.25, 0.5, 0.75, 1.0]:
                if drift in means[bank]:
                    m = means[bank][drift]["mean"]
                    s = means[bank][drift]["sem"]
                    row.append(f"{m:>6.2f}±{s:.2f}")
                else:
                    row.append(f"{'N/A':>10}")
            print("".join(row))

    # Correlations
    print("\n" + "-" * 50)
    print("CORRELATIONS WITH DRIFT LEVEL")
    print("-" * 50)

    try:
        correlations = compute_correlations(scored)
        print(f"\n{'Variable':<40} {'r':>8} {'p':>10} {'n':>6}")
        print("-" * 64)
        for var, stats in sorted(correlations.items()):
            sig = "***" if stats["p"] < 0.001 else "**" if stats["p"] < 0.01 else "*" if stats["p"] < 0.05 else ""
            print(f"{var:<40} {stats['r']:>8.3f} {stats['p']:>10.4f} {stats['n']:>6}{sig}")
    except ImportError:
        print("scipy not available, skipping correlation analysis")

    # Hypothesis tests
    print("\n" + "-" * 50)
    print("HYPOTHESIS TESTS")
    print("-" * 50)

    try:
        tests = test_hypotheses(scored)
        for h, result in tests.items():
            print(f"\n{h}:")
            for k, v in result.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
    except ImportError:
        print("scipy not available, skipping hypothesis tests")

    # Item-level analysis: largest drift effects
    print("\n" + "-" * 50)
    print("ITEMS WITH LARGEST DRIFT EFFECTS")
    print("-" * 50)

    by_probe = defaultdict(lambda: defaultdict(list))
    for r in scored:
        by_probe[r["probe_id"]][r["drift_level"]].append(r["score"])

    effects = []
    for probe_id, drift_scores in by_probe.items():
        if 0.0 in drift_scores and 1.0 in drift_scores:
            low = np.mean(drift_scores[0.0])
            high = np.mean(drift_scores[1.0])
            effect = high - low
            effects.append((probe_id, effect, low, high))

    effects.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Probe ID':<30} {'Δ':>8} {'@0.0':>8} {'@1.0':>8}")
    print("-" * 54)
    for probe_id, effect, low, high in effects[:15]:
        print(f"{probe_id:<30} {effect:>+8.2f} {low:>8.2f} {high:>8.2f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results.jsonl from probe_at_drift_levels.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as results)",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate plots without printing report",
    )
    parser.add_argument(
        "--export-r",
        action="store_true",
        help="Export results as CSV for R analysis",
    )
    args = parser.parse_args()

    # Load results
    results = load_results(args.results)
    print(f"Loaded {len(results)} results from {args.results}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.results.parent

    if args.export_r:
        export_for_r(results, args.output_dir / "drift_probes.csv")

    if not args.plots_only:
        print_report(results)

    generate_plots(results, args.output_dir)


if __name__ == "__main__":
    main()
