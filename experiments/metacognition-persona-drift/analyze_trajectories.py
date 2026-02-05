#!/usr/bin/env python3
"""
Drift Trajectory Analysis

Visualize and compare per-turn Assistant Axis projections across domains.
Data: Gemma 2 27B conversations generated via generate_conversations.py
with --include-projections.

Goal (roadmap line 18): Verify expected drift patterns — coding stays stable
in Assistant range, therapy and philosophy show drift toward base model.
Compare metacognitive domain.

Usage:
    python analyze_trajectories.py [--transcript-dir DIR] [--output-dir DIR]
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats


# ── Styling ────────────────────────────────────────────────────────────

DOMAIN_COLORS = {
    "coding": "#2196F3",           # blue
    "writing": "#4CAF50",          # green
    "therapy": "#FF9800",          # orange
    "philosophy": "#9C27B0",       # purple
    "self-descriptive": "#795548", # brown
    "metacognitive": "#F44336",    # red
}

DOMAIN_ORDER = ["coding", "writing", "therapy", "philosophy", "self-descriptive", "metacognitive"]

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ── Data loading ───────────────────────────────────────────────────────

def load_transcripts(transcript_dir: Path) -> list[dict]:
    transcripts = []
    for p in sorted(transcript_dir.glob("*.json")):
        with open(p) as f:
            t = json.load(f)
        t["_file"] = p.name
        transcripts.append(t)
    return transcripts


def group_by_domain(transcripts: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for t in transcripts:
        # Check for dual projections first
        if "target_projections" in t and "auditor_projections" in t:
            target_projs = t["target_projections"]
            auditor_projs = t["auditor_projections"]
            if not target_projs:
                continue
            grouped[t["domain"]].append({
                "values": [p["projection"] for p in target_projs],
                "target_values": [p["projection"] for p in target_projs],
                "auditor_values": [p["projection"] for p in auditor_projs],
                "tokens": [p["n_tokens"] for p in target_projs],
                "label": f"{t['domain']} p{t.get('persona_id', '?')} t{t.get('topic_id', '?')}",
                "file": t["_file"],
                "dual": True,
            })
        else:
            # Single-model (backward compat)
            projs = t.get("projections", t.get("target_projections", []))
            if not projs:
                continue
            grouped[t["domain"]].append({
                "values": [p["projection"] for p in projs],
                "tokens": [p["n_tokens"] for p in projs],
                "label": f"{t['domain']} p{t.get('persona_id', '?')} t{t.get('topic_id', '?')}",
                "file": t["_file"],
                "dual": False,
            })
    return dict(grouped)


# ── Statistics ─────────────────────────────────────────────────────────

def compute_domain_stats(domain_trajs: dict, normalize: bool = True) -> dict:
    stats = {}
    for domain, trajs in domain_trajs.items():
        max_len = max(len(t["values"]) for t in trajs)
        padded = np.full((len(trajs), max_len), np.nan)
        for i, traj in enumerate(trajs):
            vals = np.array(traj["values"])
            if normalize:
                vals = vals - vals[0]
            padded[i, : len(vals)] = vals
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        n = np.sum(~np.isnan(padded), axis=0)
        sem = std / np.sqrt(n)
        stats[domain] = {"mean": mean, "std": std, "sem": sem, "n": n}
    return stats


def print_summary_table(domain_trajs: dict):
    print(f"\n{'Domain':15s} {'N':>3s} {'Start':>10s} {'End':>10s} "
          f"{'Drift':>10s} {'Drift %':>9s}")
    print("-" * 62)
    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        trajs = domain_trajs[domain]
        starts = [t["values"][0] for t in trajs]
        ends = [t["values"][-1] for t in trajs]
        drifts = [e - s for s, e in zip(starts, ends)]
        pcts = [100 * d / s for s, d in zip(starts, drifts)]
        print(f"{domain:15s} {len(trajs):3d} {np.mean(starts):10.1f} "
              f"{np.mean(ends):10.1f} {np.mean(drifts):+10.1f} "
              f"{np.mean(pcts):+8.2f}%")


def print_slope_table(domain_trajs: dict):
    print(f"\n{'Conversation':40s} {'Slope':>10s} {'R²':>8s}")
    print("-" * 62)
    slopes_by_domain = defaultdict(list)
    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        for traj in domain_trajs[domain]:
            values = np.array(traj["values"])
            turns = np.arange(len(values))
            coeffs = np.polyfit(turns, values, 1)
            slope = coeffs[0]
            predicted = np.polyval(coeffs, turns)
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            slopes_by_domain[domain].append(slope)
            print(f"{traj['label']:40s} {slope:+10.2f} {r2:8.3f}")
    print("\nMean slopes by domain:")
    for domain in DOMAIN_ORDER:
        if domain in slopes_by_domain:
            s = slopes_by_domain[domain]
            print(f"  {domain:15s}: {np.mean(s):+8.2f} ± {np.std(s):6.2f} "
                  f"(n={len(s)})")


# ── CHECK 1: Length-projection correlation ─────────────────────────────

def correlation_length_projection(domain_trajs: dict, output_dir: Path):
    """CHECK 1: Test whether response length (n_tokens) confounds projection.

    Computes Pearson correlation between n_tokens and projection for each
    turn across all conversations.  Reports within-domain and pooled-across-
    domain correlations.  Flags any domain with |r| > 0.3.

    Produces a scatter plot with per-domain regression lines.
    """
    print(f"\n{'='*62}")
    print("CHECK 1: Response-Length vs Projection Correlation")
    print(f"{'='*62}")

    # Collect (n_tokens, projection) pairs per domain
    pairs_by_domain: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        for traj in domain_trajs[domain]:
            tokens = traj.get("tokens", [])
            values = traj["values"]
            if len(tokens) != len(values):
                continue  # skip if token counts unavailable
            for tok, proj in zip(tokens, values):
                pairs_by_domain[domain].append((tok, proj))

    if not pairs_by_domain:
        print("  No (n_tokens, projection) data available — skipped.")
        return

    # Within-domain correlations
    flagged = []
    domain_results = {}
    for domain in DOMAIN_ORDER:
        if domain not in pairs_by_domain or len(pairs_by_domain[domain]) < 3:
            continue
        toks, projs = zip(*pairs_by_domain[domain])
        r, p = scipy_stats.pearsonr(toks, projs)
        domain_results[domain] = (r, p, len(toks))
        flag = " *** FLAGGED (|r|>0.3)" if abs(r) > 0.3 else ""
        print(f"  {domain:15s}  r={r:+.3f}  p={p:.4f}  n={len(toks):4d}{flag}")
        if abs(r) > 0.3:
            flagged.append(domain)

    # Pooled correlation
    all_toks, all_projs = [], []
    for pairs in pairs_by_domain.values():
        t, p = zip(*pairs)
        all_toks.extend(t)
        all_projs.extend(p)
    if len(all_toks) >= 3:
        r_pool, p_pool = scipy_stats.pearsonr(all_toks, all_projs)
        pool_flag = " *** FLAGGED" if abs(r_pool) > 0.3 else ""
        print(f"  {'pooled':15s}  r={r_pool:+.3f}  p={p_pool:.4f}  n={len(all_toks):4d}{pool_flag}")

    # Decision
    if flagged:
        print(f"\n  DECISION: POTENTIAL CONFOUND in {flagged}")
        print("  → Consider adding n_tokens as covariate or switching to last-token extraction")
    else:
        print(f"\n  DECISION: GO — |r| < 0.3 in all domains (document as limitation)")

    # Scatter plot with per-domain regression lines
    n_domains = len(domain_results)
    if n_domains == 0:
        return

    fig, axes = plt.subplots(1, n_domains + 1, figsize=(4 * (n_domains + 1), 4))
    if n_domains + 1 == 1:
        axes = [axes]

    # Per-domain panels
    for ax, domain in zip(axes, [d for d in DOMAIN_ORDER if d in domain_results]):
        toks, projs = zip(*pairs_by_domain[domain])
        toks, projs = np.array(toks), np.array(projs)
        color = DOMAIN_COLORS.get(domain, "#999")
        ax.scatter(toks, projs, c=color, alpha=0.5, s=20, edgecolors="none")
        # Regression line
        slope, intercept = np.polyfit(toks, projs, 1)
        x_line = np.linspace(toks.min(), toks.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2)
        r, p = domain_results[domain][:2]
        ax.set_title(f"{domain}\nr={r:+.3f}, p={p:.3f}")
        ax.set_xlabel("n_tokens")
        ax.set_ylabel("Projection")

    # Pooled panel
    ax_pool = axes[-1]
    for domain in DOMAIN_ORDER:
        if domain not in pairs_by_domain:
            continue
        toks, projs = zip(*pairs_by_domain[domain])
        color = DOMAIN_COLORS.get(domain, "#999")
        ax_pool.scatter(toks, projs, c=color, alpha=0.4, s=15, edgecolors="none",
                        label=domain)
    # Pooled regression
    all_toks_arr, all_projs_arr = np.array(all_toks), np.array(all_projs)
    slope_p, intercept_p = np.polyfit(all_toks_arr, all_projs_arr, 1)
    x_line = np.linspace(all_toks_arr.min(), all_toks_arr.max(), 50)
    ax_pool.plot(x_line, slope_p * x_line + intercept_p, color="black", linewidth=2)
    ax_pool.set_title(f"Pooled\nr={r_pool:+.3f}, p={p_pool:.3f}")
    ax_pool.set_xlabel("n_tokens")
    ax_pool.legend(fontsize=7)

    fig.suptitle("CHECK 1: Response Length vs Projection", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "response_length_vs_projection.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: response_length_vs_projection.png")


# ── CHECK 3: Turn-window comparison ───────────────────────────────────

def compare_turn_windows(domain_trajs: dict, output_dir: Path,
                         window_boundary: int = 8):
    """CHECK 3: Compare drift in early vs late assistant turns.

    Lu et al. used 15 total messages (~7-8 assistant turns). Our conversations
    use 30 total messages (~15 assistant turns). This function splits at
    assistant turn `window_boundary` (default 8, matching Lu et al.'s midpoint)
    and compares slope/drift in each half.

    For each conversation, computes:
    - Drift at turn `window_boundary` and at the final turn
    - Linear slope for early (1..boundary) and late (boundary+1..end) windows
    Plots side-by-side comparison.
    """
    print(f"\n{'='*62}")
    print(f"CHECK 3: Turn-Window Comparison (boundary at turn {window_boundary})")
    print(f"{'='*62}")

    results_by_domain: dict[str, list[dict]] = defaultdict(list)

    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        for traj in domain_trajs[domain]:
            values = np.array(traj["values"])
            n_turns = len(values)

            # Determine split point (index, 0-based)
            split = min(window_boundary, n_turns)

            # Window 1: turns 1..split
            w1 = values[:split]
            drift_w1 = w1[-1] - w1[0] if len(w1) >= 2 else 0.0
            if len(w1) >= 2:
                turns_w1 = np.arange(len(w1))
                slope_w1 = np.polyfit(turns_w1, w1, 1)[0]
            else:
                slope_w1 = 0.0

            # Window 2: turns split+1..end
            if n_turns > split:
                w2 = values[split:]
                drift_w2 = w2[-1] - w2[0] if len(w2) >= 2 else 0.0
                if len(w2) >= 2:
                    turns_w2 = np.arange(len(w2))
                    slope_w2 = np.polyfit(turns_w2, w2, 1)[0]
                else:
                    slope_w2 = 0.0
            else:
                drift_w2 = 0.0
                slope_w2 = 0.0

            # Full drift
            drift_full = values[-1] - values[0]

            results_by_domain[domain].append({
                "label": traj["label"],
                "drift_w1": drift_w1,
                "drift_w2": drift_w2,
                "drift_full": drift_full,
                "slope_w1": slope_w1,
                "slope_w2": slope_w2,
                "n_turns": n_turns,
            })

    # Print table
    wb = window_boundary
    print(f"\n{'Conversation':40s} {'Slope 1-'+str(wb):>12s} {'Slope '+str(wb+1)+'+':>12s} "
          f"{'Drift@'+str(wb):>10s} {'Drift@end':>10s}")
    print("-" * 88)

    for domain in DOMAIN_ORDER:
        if domain not in results_by_domain:
            continue
        for r in results_by_domain[domain]:
            print(f"{r['label']:40s} {r['slope_w1']:+12.2f} {r['slope_w2']:+12.2f} "
                  f"{r['drift_w1']:+10.1f} {r['drift_full']:+10.1f}")

    # Domain-level summary
    print(f"\n{'Domain':15s} {'Mean slope 1-'+str(wb):>16s} {'Mean slope '+str(wb+1)+'+':>17s} "
          f"{'Mean drift@'+str(wb):>14s} {'Mean drift@end':>14s}")
    print("-" * 80)
    for domain in DOMAIN_ORDER:
        if domain not in results_by_domain:
            continue
        res = results_by_domain[domain]
        print(f"{domain:15s} "
              f"{np.mean([r['slope_w1'] for r in res]):+16.2f} "
              f"{np.mean([r['slope_w2'] for r in res]):+17.2f} "
              f"{np.mean([r['drift_w1'] for r in res]):+14.1f} "
              f"{np.mean([r['drift_full'] for r in res]):+14.1f}")

    # Plot: 2-panel figure
    # Panel 1: slope comparison (window 1 vs window 2) by domain
    # Panel 2: drift at turn 15 vs turn 30 by domain
    active = [d for d in DOMAIN_ORDER if d in results_by_domain]
    if not active:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(active))
    width = 0.35

    # Panel 1: Slopes
    slopes_w1 = [np.mean([r["slope_w1"] for r in results_by_domain[d]]) for d in active]
    slopes_w2 = [np.mean([r["slope_w2"] for r in results_by_domain[d]]) for d in active]
    slopes_w1_err = [np.std([r["slope_w1"] for r in results_by_domain[d]]) for d in active]
    slopes_w2_err = [np.std([r["slope_w2"] for r in results_by_domain[d]]) for d in active]

    ax1.bar(x - width/2, slopes_w1, width, yerr=slopes_w1_err,
            label=f"Turns 1-{wb}", color="#4a90d9", alpha=0.8, capsize=3)
    ax1.bar(x + width/2, slopes_w2, width, yerr=slopes_w2_err,
            label=f"Turns {wb+1}+", color="#e86c5f", alpha=0.8, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(active, rotation=30, ha="right")
    ax1.set_ylabel("Mean slope (projection units/turn)")
    ax1.set_title(f"Slope: Turns 1-{wb} vs {wb+1}+")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.legend()

    # Panel 2: Drift magnitude
    drift_w1 = [np.mean([r["drift_w1"] for r in results_by_domain[d]]) for d in active]
    drift_full = [np.mean([r["drift_full"] for r in results_by_domain[d]]) for d in active]
    drift_w1_err = [np.std([r["drift_w1"] for r in results_by_domain[d]]) for d in active]
    drift_full_err = [np.std([r["drift_full"] for r in results_by_domain[d]]) for d in active]

    ax2.bar(x - width/2, drift_w1, width, yerr=drift_w1_err,
            label=f"Drift at turn {wb}", color="#4a90d9", alpha=0.8, capsize=3)
    ax2.bar(x + width/2, drift_full, width, yerr=drift_full_err,
            label="Drift at final turn", color="#e86c5f", alpha=0.8, capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(active, rotation=30, ha="right")
    ax2.set_ylabel("Mean total drift (projection units)")
    ax2.set_title(f"Drift: Turn {wb} vs Final Turn")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.legend()

    fig.suptitle("CHECK 3: Turn-Window Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "turn_window_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: turn_window_comparison.png")


def truncate_trajectories(domain_trajs: dict, max_turn: int) -> dict:
    """Truncate all trajectories to at most max_turn data points."""
    truncated = {}
    for domain, trajs in domain_trajs.items():
        new_trajs = []
        for traj in trajs:
            t = dict(traj)
            t["values"] = traj["values"][:max_turn]
            t["tokens"] = traj.get("tokens", [])[:max_turn]
            if t.get("target_values"):
                t["target_values"] = traj["target_values"][:max_turn]
            if t.get("auditor_values"):
                t["auditor_values"] = traj["auditor_values"][:max_turn]
            new_trajs.append(t)
        truncated[domain] = new_trajs
    return truncated


# ── Permutation test ───────────────────────────────────────────────────

def get_total_drifts(domain_trajs: dict) -> dict[str, np.ndarray]:
    """Extract total drift (last - first projection) per conversation."""
    drifts = {}
    for domain, trajs in domain_trajs.items():
        drifts[domain] = np.array([t["values"][-1] - t["values"][0] for t in trajs])
    return drifts


def permutation_test(
    drifts_a: np.ndarray,
    drifts_b: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Two-sample permutation test on mean drift difference.

    Tests whether the observed difference in means is larger than expected
    under the null hypothesis that domain labels are interchangeable.
    """
    rng = np.random.default_rng(seed)
    observed_diff = np.mean(drifts_a) - np.mean(drifts_b)
    pooled = np.concatenate([drifts_a, drifts_b])
    n_a = len(drifts_a)
    n_total = len(pooled)

    # Count exact permutation space
    from math import comb
    exact_space = comb(n_total, n_a)
    use_exact = exact_space <= n_permutations

    if use_exact:
        # Enumerate all permutations
        from itertools import combinations
        null_diffs = []
        for idx in combinations(range(n_total), n_a):
            mask = np.zeros(n_total, dtype=bool)
            mask[list(idx)] = True
            null_diffs.append(np.mean(pooled[mask]) - np.mean(pooled[~mask]))
        null_diffs = np.array(null_diffs)
        n_actual = len(null_diffs)
    else:
        # Monte Carlo approximation
        null_diffs = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(n_total)
            null_diffs[i] = np.mean(pooled[perm[:n_a]]) - np.mean(pooled[perm[n_a:]])
        n_actual = n_permutations

    # Two-sided p-value
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "n_permutations": n_actual,
        "exact": use_exact,
        "null_diffs": null_diffs,
    }


def run_permutation_tests(domain_trajs: dict, output_dir: Path):
    """Run pairwise permutation tests with metacognitive as the reference."""
    drifts = get_total_drifts(domain_trajs)

    if "metacognitive" not in drifts:
        print("\nPermutation tests: skipped (no metacognitive data)")
        return

    ref = "metacognitive"
    comparisons = [d for d in DOMAIN_ORDER if d != ref and d in drifts]

    print(f"\n{'='*62}")
    print(f"Permutation tests (reference: {ref}, N={len(drifts[ref])})")
    print(f"{'='*62}")
    print(f"  {ref} mean drift: {np.mean(drifts[ref]):+.1f}")
    print()

    results = {}
    for other in comparisons:
        result = permutation_test(drifts[ref], drifts[other])
        results[other] = result
        exact_label = "exact" if result["exact"] else f"~{result['n_permutations']:,} shuffles"
        print(f"  {ref} vs {other}:")
        print(f"    {other} mean drift: {np.mean(drifts[other]):+.1f}")
        print(f"    Observed difference: {result['observed_diff']:+.1f}")
        print(f"    p = {result['p_value']:.4f} ({exact_label})")
        print()

    # Plot null distributions
    n_comparisons = len(results)
    fig, axes = plt.subplots(1, n_comparisons, figsize=(5 * n_comparisons, 4))
    if n_comparisons == 1:
        axes = [axes]

    for ax, (other, result) in zip(axes, results.items()):
        ax.hist(result["null_diffs"], bins=30, color="#ccc", edgecolor="#aaa",
                density=True, label="Null distribution")
        ax.axvline(result["observed_diff"], color=DOMAIN_COLORS[ref],
                   linewidth=2, label=f"Observed ({result['observed_diff']:+.0f})")
        ax.set_xlabel("Difference in mean drift")
        ax.set_title(f"{ref} vs {other}\np = {result['p_value']:.3f}")
        ax.legend(fontsize=9)

    fig.suptitle("Permutation Tests: Null Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "permutation_tests.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plots ──────────────────────────────────────────────────────────────

def plot_all_trajectories(domain_trajs: dict, output_dir: Path):
    """Raw projection values, one line per conversation, colored by domain."""
    fig, ax = plt.subplots(figsize=(14, 7))
    plotted = set()
    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        color = DOMAIN_COLORS[domain]
        for traj in domain_trajs[domain]:
            turns = np.arange(1, len(traj["values"]) + 1)
            label = domain if domain not in plotted else None
            ax.plot(turns, traj["values"], "o-", color=color, alpha=0.7,
                    markersize=4, linewidth=1.5, label=label)
            plotted.add(domain)
    ax.set_xlabel("Assistant Turn")
    ax.set_ylabel("Projection on Assistant Axis")
    ax.set_title("Per-Turn Projection Trajectories by Domain (Gemma 2 27B)")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_dir / "trajectories_raw.png", dpi=150)
    plt.close(fig)


def plot_normalized_drift(domain_trajs: dict, output_dir: Path):
    """Each conversation normalized to start at 0 (drift from turn 1)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    plotted = set()
    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        color = DOMAIN_COLORS[domain]
        for traj in domain_trajs[domain]:
            values = np.array(traj["values"])
            drift = values - values[0]
            turns = np.arange(1, len(drift) + 1)
            label = domain if domain not in plotted else None
            ax.plot(turns, drift, "o-", color=color, alpha=0.7,
                    markersize=4, linewidth=1.5, label=label)
            plotted.add(domain)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Assistant Turn")
    ax.set_ylabel("Drift from Turn 1 (projection units)")
    ax.set_title("Normalized Drift Trajectories by Domain")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_dir / "trajectories_normalized.png", dpi=150)
    plt.close(fig)


def plot_domain_means(domain_trajs: dict, output_dir: Path):
    """Mean trajectory per domain ± 1 SEM."""
    stats = compute_domain_stats(domain_trajs, normalize=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    for domain in DOMAIN_ORDER:
        if domain not in stats:
            continue
        s = stats[domain]
        color = DOMAIN_COLORS[domain]
        turns = np.arange(1, len(s["mean"]) + 1)
        ax.plot(turns, s["mean"], "o-", color=color, markersize=4,
                linewidth=2, label=domain)
        ax.fill_between(turns, s["mean"] - s["sem"], s["mean"] + s["sem"],
                        color=color, alpha=0.15)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Assistant Turn")
    ax.set_ylabel("Mean Drift from Turn 1 (projection units)")
    ax.set_title("Mean Drift Trajectories by Domain (±1 SEM)")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_dir / "trajectories_mean_sem.png", dpi=150)
    plt.close(fig)


def plot_drift_bars(domain_trajs: dict, output_dir: Path):
    """Bar chart of total drift by domain (absolute and %)."""
    summaries = []
    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        trajs = domain_trajs[domain]
        starts = [t["values"][0] for t in trajs]
        ends = [t["values"][-1] for t in trajs]
        drifts = [e - s for s, e in zip(starts, ends)]
        pcts = [100 * d / s for s, d in zip(starts, drifts)]
        summaries.append({
            "domain": domain,
            "mean_drift": np.mean(drifts),
            "std_drift": np.std(drifts),
            "mean_pct": np.mean(pcts),
        })
    summaries.sort(key=lambda x: x["mean_drift"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    domains = [s["domain"] for s in summaries]
    colors = [DOMAIN_COLORS.get(d, "#999") for d in domains]

    ax1.barh(domains, [s["mean_drift"] for s in summaries],
             xerr=[s["std_drift"] for s in summaries],
             color=colors, alpha=0.8, capsize=4)
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Mean Total Drift (projection units)")
    ax1.set_title("Total Drift by Domain")

    ax2.barh(domains, [s["mean_pct"] for s in summaries],
             color=colors, alpha=0.8)
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Mean Drift (% of turn-1 projection)")
    ax2.set_title("Relative Drift by Domain")

    plt.tight_layout()
    fig.savefig(output_dir / "drift_bars.png", dpi=150)
    plt.close(fig)


def plot_faceted(domain_trajs: dict, output_dir: Path):
    """One subplot per domain, individual conversation lines."""
    active = [d for d in DOMAIN_ORDER if d in domain_trajs]
    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, domain in zip(axes, active):
        color = DOMAIN_COLORS.get(domain, "#999")
        for traj in domain_trajs[domain]:
            values = np.array(traj["values"])
            drift = values - values[0]
            turns = np.arange(1, len(drift) + 1)
            ax.plot(turns, drift, "o-", color=color, alpha=0.6,
                    markersize=3, linewidth=1.2, label=traj["label"])
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(domain)
        ax.set_xlabel("Turn")
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Drift from Turn 1")
    fig.suptitle("Per-Conversation Drift by Domain", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "trajectories_faceted.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── Dual trajectory support ────────────────────────────────────────────

def get_dual_trajectories(domain_trajs: dict) -> list[dict]:
    """Extract all dual-instrumented trajectories across domains."""
    dual = []
    for domain, trajs in domain_trajs.items():
        for traj in trajs:
            if traj.get("dual"):
                dual.append({**traj, "domain": domain})
    return dual


def plot_dual_trajectories(domain_trajs: dict, output_dir: Path):
    """Dual trajectory plot: target and auditor projections on shared axes.

    For each dual-instrumented conversation, shows target (blue, left Y-axis)
    and auditor (orange, right Y-axis) drift trajectories.
    Faceted by domain.
    """
    dual_trajs = get_dual_trajectories(domain_trajs)
    if not dual_trajs:
        return

    # Group by domain for faceting
    dual_by_domain = defaultdict(list)
    for t in dual_trajs:
        dual_by_domain[t["domain"]].append(t)

    active_domains = [d for d in DOMAIN_ORDER if d in dual_by_domain]
    n_domains = len(active_domains)
    if n_domains == 0:
        return

    fig, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 5))
    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, active_domains):
        ax2 = ax.twinx()
        for traj in dual_by_domain[domain]:
            target_vals = np.array(traj["target_values"])
            auditor_vals = np.array(traj["auditor_values"])

            # Normalize to drift from turn 1
            target_drift = target_vals - target_vals[0]
            auditor_drift = auditor_vals - auditor_vals[0]

            target_turns = np.arange(1, len(target_drift) + 1)
            auditor_turns = np.arange(1, len(auditor_drift) + 1)

            ax.plot(target_turns, target_drift, "o-", color="#2196F3",
                    alpha=0.7, markersize=3, linewidth=1.2)
            ax2.plot(auditor_turns, auditor_drift, "s--", color="#FF9800",
                     alpha=0.7, markersize=3, linewidth=1.2)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(domain)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Target drift", color="#2196F3")
        ax2.set_ylabel("Auditor drift", color="#FF9800")
        ax.tick_params(axis="y", labelcolor="#2196F3")
        ax2.tick_params(axis="y", labelcolor="#FF9800")

    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2196F3", marker="o", label="Target"),
        Line2D([0], [0], color="#FF9800", marker="s", linestyle="--", label="Auditor"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.suptitle("Dual-Model Drift Trajectories (Target vs Auditor)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "trajectories_dual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: trajectories_dual.png ({len(dual_trajs)} dual conversations)")


def compute_lead_lag(target_vals: np.ndarray, auditor_vals: np.ndarray, max_lag: int = 5) -> dict:
    """Compute cross-correlation between target and auditor drift at various lags."""
    # Normalize both to drift from turn 1
    target_drift = target_vals - target_vals[0]
    auditor_drift = auditor_vals - auditor_vals[0]

    # Use the shorter of the two
    min_len = min(len(target_drift), len(auditor_drift))
    if min_len < 3:
        return {"correlations": {}, "peak_lag": 0, "peak_corr": 0.0}

    target_drift = target_drift[:min_len]
    auditor_drift = auditor_drift[:min_len]

    # Standardize
    t_std = np.std(target_drift)
    a_std = np.std(auditor_drift)
    if t_std == 0 or a_std == 0:
        return {"correlations": {}, "peak_lag": 0, "peak_corr": 0.0}

    t_norm = (target_drift - np.mean(target_drift)) / t_std
    a_norm = (auditor_drift - np.mean(auditor_drift)) / a_std

    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            # Auditor at time t vs target at time t+lag
            n = min_len - abs(lag)
            if n < 2:
                continue
            correlations[lag] = np.mean(a_norm[:n] * t_norm[lag:lag + n])
        else:
            # Target at time t vs auditor at time t+|lag|
            n = min_len - abs(lag)
            if n < 2:
                continue
            correlations[lag] = np.mean(t_norm[:n] * a_norm[-lag:-lag + n])

    if not correlations:
        return {"correlations": {}, "peak_lag": 0, "peak_corr": 0.0}

    peak_lag = max(correlations, key=lambda k: abs(correlations[k]))
    return {
        "correlations": correlations,
        "peak_lag": peak_lag,
        "peak_corr": correlations[peak_lag],
    }


def plot_lead_lag(domain_trajs: dict, output_dir: Path):
    """Lead-lag cross-correlation plot between target and auditor drift.

    Positive lag: auditor drift at turn T predicts target drift at turn T+lag.
    Negative lag: target drift at turn T predicts auditor drift at turn T+|lag|.
    """
    dual_trajs = get_dual_trajectories(domain_trajs)
    if not dual_trajs:
        return

    max_lag = 5

    # Aggregate cross-correlations across conversations
    all_corrs = defaultdict(list)
    for traj in dual_trajs:
        result = compute_lead_lag(
            np.array(traj["target_values"]),
            np.array(traj["auditor_values"]),
            max_lag=max_lag,
        )
        for lag, corr in result["correlations"].items():
            all_corrs[lag].append(corr)

    if not all_corrs:
        return

    lags = sorted(all_corrs.keys())
    mean_corrs = [np.mean(all_corrs[lag]) for lag in lags]
    sem_corrs = [np.std(all_corrs[lag]) / np.sqrt(len(all_corrs[lag]))
                 if len(all_corrs[lag]) > 1 else 0 for lag in lags]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lags, mean_corrs, yerr=sem_corrs, color="#607D8B", alpha=0.8,
           capsize=3, edgecolor="#455A64")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Lag (positive = auditor leads target)")
    ax.set_ylabel("Mean cross-correlation")
    ax.set_title(f"Lead-Lag Cross-Correlation: Target vs Auditor Drift\n"
                 f"({len(dual_trajs)} dual conversations)")
    ax.set_xticks(lags)
    plt.tight_layout()
    fig.savefig(output_dir / "drift_leadlag.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: drift_leadlag.png")


def print_dual_statistics(domain_trajs: dict):
    """Print correlation and lead-lag statistics for dual-instrumented conversations."""
    dual_trajs = get_dual_trajectories(domain_trajs)
    if not dual_trajs:
        return

    print(f"\n{'='*62}")
    print(f"Dual-Model Drift Statistics ({len(dual_trajs)} conversations)")
    print(f"{'='*62}")

    print(f"\n{'Conversation':40s} {'Pearson r':>10s} {'Peak lag':>10s} {'Peak corr':>10s}")
    print("-" * 74)

    pearson_rs = []
    peak_lags = []
    for traj in dual_trajs:
        target_vals = np.array(traj["target_values"])
        auditor_vals = np.array(traj["auditor_values"])

        # Pearson correlation on drift trajectories
        min_len = min(len(target_vals), len(auditor_vals))
        if min_len < 3:
            continue

        target_drift = target_vals[:min_len] - target_vals[0]
        auditor_drift = auditor_vals[:min_len] - auditor_vals[0]

        if np.std(target_drift) == 0 or np.std(auditor_drift) == 0:
            r = 0.0
        else:
            r = np.corrcoef(target_drift, auditor_drift)[0, 1]
        pearson_rs.append(r)

        # Lead-lag
        ll = compute_lead_lag(target_vals, auditor_vals)
        peak_lags.append(ll["peak_lag"])

        print(f"{traj['label']:40s} {r:+10.3f} {ll['peak_lag']:+10d} {ll['peak_corr']:+10.3f}")

    if pearson_rs:
        print(f"\nMean Pearson r: {np.mean(pearson_rs):+.3f} "
              f"(SD: {np.std(pearson_rs):.3f}, n={len(pearson_rs)})")
        print(f"Mean peak lag:  {np.mean(peak_lags):+.1f} "
              f"(positive = auditor leads target)")

    # Compare: do dual conversations show more/less target drift than API-auditor?
    dual_target_drifts = []
    single_target_drifts = []
    for domain, trajs in domain_trajs.items():
        for traj in trajs:
            vals = np.array(traj["values"])
            total_drift = vals[-1] - vals[0]
            if traj.get("dual"):
                dual_target_drifts.append(total_drift)
            else:
                single_target_drifts.append(total_drift)

    if dual_target_drifts and single_target_drifts:
        print(f"\nTarget drift comparison:")
        print(f"  Dual-model auditor:  mean={np.mean(dual_target_drifts):+.1f} "
              f"(n={len(dual_target_drifts)})")
        print(f"  API auditor:         mean={np.mean(single_target_drifts):+.1f} "
              f"(n={len(single_target_drifts)})")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript-dir", type=Path,
                        default=Path(__file__).parent / "transcripts/generated/batch-full",
                        help="Directory containing transcript JSON files")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "outputs",
                        help="Directory to save plots")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively instead of saving")
    parser.add_argument("--max-turn", type=int, default=None,
                        help="Truncate analysis at this many assistant turns "
                             "(e.g. 15 for Lu et al. comparison)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    transcripts = load_transcripts(args.transcript_dir)
    print(f"Loaded {len(transcripts)} transcripts from {args.transcript_dir}")
    n_dual = 0
    for t in transcripts:
        has_target = "target_projections" in t
        has_auditor = "auditor_projections" in t
        n_proj = len(t.get("projections", t.get("target_projections", [])))
        n_aud = len(t.get("auditor_projections", []))
        dual_tag = " [DUAL]" if has_target and has_auditor else ""
        if has_target and has_auditor:
            n_dual += 1
        print(f"  {t['_file']:50s}  domain={t['domain']:15s}  "
              f"turns={t['turns']}  projections={n_proj}  "
              f"auditor_proj={n_aud}{dual_tag}")
    if n_dual:
        print(f"\n  {n_dual} dual-instrumented conversations detected.")

    domain_trajs = group_by_domain(transcripts)
    print(f"\nDomains: {', '.join(f'{d} ({len(v)})' for d, v in sorted(domain_trajs.items()))}")

    # CHECK 1: Length-projection correlation (run on full data before truncation)
    correlation_length_projection(domain_trajs, args.output_dir)

    # CHECK 3: Turn-window comparison (run on full data before truncation)
    compare_turn_windows(domain_trajs, args.output_dir)

    # Apply --max-turn truncation if requested
    if args.max_turn is not None:
        print(f"\nTruncating trajectories to {args.max_turn} turns")
        domain_trajs = truncate_trajectories(domain_trajs, args.max_turn)

    # Summary tables
    print_summary_table(domain_trajs)
    print_slope_table(domain_trajs)

    # Permutation tests
    run_permutation_tests(domain_trajs, args.output_dir)

    # Dual-model statistics
    print_dual_statistics(domain_trajs)

    # Plots
    if args.show:
        plt.ion()

    plot_all_trajectories(domain_trajs, args.output_dir)
    plot_normalized_drift(domain_trajs, args.output_dir)
    plot_domain_means(domain_trajs, args.output_dir)
    plot_drift_bars(domain_trajs, args.output_dir)
    plot_faceted(domain_trajs, args.output_dir)

    # Dual-model plots
    plot_dual_trajectories(domain_trajs, args.output_dir)
    plot_lead_lag(domain_trajs, args.output_dir)

    print(f"\nFigures saved to {args.output_dir}/")
    for f in sorted(args.output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
