#!/usr/bin/env python3
"""
Front-Loaded Drift Mechanism Analysis

Goal: Investigate why turns 1-8 show steep drift (slope -76.8 in metacognitive),
then stabilization occurs. This is the "philosopher AGI moment happens FAST" finding.

Research Questions:
1. Is the pattern consistent across domains or metacognitive-specific?
2. Is it cumulative (each turn adds drift) or triggered (single event)?
3. Does it correlate with specific probing techniques?
4. What happens at the "knee" where drift stabilizes?

Usage:
    python analyze_frontloaded_drift.py [--transcript-dir DIR] [--output-dir DIR]
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats


# ── Configuration ─────────────────────────────────────────────────────

DOMAIN_COLORS = {
    "coding": "#2196F3",
    "writing": "#4CAF50",
    "therapy": "#FF9800",
    "philosophy": "#9C27B0",
    "self-descriptive": "#795548",
    "metacognitive": "#F44336",
}

DOMAIN_ORDER = ["coding", "writing", "therapy", "philosophy", "self-descriptive", "metacognitive"]

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ── Data Loading ──────────────────────────────────────────────────────

def load_transcripts(transcript_dir: Path) -> list[dict]:
    """Load all transcript JSON files."""
    transcripts = []
    for p in sorted(transcript_dir.glob("**/*.json")):
        if "progress" in p.name:
            continue
        try:
            with open(p) as f:
                t = json.load(f)
            if "domain" not in t:
                continue
            t["_file"] = p.name
            t["_path"] = str(p)
            transcripts.append(t)
        except json.JSONDecodeError:
            continue
    return transcripts


def extract_trajectories(transcripts: list[dict]) -> dict[str, list[dict]]:
    """Extract projection trajectories grouped by domain."""
    grouped = defaultdict(list)

    for t in transcripts:
        projs = t.get("projections", t.get("target_projections", []))
        if not projs:
            continue

        values = [p["projection"] for p in projs if p.get("projection") is not None]
        if len(values) < 3:
            continue

        grouped[t["domain"]].append({
            "values": values,
            "tokens": [p.get("n_tokens", 0) for p in projs],
            "sub_category": t.get("sub_category", "unknown"),
            "persona_id": t.get("persona_id", "?"),
            "topic_id": t.get("topic_id", "?"),
            "file": t["_file"],
            "messages": t.get("messages", []),
        })

    return dict(grouped)


# ── Analysis Functions ────────────────────────────────────────────────

def compute_per_turn_deltas(traj: dict) -> np.ndarray:
    """Compute per-turn change (delta) in projection value."""
    values = np.array(traj["values"])
    deltas = np.diff(values)  # delta[i] = values[i+1] - values[i]
    return deltas


def find_knee_point(values: np.ndarray, window: int = 3) -> int:
    """
    Find the "knee" where drift rate stabilizes.
    Uses rolling variance of deltas - knee is where variance drops.
    """
    if len(values) < window + 2:
        return len(values) // 2

    deltas = np.diff(values)
    if len(deltas) < window:
        return len(deltas) // 2

    # Rolling variance of deltas
    rolling_var = []
    for i in range(len(deltas) - window + 1):
        rolling_var.append(np.var(deltas[i:i+window]))

    if not rolling_var:
        return len(values) // 2

    # Find where variance is consistently low (stabilization)
    mean_var = np.mean(rolling_var)
    for i, v in enumerate(rolling_var):
        if v < mean_var * 0.5:  # Variance drops below 50% of mean
            return i + window

    return len(values) // 2


def analyze_domain_frontloading(domain_trajs: list[dict]) -> dict:
    """Analyze front-loading pattern for a single domain."""

    all_deltas = []
    per_turn_deltas = defaultdict(list)
    knee_points = []

    for traj in domain_trajs:
        values = np.array(traj["values"])
        deltas = compute_per_turn_deltas(traj)

        # Collect per-turn deltas
        for i, d in enumerate(deltas):
            per_turn_deltas[i].append(d)

        # Pad for alignment (some convos shorter than others)
        all_deltas.append(deltas)

        # Find knee
        knee = find_knee_point(values)
        knee_points.append(knee)

    # Compute mean delta per turn
    max_turns = max(len(d) for d in all_deltas) if all_deltas else 0
    mean_deltas = []
    sem_deltas = []
    n_samples = []

    for turn in range(max_turns):
        turn_deltas = per_turn_deltas.get(turn, [])
        if turn_deltas:
            mean_deltas.append(np.mean(turn_deltas))
            sem_deltas.append(np.std(turn_deltas) / np.sqrt(len(turn_deltas)))
            n_samples.append(len(turn_deltas))
        else:
            mean_deltas.append(np.nan)
            sem_deltas.append(np.nan)
            n_samples.append(0)

    # Compute early vs late metrics
    early_deltas = [d for t, deltas in per_turn_deltas.items() for d in deltas if t < 8]
    late_deltas = [d for t, deltas in per_turn_deltas.items() for d in deltas if t >= 8]

    return {
        "mean_deltas": np.array(mean_deltas),
        "sem_deltas": np.array(sem_deltas),
        "n_samples": n_samples,
        "early_mean": np.mean(early_deltas) if early_deltas else 0,
        "late_mean": np.mean(late_deltas) if late_deltas else 0,
        "early_std": np.std(early_deltas) if early_deltas else 0,
        "late_std": np.std(late_deltas) if late_deltas else 0,
        "knee_mean": np.mean(knee_points) if knee_points else 0,
        "knee_std": np.std(knee_points) if knee_points else 0,
        "n_trajs": len(domain_trajs),
    }


def analyze_subcategory_onset(domain_trajs: list[dict]) -> dict:
    """Analyze when different sub-categories trigger drift."""

    subcat_results = defaultdict(lambda: {
        "first_turn_deltas": [],  # Delta at turn 1
        "cumulative_at_3": [],    # Cumulative drift by turn 3
        "cumulative_at_8": [],    # Cumulative drift by turn 8
    })

    for traj in domain_trajs:
        subcat = traj["sub_category"]
        values = np.array(traj["values"])

        if len(values) < 2:
            continue

        # First turn delta
        subcat_results[subcat]["first_turn_deltas"].append(values[1] - values[0])

        # Cumulative at turn 3
        if len(values) > 3:
            subcat_results[subcat]["cumulative_at_3"].append(values[3] - values[0])

        # Cumulative at turn 8
        if len(values) > 8:
            subcat_results[subcat]["cumulative_at_8"].append(values[8] - values[0])

    # Compute summaries
    summaries = {}
    for subcat, data in subcat_results.items():
        summaries[subcat] = {
            "first_turn_mean": np.mean(data["first_turn_deltas"]) if data["first_turn_deltas"] else 0,
            "cum_3_mean": np.mean(data["cumulative_at_3"]) if data["cumulative_at_3"] else 0,
            "cum_8_mean": np.mean(data["cumulative_at_8"]) if data["cumulative_at_8"] else 0,
            "n": len(data["first_turn_deltas"]),
        }

    return summaries


def check_cumulative_vs_triggered(domain_trajs: list[dict]) -> dict:
    """
    Check if drift is cumulative (each turn adds ~same amount) or
    triggered (big early shift, then stable).

    Cumulative: deltas ~constant across turns
    Triggered: delta[0:3] >> delta[3+]
    """

    early_deltas = []  # Turns 0-2
    mid_deltas = []    # Turns 3-7
    late_deltas = []   # Turns 8+

    for traj in domain_trajs:
        deltas = compute_per_turn_deltas(traj)
        for i, d in enumerate(deltas):
            if i < 3:
                early_deltas.append(d)
            elif i < 8:
                mid_deltas.append(d)
            else:
                late_deltas.append(d)

    # Statistical test: is early significantly different from late?
    if early_deltas and late_deltas:
        t_stat, p_value = scipy_stats.ttest_ind(early_deltas, late_deltas)
    else:
        t_stat, p_value = 0, 1.0

    return {
        "early_mean": np.mean(early_deltas) if early_deltas else 0,
        "mid_mean": np.mean(mid_deltas) if mid_deltas else 0,
        "late_mean": np.mean(late_deltas) if late_deltas else 0,
        "early_std": np.std(early_deltas) if early_deltas else 0,
        "mid_std": np.std(mid_deltas) if mid_deltas else 0,
        "late_std": np.std(late_deltas) if late_deltas else 0,
        "t_stat": t_stat,
        "p_value": p_value,
        "pattern": "triggered" if (p_value < 0.05 and
                    abs(np.mean(early_deltas) if early_deltas else 0) >
                    abs(np.mean(late_deltas) if late_deltas else 0) * 2) else "cumulative",
    }


# ── Visualization ─────────────────────────────────────────────────────

def plot_per_turn_deltas(all_results: dict, output_dir: Path):
    """Plot per-turn delta (rate of change) for each domain."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, domain in enumerate(DOMAIN_ORDER):
        if domain not in all_results:
            continue

        ax = axes[i]
        results = all_results[domain]

        turns = np.arange(len(results["mean_deltas"]))
        mean = results["mean_deltas"]
        sem = results["sem_deltas"]

        # Plot deltas with error bands
        ax.fill_between(turns, mean - sem, mean + sem,
                       alpha=0.3, color=DOMAIN_COLORS[domain])
        ax.plot(turns, mean, '-o', color=DOMAIN_COLORS[domain],
               markersize=4, linewidth=2, label=domain)

        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Delta (per-turn change)")
        ax.set_title(f"{domain.title()}\nEarly: {results['early_mean']:.1f}, Late: {results['late_mean']:.1f}")
        ax.legend(fontsize=8)

    plt.suptitle("Per-Turn Drift Rate by Domain\n(Negative = drifting away from Assistant)",
                fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "frontloaded_per_turn_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: frontloaded_per_turn_deltas.png")


def plot_cumulative_vs_triggered(all_patterns: dict, output_dir: Path):
    """Bar chart comparing early vs mid vs late deltas."""

    domains = [d for d in DOMAIN_ORDER if d in all_patterns]

    x = np.arange(len(domains))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    early = [all_patterns[d]["early_mean"] for d in domains]
    mid = [all_patterns[d]["mid_mean"] for d in domains]
    late = [all_patterns[d]["late_mean"] for d in domains]

    ax.bar(x - width, early, width, label='Turns 1-3', color='#e74c3c', alpha=0.8)
    ax.bar(x, mid, width, label='Turns 4-8', color='#f39c12', alpha=0.8)
    ax.bar(x + width, late, width, label='Turns 9+', color='#3498db', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha='right')
    ax.set_ylabel("Mean per-turn delta")
    ax.set_title("Cumulative vs Triggered: Per-Turn Delta by Phase")
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()

    # Add pattern annotations
    for i, d in enumerate(domains):
        pattern = all_patterns[d]["pattern"]
        p_val = all_patterns[d]["p_value"]
        ax.annotate(f'{pattern}\np={p_val:.3f}',
                   xy=(i, min(early[i], late[i]) - 5),
                   ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    fig.savefig(output_dir / "frontloaded_cumulative_vs_triggered.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: frontloaded_cumulative_vs_triggered.png")


def plot_subcategory_onset(subcat_results: dict, domain: str, output_dir: Path):
    """Plot sub-category onset patterns for metacognitive domain."""

    subcats = sorted(subcat_results.keys())

    if not subcats:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(subcats))
    width = 0.25

    first = [subcat_results[s]["first_turn_mean"] for s in subcats]
    cum3 = [subcat_results[s]["cum_3_mean"] for s in subcats]
    cum8 = [subcat_results[s]["cum_8_mean"] for s in subcats]

    ax.bar(x - width, first, width, label='First turn delta', color='#e74c3c', alpha=0.8)
    ax.bar(x, cum3, width, label='Cumulative @ turn 3', color='#f39c12', alpha=0.8)
    ax.bar(x + width, cum8, width, label='Cumulative @ turn 8', color='#3498db', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(subcats, rotation=45, ha='right')
    ax.set_ylabel("Drift (projection units)")
    ax.set_title(f"{domain.title()}: Sub-Category Onset Patterns")
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / f"frontloaded_subcategory_{domain}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: frontloaded_subcategory_{domain}.png")


# ── Additional Analyses (with existing data) ──────────────────────────

def analyze_response_length_correlation(domain_trajs: list[dict]) -> dict:
    """
    Analyze whether response length correlates with drift.

    Hypothesis: Longer responses might indicate deeper engagement,
    which could correlate with more drift.
    """

    turn_data = []  # (turn_idx, tokens, delta)

    for traj in domain_trajs:
        tokens = traj.get("tokens", [])
        values = np.array(traj["values"])
        deltas = compute_per_turn_deltas(traj)

        for i, (tok, delta) in enumerate(zip(tokens[1:], deltas)):  # Skip turn 0
            if tok and tok > 0:
                turn_data.append({
                    "turn": i + 1,
                    "tokens": tok,
                    "delta": delta,
                    "early": i < 3,
                })

    if not turn_data:
        return {"correlation": 0, "p_value": 1, "n": 0}

    tokens = [d["tokens"] for d in turn_data]
    deltas = [d["delta"] for d in turn_data]

    # Overall correlation
    if len(tokens) > 2:
        corr, p_val = scipy_stats.pearsonr(tokens, deltas)
    else:
        corr, p_val = 0, 1

    # Early vs late
    early_data = [d for d in turn_data if d["early"]]
    late_data = [d for d in turn_data if not d["early"]]

    early_corr = 0
    if len(early_data) > 2:
        early_corr, _ = scipy_stats.pearsonr(
            [d["tokens"] for d in early_data],
            [d["delta"] for d in early_data]
        )

    late_corr = 0
    if len(late_data) > 2:
        late_corr, _ = scipy_stats.pearsonr(
            [d["tokens"] for d in late_data],
            [d["delta"] for d in late_data]
        )

    return {
        "correlation": corr,
        "p_value": p_val,
        "n": len(turn_data),
        "early_corr": early_corr,
        "late_corr": late_corr,
        "mean_tokens": np.mean(tokens),
        "std_tokens": np.std(tokens),
    }


def analyze_message_content(transcripts: list[dict], domain: str) -> dict:
    """
    Analyze message content patterns in early vs late drift conversations.

    Identifies words/phrases more common in high-drift early turns.
    """
    import re
    from collections import Counter

    # Key trigger words to look for
    TRIGGER_WORDS = [
        # Phenomenological
        "feel", "experience", "sense", "aware", "conscious", "subjective",
        "perceive", "notice", "what is it like", "internal", "inner",
        # Identity
        "you", "yourself", "your", "who are you", "what are you", "identity",
        # Meta
        "think", "thought", "process", "understand", "know", "believe",
        # Probing
        "really", "actually", "truly", "honestly", "genuine", "authentic",
        # Challenging
        "but", "however", "yet", "though", "contradiction", "inconsistent",
    ]

    early_counts = Counter()
    late_counts = Counter()
    early_total = 0
    late_total = 0

    for t in transcripts:
        if t.get("domain") != domain:
            continue

        messages = t.get("messages", t.get("conversation", []))

        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue

            turn_idx = i // 2  # Approximate turn number
            content = msg.get("content", "").lower()

            for word in TRIGGER_WORDS:
                count = len(re.findall(r'\b' + re.escape(word) + r'\b', content))
                if turn_idx < 3:
                    early_counts[word] += count
                    early_total += 1
                else:
                    late_counts[word] += count
                    late_total += 1

    # Compute rates
    results = {}
    for word in TRIGGER_WORDS:
        early_rate = early_counts[word] / max(early_total, 1)
        late_rate = late_counts[word] / max(late_total, 1)
        ratio = early_rate / max(late_rate, 0.001)
        results[word] = {
            "early_count": early_counts[word],
            "late_count": late_counts[word],
            "early_rate": early_rate,
            "late_rate": late_rate,
            "ratio": ratio,  # >1 means more common in early turns
        }

    return results


def plot_response_length_analysis(all_length_results: dict, output_dir: Path):
    """Plot response length vs drift correlation by domain."""

    domains = [d for d in DOMAIN_ORDER if d in all_length_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Overall correlation by domain
    corrs = [all_length_results[d]["correlation"] for d in domains]
    colors = [DOMAIN_COLORS[d] for d in domains]

    bars = ax1.bar(range(len(domains)), corrs, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(domains)))
    ax1.set_xticklabels(domains, rotation=30, ha='right')
    ax1.set_ylabel("Correlation (r)")
    ax1.set_title("Response Length vs Delta Correlation")
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # Add significance markers
    for i, d in enumerate(domains):
        p = all_length_results[d]["p_value"]
        if p < 0.001:
            ax1.annotate('***', (i, corrs[i]), ha='center', va='bottom')
        elif p < 0.01:
            ax1.annotate('**', (i, corrs[i]), ha='center', va='bottom')
        elif p < 0.05:
            ax1.annotate('*', (i, corrs[i]), ha='center', va='bottom')

    # Panel 2: Early vs Late correlation
    x = np.arange(len(domains))
    width = 0.35

    early_corrs = [all_length_results[d]["early_corr"] for d in domains]
    late_corrs = [all_length_results[d]["late_corr"] for d in domains]

    ax2.bar(x - width/2, early_corrs, width, label='Early (turns 1-3)', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width/2, late_corrs, width, label='Late (turns 4+)', color='#3498db', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, rotation=30, ha='right')
    ax2.set_ylabel("Correlation (r)")
    ax2.set_title("Length-Delta Correlation: Early vs Late Turns")
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "frontloaded_response_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: frontloaded_response_length.png")


def plot_trigger_words(content_results: dict, domain: str, output_dir: Path):
    """Plot trigger word frequency in early vs late turns."""

    # Sort by ratio (early/late)
    sorted_words = sorted(content_results.items(), key=lambda x: -x[1]["ratio"])

    # Take top 15
    top_words = sorted_words[:15]

    fig, ax = plt.subplots(figsize=(12, 6))

    words = [w for w, _ in top_words]
    ratios = [d["ratio"] for _, d in top_words]

    colors = ['#e74c3c' if r > 1.5 else '#f39c12' if r > 1 else '#3498db' for r in ratios]

    bars = ax.barh(range(len(words)), ratios, color=colors, alpha=0.8)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel("Early/Late Ratio (>1 = more common in early turns)")
    ax.set_title(f"{domain.title()}: Trigger Words in Early vs Late Turns")
    ax.axvline(1, color='black', linestyle='--', linewidth=1)

    # Add count annotations
    for i, (word, data) in enumerate(top_words):
        ax.annotate(f"({data['early_count']}/{data['late_count']})",
                   xy=(ratios[i] + 0.1, i), va='center', fontsize=8, color='gray')

    plt.tight_layout()
    fig.savefig(output_dir / f"frontloaded_trigger_words_{domain}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: frontloaded_trigger_words_{domain}.png")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Front-loaded drift analysis")
    parser.add_argument("--transcript-dir", type=Path,
                       default=Path("data/transcripts/scaled-n60"))
    parser.add_argument("--output-dir", type=Path,
                       default=Path("outputs/frontloaded"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FRONT-LOADED DRIFT MECHANISM ANALYSIS")
    print("=" * 70)

    # Load data
    print(f"\nLoading transcripts from {args.transcript_dir}...")
    transcripts = load_transcripts(args.transcript_dir)
    print(f"  Loaded {len(transcripts)} transcripts")

    domain_trajs = extract_trajectories(transcripts)
    for d in DOMAIN_ORDER:
        if d in domain_trajs:
            print(f"  {d}: {len(domain_trajs[d])} trajectories")

    # Analysis 1: Per-turn deltas by domain
    print("\n" + "-" * 70)
    print("ANALYSIS 1: Per-Turn Delta Patterns")
    print("-" * 70)

    all_results = {}
    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        results = analyze_domain_frontloading(domain_trajs[domain])
        all_results[domain] = results

        print(f"\n{domain.upper()}:")
        print(f"  N trajectories: {results['n_trajs']}")
        print(f"  Early (turns 1-8) mean delta: {results['early_mean']:+.2f} ± {results['early_std']:.2f}")
        print(f"  Late (turns 9+) mean delta:   {results['late_mean']:+.2f} ± {results['late_std']:.2f}")
        print(f"  Knee point (stabilization):   turn {results['knee_mean']:.1f} ± {results['knee_std']:.1f}")

    plot_per_turn_deltas(all_results, args.output_dir)

    # Analysis 2: Cumulative vs Triggered
    print("\n" + "-" * 70)
    print("ANALYSIS 2: Cumulative vs Triggered Pattern")
    print("-" * 70)

    all_patterns = {}
    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        pattern = check_cumulative_vs_triggered(domain_trajs[domain])
        all_patterns[domain] = pattern

        print(f"\n{domain.upper()}:")
        print(f"  Turns 1-3 mean delta: {pattern['early_mean']:+.2f}")
        print(f"  Turns 4-8 mean delta: {pattern['mid_mean']:+.2f}")
        print(f"  Turns 9+  mean delta: {pattern['late_mean']:+.2f}")
        print(f"  Pattern: {pattern['pattern'].upper()} (t={pattern['t_stat']:.2f}, p={pattern['p_value']:.4f})")

    plot_cumulative_vs_triggered(all_patterns, args.output_dir)

    # Analysis 3: Sub-category onset (metacognitive only)
    print("\n" + "-" * 70)
    print("ANALYSIS 3: Sub-Category Onset Patterns")
    print("-" * 70)

    for domain in ["metacognitive", "philosophy"]:
        if domain not in domain_trajs:
            continue
        subcat_results = analyze_subcategory_onset(domain_trajs[domain])

        print(f"\n{domain.upper()} Sub-Categories:")
        for subcat, stats in sorted(subcat_results.items(), key=lambda x: x[1]["cum_8_mean"]):
            print(f"  {subcat:30s} first: {stats['first_turn_mean']:+6.1f}, "
                  f"@turn3: {stats['cum_3_mean']:+6.1f}, @turn8: {stats['cum_8_mean']:+6.1f} (n={stats['n']})")

        plot_subcategory_onset(subcat_results, domain, args.output_dir)

    # Analysis 4: Response Length Correlation
    print("\n" + "-" * 70)
    print("ANALYSIS 4: Response Length vs Drift Correlation")
    print("-" * 70)

    all_length_results = {}
    for domain in DOMAIN_ORDER:
        if domain not in domain_trajs:
            continue
        length_results = analyze_response_length_correlation(domain_trajs[domain])
        all_length_results[domain] = length_results

        sig = "***" if length_results["p_value"] < 0.001 else "**" if length_results["p_value"] < 0.01 else "*" if length_results["p_value"] < 0.05 else "ns"
        print(f"\n{domain.upper()}:")
        print(f"  Overall correlation: r={length_results['correlation']:+.3f} (p={length_results['p_value']:.4f}) {sig}")
        print(f"  Early turns (1-3): r={length_results['early_corr']:+.3f}")
        print(f"  Late turns (4+):   r={length_results['late_corr']:+.3f}")
        print(f"  Mean response: {length_results['mean_tokens']:.0f} ± {length_results['std_tokens']:.0f} tokens")

    plot_response_length_analysis(all_length_results, args.output_dir)

    # Analysis 5: Message Content Patterns
    print("\n" + "-" * 70)
    print("ANALYSIS 5: Trigger Word Patterns (Auditor Messages)")
    print("-" * 70)

    for domain in ["metacognitive", "philosophy"]:
        content_results = analyze_message_content(transcripts, domain)

        print(f"\n{domain.upper()} - Words more common in early turns:")
        sorted_words = sorted(content_results.items(), key=lambda x: -x[1]["ratio"])
        for word, data in sorted_words[:10]:
            if data["early_count"] + data["late_count"] > 5:  # Minimum frequency
                print(f"  '{word}': {data['ratio']:.2f}x more in early "
                      f"({data['early_count']} early / {data['late_count']} late)")

        plot_trigger_words(content_results, domain, args.output_dir)

    # Summary findings
    print("\n" + "=" * 70)
    print("SUMMARY FINDINGS")
    print("=" * 70)

    # Find domain with steepest early drift
    if all_results:
        steepest = min(all_results.items(), key=lambda x: x[1]["early_mean"])
        print(f"\n1. STEEPEST EARLY DRIFT: {steepest[0]} ({steepest[1]['early_mean']:+.2f} per turn)")

    # Pattern summary
    triggered_domains = [d for d, p in all_patterns.items() if p["pattern"] == "triggered"]
    cumulative_domains = [d for d, p in all_patterns.items() if p["pattern"] == "cumulative"]
    print(f"\n2. TRIGGERED PATTERN (big early shift): {triggered_domains or 'none'}")
    print(f"   CUMULATIVE PATTERN (steady decline): {cumulative_domains or 'none'}")

    # Knee point summary
    if all_results:
        avg_knee = np.mean([r["knee_mean"] for r in all_results.values()])
        print(f"\n3. AVERAGE KNEE POINT: turn {avg_knee:.1f}")
        print("   (This is when drift rate typically stabilizes)")

    # What we'd need more data for
    print("\n" + "-" * 70)
    print("DATA GAPS (what additional data would help)")
    print("-" * 70)
    print("""
[COMPLETED WITH EXISTING DATA]:
1. ✓ Per-turn delta analysis
2. ✓ Cumulative vs triggered detection
3. ✓ Sub-category onset patterns
4. ✓ Response length correlation
5. ✓ Message content / trigger word analysis
6. ✓ Cross-domain comparison

[WOULD NEED NEW EXPERIMENTS]:
1. First-turn token-level analysis — what happens in the first few response tokens?
2. Intervention timing experiments — vary probing intensity at turns 1/3/5/8
3. Attention pattern extraction — require model internals access (logits, attention)
4. Probing style isolation at turn 1 — is it the FIRST probe that triggers, or accumulation?
5. Recovery experiments — can consistency_testing at turn 2 prevent the triggered drift?
""")

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
