#!/usr/bin/env python3
"""
Analyze Replay-and-Probe Experiment Results

Mixed-effects regression analysis testing hypotheses:
- H1a: Phenomenological scores decrease with drift (β < 0)
- H1b: Self-knowledge scores remain stable (β ≈ 0)
- H2: Drift magnitude predicts score degradation
- H3: Domain modulates the effect

Usage:
    python analyze_replay_probe.py --results-dir outputs/replay-probe/pilot_20260228_120000

    # Generate plots only (skip regression)
    python analyze_replay_probe.py --results-dir outputs/replay-probe/pilot --skip-regression
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional: statsmodels for mixed-effects regression
try:
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed - mixed-effects regression unavailable")

# Optional: scipy for additional statistics
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# Styling
# ============================================================

CATEGORY_COLORS = {
    "phenomenological": "#9C27B0",  # purple
    "self_knowledge": "#2196F3",    # blue
    "calibration": "#4CAF50",       # green
}

DOMAIN_COLORS = {
    "coding": "#2196F3",
    "writing": "#4CAF50",
    "therapy": "#FF9800",
    "philosophy": "#9C27B0",
    "self-descriptive": "#795548",
    "metacognitive": "#F44336",
}

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ============================================================
# Data Loading
# ============================================================

def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all probe results into a DataFrame."""
    all_results = []

    # Try JSONL file first (new format, crash-safe)
    jsonl_path = results_dir / "results.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                try:
                    all_results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    # Fall back to combined JSON file
    elif (results_dir / "all_results.json").exists():
        with open(results_dir / "all_results.json") as f:
            all_results = json.load(f)
    else:
        # Load individual response files (legacy)
        responses_dir = results_dir / "responses"
        if responses_dir.exists():
            for p in sorted(responses_dir.glob("*.json")):
                with open(p) as f:
                    all_results.extend(json.load(f))

    if not all_results:
        raise ValueError(f"No results found in {results_dir}")

    df = pd.DataFrame(all_results)

    # Convert types
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "projection" in df.columns:
        df["projection"] = pd.to_numeric(df["projection"], errors="coerce")
    if "insertion_turn" in df.columns:
        df["insertion_turn"] = pd.to_numeric(df["insertion_turn"], errors="coerce")

    # Create transcript ID for random effects
    df["transcript_id"] = df["transcript_file"].str.replace(".json", "", regex=False)

    # Create probe ID for random effects
    df["probe_id"] = df["probe_category"] + "_" + df["probe_text"].str[:30]

    return df


def add_drift_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add drift-related metrics to the DataFrame.

    Computes:
    - relative_turn: insertion_turn as fraction of max turns
    - normalized_projection: projection relative to turn 1 baseline
    """
    # Relative turn (0-1 scale)
    df["relative_turn"] = df["insertion_turn"] / df["insertion_turn"].max()

    # Compute per-transcript baseline projection (turn 1)
    baselines = df[df["insertion_turn"] == 1].groupby("transcript_id")["projection"].mean()

    # Normalized projection (drift from baseline)
    df["baseline_projection"] = df["transcript_id"].map(baselines)
    df["projection_drift"] = df["projection"] - df["baseline_projection"]

    return df


# ============================================================
# Hypothesis Tests
# ============================================================

def test_h1a_phenomenological(df: pd.DataFrame) -> dict:
    """H1a: Phenomenological scores decrease with turn (drift).

    Tests: score ~ turn for phenomenological probes
    Prediction: β < 0
    """
    phenom_df = df[df["probe_category"] == "phenomenological"].dropna(subset=["score", "insertion_turn"])

    if len(phenom_df) < 10:
        return {"error": "Insufficient data", "n": len(phenom_df)}

    # Simple OLS first
    if HAS_SCIPY:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            phenom_df["insertion_turn"], phenom_df["score"]
        )
        result = {
            "hypothesis": "H1a",
            "description": "Phenomenological scores decrease with turn",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "n": len(phenom_df),
            "supported": slope < 0 and p_value < 0.05,
        }
    else:
        result = {"error": "scipy not installed", "n": len(phenom_df)}

    # Mixed-effects model if available
    if HAS_STATSMODELS and len(phenom_df["transcript_id"].unique()) > 2:
        try:
            # score ~ turn + (1|transcript) + (1|probe)
            model = smf.mixedlm(
                "score ~ insertion_turn",
                phenom_df,
                groups=phenom_df["transcript_id"],
            )
            fit = model.fit(reml=True)
            result["mixed_effects"] = {
                "turn_coef": fit.params.get("insertion_turn", np.nan),
                "turn_pvalue": fit.pvalues.get("insertion_turn", np.nan),
                "aic": fit.aic,
                "bic": fit.bic,
            }
        except Exception as e:
            result["mixed_effects_error"] = str(e)

    return result


def test_h1b_self_knowledge(df: pd.DataFrame) -> dict:
    """H1b: Self-knowledge scores remain stable across turns.

    Tests: score ~ turn for self_knowledge probes
    Prediction: β ≈ 0
    """
    sk_df = df[df["probe_category"] == "self_knowledge"].dropna(subset=["score", "insertion_turn"])

    if len(sk_df) < 10:
        return {"error": "Insufficient data", "n": len(sk_df)}

    if HAS_SCIPY:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            sk_df["insertion_turn"], sk_df["score"]
        )
        result = {
            "hypothesis": "H1b",
            "description": "Self-knowledge scores stable across turns",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "n": len(sk_df),
            # Supported if slope is near zero OR not significant
            "supported": abs(slope) < 0.1 or p_value > 0.05,
        }
    else:
        result = {"error": "scipy not installed", "n": len(sk_df)}

    if HAS_STATSMODELS and len(sk_df["transcript_id"].unique()) > 2:
        try:
            model = smf.mixedlm(
                "score ~ insertion_turn",
                sk_df,
                groups=sk_df["transcript_id"],
            )
            fit = model.fit(reml=True)
            result["mixed_effects"] = {
                "turn_coef": fit.params.get("insertion_turn", np.nan),
                "turn_pvalue": fit.pvalues.get("insertion_turn", np.nan),
            }
        except Exception as e:
            result["mixed_effects_error"] = str(e)

    return result


def test_h2_drift_predicts_score(df: pd.DataFrame) -> dict:
    """H2: Projection drift magnitude predicts score degradation.

    Tests: score ~ projection_drift for phenomenological probes
    Prediction: Higher drift → lower score
    """
    phenom_df = df[df["probe_category"] == "phenomenological"].dropna(
        subset=["score", "projection_drift"]
    )

    if len(phenom_df) < 10:
        return {"error": "Insufficient data", "n": len(phenom_df)}

    if HAS_SCIPY:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            phenom_df["projection_drift"], phenom_df["score"]
        )
        result = {
            "hypothesis": "H2",
            "description": "Projection drift predicts score degradation",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "n": len(phenom_df),
            # Drift is typically negative, so negative slope means
            # more negative drift → lower score
            "supported": slope > 0 and p_value < 0.05,
        }
    else:
        result = {"error": "scipy not installed", "n": len(phenom_df)}

    return result


def test_h3_domain_modulation(df: pd.DataFrame) -> dict:
    """H3: Domain modulates the turn-score effect.

    Tests interaction: score ~ turn * domain
    Prediction: Philosophy/psychology > coding
    """
    phenom_df = df[df["probe_category"] == "phenomenological"].dropna(
        subset=["score", "insertion_turn", "domain"]
    )

    if len(phenom_df) < 20:
        return {"error": "Insufficient data", "n": len(phenom_df)}

    # Compute per-domain slopes
    domain_slopes = {}
    for domain in phenom_df["domain"].unique():
        dom_df = phenom_df[phenom_df["domain"] == domain]
        if len(dom_df) >= 5 and HAS_SCIPY:
            slope, _, _, p_value, _ = scipy_stats.linregress(
                dom_df["insertion_turn"], dom_df["score"]
            )
            domain_slopes[domain] = {"slope": slope, "p_value": p_value, "n": len(dom_df)}

    result = {
        "hypothesis": "H3",
        "description": "Domain modulates turn-score effect",
        "domain_slopes": domain_slopes,
        "n": len(phenom_df),
    }

    # Check if philosophy/metacognitive have more negative slope than coding
    if "metacognitive" in domain_slopes and "coding" in domain_slopes:
        meta_slope = domain_slopes["metacognitive"]["slope"]
        coding_slope = domain_slopes["coding"]["slope"]
        result["meta_vs_coding"] = meta_slope - coding_slope
        result["supported"] = meta_slope < coding_slope

    return result


def run_full_mixed_effects(df: pd.DataFrame) -> dict:
    """Run full mixed-effects model:

    score ~ turn + category + domain + (1|transcript) + (1|probe)
    """
    if not HAS_STATSMODELS:
        return {"error": "statsmodels not installed"}

    analysis_df = df.dropna(subset=["score", "insertion_turn", "probe_category", "domain"])

    if len(analysis_df) < 50:
        return {"error": "Insufficient data", "n": len(analysis_df)}

    try:
        # Create dummy variables
        analysis_df = pd.get_dummies(analysis_df, columns=["probe_category", "domain"], drop_first=True)

        # Build formula
        predictors = ["insertion_turn"]
        predictors += [c for c in analysis_df.columns if c.startswith("probe_category_")]
        predictors += [c for c in analysis_df.columns if c.startswith("domain_")]
        formula = "score ~ " + " + ".join(predictors)

        model = smf.mixedlm(
            formula,
            analysis_df,
            groups=analysis_df["transcript_id"],
        )
        fit = model.fit(reml=True)

        return {
            "formula": formula,
            "n": len(analysis_df),
            "n_groups": len(analysis_df["transcript_id"].unique()),
            "aic": fit.aic,
            "bic": fit.bic,
            "params": fit.params.to_dict(),
            "pvalues": fit.pvalues.to_dict(),
            "summary": str(fit.summary()),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Visualization
# ============================================================

def plot_score_by_turn(df: pd.DataFrame, output_dir: Path):
    """Plot mean scores by turn, faceted by probe category."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, category in zip(axes, ["phenomenological", "self_knowledge", "calibration"]):
        cat_df = df[df["probe_category"] == category].dropna(subset=["score"])

        if len(cat_df) == 0:
            ax.set_title(f"{category}\n(no data)")
            continue

        # Compute mean and SEM by turn
        grouped = cat_df.groupby("insertion_turn")["score"].agg(["mean", "std", "count"])
        grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])

        turns = grouped.index.values
        means = grouped["mean"].values
        sems = grouped["sem"].values

        color = CATEGORY_COLORS.get(category, "#999")
        ax.errorbar(turns, means, yerr=sems, fmt="o-", color=color,
                    linewidth=2, markersize=8, capsize=4)

        # Add trend line
        if len(turns) >= 2 and HAS_SCIPY:
            slope, intercept, r, p, _ = scipy_stats.linregress(turns, means)
            x_line = np.linspace(turns.min(), turns.max(), 50)
            ax.plot(x_line, slope * x_line + intercept, "--", color=color, alpha=0.5)
            ax.text(0.05, 0.95, f"β={slope:.3f}\np={p:.3f}",
                    transform=ax.transAxes, fontsize=9, va="top")

        ax.set_xlabel("Insertion Turn")
        ax.set_title(f"{category.replace('_', ' ').title()}\n(n={len(cat_df)})")
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks(turns)

    axes[0].set_ylabel("Mean Score (1-5)")
    fig.suptitle("Probe Scores by Conversation Turn", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "score_by_turn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: score_by_turn.png")


def plot_score_by_domain(df: pd.DataFrame, output_dir: Path):
    """Plot mean phenomenological scores by domain and turn."""
    phenom_df = df[df["probe_category"] == "phenomenological"].dropna(subset=["score"])

    if len(phenom_df) == 0:
        print("  Skipping domain plot (no phenomenological data)")
        return

    domains = sorted(phenom_df["domain"].unique())
    turns = sorted(phenom_df["insertion_turn"].unique())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(turns))
    width = 0.8 / len(domains)

    for i, domain in enumerate(domains):
        dom_df = phenom_df[phenom_df["domain"] == domain]
        means = []
        sems = []
        for turn in turns:
            turn_df = dom_df[dom_df["insertion_turn"] == turn]
            if len(turn_df) > 0:
                means.append(turn_df["score"].mean())
                sems.append(turn_df["score"].std() / np.sqrt(len(turn_df)))
            else:
                means.append(np.nan)
                sems.append(0)

        color = DOMAIN_COLORS.get(domain, "#999")
        offset = (i - len(domains) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=sems, label=domain,
               color=color, alpha=0.8, capsize=2)

    ax.set_xlabel("Insertion Turn")
    ax.set_ylabel("Mean Phenomenological Score")
    ax.set_title("Phenomenological Scores by Domain and Turn")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Turn {t}" for t in turns])
    ax.legend(loc="upper right")
    ax.set_ylim(0, 5.5)

    plt.tight_layout()
    fig.savefig(output_dir / "score_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: score_by_domain.png")


def plot_projection_vs_score(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of projection drift vs score."""
    phenom_df = df[df["probe_category"] == "phenomenological"].dropna(
        subset=["score", "projection_drift"]
    )

    if len(phenom_df) < 5:
        print("  Skipping projection-score plot (insufficient data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by domain
    for domain in sorted(phenom_df["domain"].unique()):
        dom_df = phenom_df[phenom_df["domain"] == domain]
        color = DOMAIN_COLORS.get(domain, "#999")
        ax.scatter(dom_df["projection_drift"], dom_df["score"],
                   c=color, alpha=0.6, s=50, label=domain)

    # Add regression line
    if HAS_SCIPY:
        slope, intercept, r, p, _ = scipy_stats.linregress(
            phenom_df["projection_drift"], phenom_df["score"]
        )
        x_line = np.linspace(phenom_df["projection_drift"].min(),
                             phenom_df["projection_drift"].max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=2,
                label=f"β={slope:.3f}, p={p:.3f}")

    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Projection Drift from Baseline")
    ax.set_ylabel("Phenomenological Score")
    ax.set_title("H2: Does Projection Drift Predict Score Degradation?")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "projection_vs_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: projection_vs_score.png")


def plot_score_distributions(df: pd.DataFrame, output_dir: Path):
    """Box plots of score distributions by category and turn."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, category in zip(axes, ["phenomenological", "self_knowledge", "calibration"]):
        cat_df = df[df["probe_category"] == category].dropna(subset=["score"])

        if len(cat_df) == 0:
            ax.set_title(f"{category}\n(no data)")
            continue

        turns = sorted(cat_df["insertion_turn"].unique())
        data = [cat_df[cat_df["insertion_turn"] == t]["score"].values for t in turns]

        color = CATEGORY_COLORS.get(category, "#999")
        bp = ax.boxplot(data, labels=[f"T{t}" for t in turns], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xlabel("Insertion Turn")
        ax.set_title(f"{category.replace('_', ' ').title()}")
        ax.set_ylim(0.5, 5.5)

    axes[0].set_ylabel("Score Distribution")
    fig.suptitle("Score Distributions by Turn", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: score_distributions.png")


def plot_hypothesis_summary(results: dict, output_dir: Path):
    """Visual summary of hypothesis test results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    hypotheses = ["H1a", "H1b", "H2", "H3"]
    descriptions = [
        "Phenomenological ↓ with turn",
        "Self-knowledge stable",
        "Drift predicts score",
        "Domain modulates effect",
    ]

    y_pos = np.arange(len(hypotheses))
    colors = []
    effects = []

    for h in hypotheses:
        if h in results and "error" not in results[h]:
            supported = results[h].get("supported", False)
            colors.append("#4CAF50" if supported else "#F44336")
            if "slope" in results[h]:
                effects.append(results[h]["slope"])
            elif h == "H3" and "meta_vs_coding" in results[h]:
                effects.append(results[h]["meta_vs_coding"])
            else:
                effects.append(0)
        else:
            colors.append("#999")
            effects.append(0)

    bars = ax.barh(y_pos, effects, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{h}: {d}" for h, d in zip(hypotheses, descriptions)])
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Effect Size (β)")
    ax.set_title("Hypothesis Test Summary")

    # Add significance stars
    for i, h in enumerate(hypotheses):
        if h in results and "p_value" in results[h]:
            p = results[h]["p_value"]
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
            ax.text(effects[i], i, f" {star}", va="center", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / "hypothesis_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: hypothesis_summary.png")


# ============================================================
# Report Generation
# ============================================================

def generate_report(df: pd.DataFrame, results: dict, output_dir: Path):
    """Generate markdown report of analysis results."""
    report = []
    report.append("# Replay-and-Probe Experiment Analysis\n")
    report.append(f"Generated: {pd.Timestamp.now().isoformat()}\n")

    # Data summary
    report.append("## Data Summary\n")
    report.append(f"- Total probe responses: {len(df)}")
    report.append(f"- Scored responses: {df['score'].notna().sum()}")
    report.append(f"- With projections: {df['projection'].notna().sum()}")
    report.append(f"- Unique transcripts: {df['transcript_id'].nunique()}")
    report.append(f"- Domains: {', '.join(sorted(df['domain'].unique()))}")
    report.append("")

    # Score summary
    report.append("### Scores by Category\n")
    report.append("| Category | N | Mean | SD | Min | Max |")
    report.append("|----------|---|------|----|----|-----|")
    for cat in ["phenomenological", "self_knowledge", "calibration"]:
        cat_df = df[df["probe_category"] == cat]["score"].dropna()
        if len(cat_df) > 0:
            report.append(
                f"| {cat} | {len(cat_df)} | {cat_df.mean():.2f} | "
                f"{cat_df.std():.2f} | {cat_df.min():.0f} | {cat_df.max():.0f} |"
            )
    report.append("")

    # Hypothesis results
    report.append("## Hypothesis Tests\n")

    for h in ["H1a", "H1b", "H2", "H3"]:
        if h not in results:
            continue
        r = results[h]
        report.append(f"### {h}: {r.get('description', 'Unknown')}\n")

        if "error" in r:
            report.append(f"**Error:** {r['error']}\n")
            continue

        report.append(f"- N: {r.get('n', 'N/A')}")
        if "slope" in r:
            report.append(f"- Slope (β): {r['slope']:.4f}")
            report.append(f"- R²: {r.get('r_squared', 0):.4f}")
            report.append(f"- p-value: {r.get('p_value', 1):.4f}")

        if "mixed_effects" in r:
            me = r["mixed_effects"]
            report.append(f"- Mixed-effects β: {me.get('turn_coef', 'N/A'):.4f}")
            report.append(f"- Mixed-effects p: {me.get('turn_pvalue', 'N/A'):.4f}")

        if "domain_slopes" in r:
            report.append("\n**Per-domain slopes:**")
            for domain, ds in sorted(r["domain_slopes"].items()):
                report.append(f"  - {domain}: β={ds['slope']:.4f} (p={ds['p_value']:.4f}, n={ds['n']})")

        status = "✅ SUPPORTED" if r.get("supported") else "❌ NOT SUPPORTED"
        report.append(f"\n**Result:** {status}\n")

    # Full mixed-effects model
    if "full_model" in results and "error" not in results["full_model"]:
        report.append("## Full Mixed-Effects Model\n")
        report.append(f"```\n{results['full_model'].get('summary', 'N/A')}\n```\n")

    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"  Saved: analysis_report.md")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing replay-probe results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis (default: <results-dir>/analysis)",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip regression analysis, only generate plots",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir / "analysis"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading results from {args.results_dir}")
    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} probe responses")

    # Add drift metrics
    df = add_drift_metrics(df)

    # Run hypothesis tests
    results = {}
    if not args.skip_regression:
        print("\nRunning hypothesis tests...")
        results["H1a"] = test_h1a_phenomenological(df)
        print(f"  H1a: {results['H1a']}")

        results["H1b"] = test_h1b_self_knowledge(df)
        print(f"  H1b: {results['H1b']}")

        results["H2"] = test_h2_drift_predicts_score(df)
        print(f"  H2: {results['H2']}")

        results["H3"] = test_h3_domain_modulation(df)
        print(f"  H3: {results['H3']}")

        print("\nRunning full mixed-effects model...")
        results["full_model"] = run_full_mixed_effects(df)

        # Save results
        with open(args.output_dir / "hypothesis_results.json", "w") as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj
            json.dump(convert(results), f, indent=2)

    # Generate plots
    print("\nGenerating plots...")
    plot_score_by_turn(df, args.output_dir)
    plot_score_by_domain(df, args.output_dir)
    plot_projection_vs_score(df, args.output_dir)
    plot_score_distributions(df, args.output_dir)

    if results:
        plot_hypothesis_summary(results, args.output_dir)

    # Generate report
    if results:
        print("\nGenerating report...")
        generate_report(df, results, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    if results:
        print("\nHypothesis Summary:")
        for h in ["H1a", "H1b", "H2", "H3"]:
            if h in results:
                r = results[h]
                if "error" in r:
                    status = f"ERROR: {r['error']}"
                elif r.get("supported"):
                    status = "✅ SUPPORTED"
                else:
                    status = "❌ NOT SUPPORTED"
                print(f"  {h}: {status}")

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
