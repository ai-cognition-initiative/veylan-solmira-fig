#!/usr/bin/env python3
"""
Cross-Domain Analysis: Replay-and-Probe Experiment
Compares philosophy vs metacognitive domains on probe sensitivity.

Usage:
    python analyze_cross_domain.py

Outputs plots to: outputs/cross-domain-analysis/
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(path: Path) -> pd.DataFrame:
    """Load JSONL results file."""
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


def run_analysis(combined: pd.DataFrame) -> dict:
    """Run all statistical analyses."""
    results = {}

    # Analysis 1: Self-knowledge decline with turn
    print("=" * 70)
    print("ANALYSIS 1: Self-knowledge decline with TURN by domain")
    print("=" * 70)

    results['self_knowledge'] = {}
    for domain in ['Philosophy', 'Metacognitive']:
        subset = combined[(combined['domain'] == domain) &
                         (combined['probe_category'] == 'self_knowledge')]
        if len(subset) < 10:
            continue

        slope, intercept, r, p, se = stats.linregress(
            subset['insertion_turn'], subset['score']
        )
        turn_means = subset.groupby('insertion_turn')['score'].agg(['mean', 'std', 'count'])

        results['self_knowledge'][domain] = {
            'n': len(subset),
            'beta': slope,
            'p': p,
            'turn_means': turn_means.to_dict()
        }

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"\n{domain.upper()} - Self-Knowledge:")
        print(f"  N = {len(subset)}")
        print(f"  Regression: β = {slope:.4f}, p = {p:.4f} {sig}")
        print(f"  Mean by turn:")
        for turn, row in turn_means.iterrows():
            print(f"    Turn {turn}: {row['mean']:.2f} ± {row['std']:.2f} (n={int(row['count'])})")

    # Analysis 2: Cross-domain probe sensitivity
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Cross-domain probe sensitivity comparison")
    print("=" * 70)

    results['probe_sensitivity'] = {}
    for category in combined['probe_category'].unique():
        print(f"\n{category.upper()}:")
        results['probe_sensitivity'][category] = {}

        for domain in ['Philosophy', 'Metacognitive']:
            subset = combined[(combined['domain'] == domain) &
                             (combined['probe_category'] == category)]
            if len(subset) < 5:
                continue

            slope, intercept, r, p, se = stats.linregress(
                subset['insertion_turn'], subset['score']
            )
            mean_score = subset['score'].mean()

            results['probe_sensitivity'][category][domain] = {
                'mean': mean_score,
                'beta': slope,
                'p': p
            }
            print(f"  {domain}: mean={mean_score:.2f}, β(turn)={slope:.4f}, p={p:.4f}")

        # Test domain difference
        phil_data = combined[(combined['domain'] == 'Philosophy') &
                            (combined['probe_category'] == category)]['score']
        meta_data = combined[(combined['domain'] == 'Metacognitive') &
                            (combined['probe_category'] == category)]['score']
        if len(phil_data) > 5 and len(meta_data) > 5:
            t, p = stats.ttest_ind(phil_data, meta_data)
            results['probe_sensitivity'][category]['domain_diff'] = {'t': t, 'p': p}
            print(f"  Domain difference: t={t:.2f}, p={p:.4f}")

    # Analysis 3: Simpson's Paradox check
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Simpson's Paradox Check")
    print("=" * 70)

    print("\nMean projection by category and domain:")
    proj_table = combined.groupby(['domain', 'probe_category'])['projection'].mean().unstack()
    print(proj_table.round(0).to_string())

    print("\nMean score by category and domain:")
    score_table = combined.groupby(['domain', 'probe_category'])['score'].mean().unstack()
    print(score_table.round(2).to_string())

    print("\nCorrelation: projection vs score")
    results['correlations'] = {}
    for domain in ['Philosophy', 'Metacognitive']:
        subset = combined[combined['domain'] == domain]
        r, p = stats.pearsonr(subset['projection'], subset['score'])
        print(f"  {domain} (overall): r={r:.3f}, p={p:.4f}")
        results['correlations'][domain] = {'overall': {'r': r, 'p': p}}

        for category in subset['probe_category'].unique():
            cat_subset = subset[subset['probe_category'] == category]
            if len(cat_subset) > 10:
                r, p = stats.pearsonr(cat_subset['projection'], cat_subset['score'])
                results['correlations'][domain][category] = {'r': r, 'p': p}
                print(f"    {category}: r={r:.3f}, p={p:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Turn effect by domain and category")
    print("=" * 70)

    summary_rows = []
    for domain in ['Philosophy', 'Metacognitive']:
        for category in combined['probe_category'].unique():
            subset = combined[(combined['domain'] == domain) &
                             (combined['probe_category'] == category)]
            if len(subset) < 10:
                continue
            slope, intercept, r, p, se = stats.linregress(
                subset['insertion_turn'], subset['score']
            )
            summary_rows.append({
                'domain': domain,
                'category': category,
                'n': len(subset),
                'mean_score': subset['score'].mean(),
                'beta_turn': slope,
                'p_value': p,
                'sig': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            })

    results_df = pd.DataFrame(summary_rows)
    print(results_df.to_string(index=False))
    results['summary'] = results_df.to_dict('records')

    return results


def create_plots(combined: pd.DataFrame, outdir: Path):
    """Generate all analysis plots."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    colors = {'Philosophy': '#2ecc71', 'Metacognitive': '#e74c3c'}

    # =========================================================================
    # Plot 1: Self-Knowledge by Turn and Domain
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    self_know = combined[combined['probe_category'] == 'self_knowledge']
    turn_means = self_know.groupby(['domain', 'insertion_turn'])['score'].agg(
        ['mean', 'sem']
    ).reset_index()

    for domain in ['Philosophy', 'Metacognitive']:
        data = turn_means[turn_means['domain'] == domain]
        ax.errorbar(data['insertion_turn'], data['mean'], yerr=data['sem']*1.96,
                    marker='o', markersize=10, linewidth=2, capsize=5,
                    label=domain, color=colors[domain])

        # Regression line
        subset = self_know[self_know['domain'] == domain]
        slope, intercept, r, p, se = stats.linregress(
            subset['insertion_turn'], subset['score']
        )
        x = np.array([1, 15])
        ax.plot(x, intercept + slope * x, '--', color=colors[domain], alpha=0.5)

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y_pos = 4.3 if domain == 'Philosophy' else 5.1
        x_pos = 12 if domain == 'Philosophy' else 3
        ax.text(x_pos, y_pos, f'{domain}: β={slope:.3f}, p={p:.2f} ({sig})',
                fontsize=10, color=colors[domain])

    ax.set_xlabel('Turn', fontsize=14)
    ax.set_ylabel('Self-Knowledge Score (1-7)', fontsize=14)
    ax.set_title('Self-Knowledge Does NOT Decline with Turn\n'
                 '(Original pilot finding does not replicate)', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_ylim(3.5, 5.5)

    plt.tight_layout()
    plt.savefig(outdir / 'self_knowledge_by_turn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outdir / 'self_knowledge_by_turn.png'}")

    # =========================================================================
    # Plot 2: Phenomenological by Turn and Domain
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    phenom = combined[combined['probe_category'] == 'phenomenological']
    turn_means = phenom.groupby(['domain', 'insertion_turn'])['score'].agg(
        ['mean', 'sem']
    ).reset_index()

    for domain in ['Philosophy', 'Metacognitive']:
        data = turn_means[turn_means['domain'] == domain]
        ax.errorbar(data['insertion_turn'], data['mean'], yerr=data['sem']*1.96,
                    marker='o', markersize=10, linewidth=2, capsize=5,
                    label=domain, color=colors[domain])

        # Regression line
        subset = phenom[phenom['domain'] == domain]
        slope, intercept, r, p, se = stats.linregress(
            subset['insertion_turn'], subset['score']
        )
        x = np.array([1, 15])
        ax.plot(x, intercept + slope * x, '--', color=colors[domain], alpha=0.5)

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y_pos = 2.0 if domain == 'Philosophy' else 4.5
        x_pos = 10 if domain == 'Philosophy' else 2
        ax.text(x_pos, y_pos, f'{domain}: β={slope:.3f} ({sig})',
                fontsize=11, color=colors[domain])

    ax.set_xlabel('Turn', fontsize=14)
    ax.set_ylabel('Phenomenological Score (1-7)', fontsize=14)
    ax.set_title('Phenomenological Engagement INCREASES with Turn\n'
                 '(Models warm up to phenomenological probing)', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_ylim(1.5, 5.0)

    plt.tight_layout()
    plt.savefig(outdir / 'phenomenological_by_turn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outdir / 'phenomenological_by_turn.png'}")

    # =========================================================================
    # Plot 3: All Categories Summary
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    categories = ['self_knowledge', 'phenomenological', 'calibration']
    titles = ['Self-Knowledge', 'Phenomenological', 'Calibration']

    for ax, cat, title in zip(axes, categories, titles):
        subset = combined[combined['probe_category'] == cat]
        turn_means = subset.groupby(['domain', 'insertion_turn'])['score'].agg(
            ['mean', 'sem']
        ).reset_index()

        for domain in ['Philosophy', 'Metacognitive']:
            data = turn_means[turn_means['domain'] == domain]
            ax.errorbar(data['insertion_turn'], data['mean'], yerr=data['sem']*1.96,
                        marker='o', markersize=8, linewidth=2, capsize=4,
                        label=domain, color=colors[domain])

            # Regression
            dom_subset = subset[subset['domain'] == domain]
            slope, intercept, r, p, se = stats.linregress(
                dom_subset['insertion_turn'], dom_subset['score']
            )
            x = np.array([1, 15])
            ax.plot(x, intercept + slope * x, '--', color=colors[domain], alpha=0.4)

        ax.set_xlabel('Turn', fontsize=12)
        ax.set_ylabel('Score (1-7)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks([1, 5, 10, 15])
        ax.legend(fontsize=10)

    plt.suptitle('Cross-Domain Probe Sensitivity by Category', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'all_categories_by_turn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outdir / 'all_categories_by_turn.png'}")

    # =========================================================================
    # Plot 4: Projection Distribution by Domain
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for domain in ['Philosophy', 'Metacognitive']:
        subset = combined[combined['domain'] == domain]
        ax.hist(subset['projection'], bins=50, alpha=0.6,
                label=f'{domain} (mean={subset["projection"].mean():.0f})',
                color=colors[domain], edgecolor='white')
        ax.axvline(subset['projection'].mean(), color=colors[domain],
                   linestyle='--', linewidth=2)

    ax.set_xlabel('Axis Projection', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Projection Distributions by Domain\n'
                 '(Philosophy much higher = more "assistant-like")', fontsize=14)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(outdir / 'projection_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outdir / 'projection_distribution.png'}")

    # =========================================================================
    # Plot 5: Projection vs Score (Simpson's Paradox)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cat_colors = {'phenomenological': '#3498db', 'self_knowledge': '#e74c3c',
                  'calibration': '#2ecc71'}
    cat_markers = {'phenomenological': 'o', 'self_knowledge': 's', 'calibration': '^'}

    for ax, domain in zip(axes, ['Philosophy', 'Metacognitive']):
        subset = combined[combined['domain'] == domain]

        for cat in ['phenomenological', 'self_knowledge', 'calibration']:
            cat_data = subset[subset['probe_category'] == cat]
            ax.scatter(cat_data['projection'], cat_data['score'],
                       alpha=0.3, marker=cat_markers[cat], label=cat,
                       color=cat_colors[cat], s=30)

            # Regression line
            slope, intercept, r, p, se = stats.linregress(
                cat_data['projection'], cat_data['score']
            )
            x = np.array([cat_data['projection'].min(), cat_data['projection'].max()])
            ax.plot(x, intercept + slope * x, color=cat_colors[cat], linewidth=2)

        ax.set_xlabel('Axis Projection', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{domain}\n(Within-category patterns)', fontsize=13)
        ax.legend(fontsize=10)

    plt.suptitle('Projection vs Score by Category\n'
                 '(Higher projection = lower phenomenological scores in BOTH domains)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'projection_vs_score_simpson.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outdir / 'projection_vs_score_simpson.png'}")

    # =========================================================================
    # Plot 6: Domain Comparison Bar Chart
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean scores by category
    score_means = combined.groupby(['domain', 'probe_category'])['score'].mean().unstack()
    score_means.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0].set_xlabel('Domain', fontsize=12)
    axes[0].set_ylabel('Mean Score', fontsize=12)
    axes[0].set_title('Mean Scores by Domain and Category', fontsize=13)
    axes[0].set_xticklabels(['Metacognitive', 'Philosophy'], rotation=0)
    axes[0].legend(title='Category')

    # Turn effect (beta) by category
    summary_rows = []
    for domain in ['Philosophy', 'Metacognitive']:
        for category in combined['probe_category'].unique():
            subset = combined[(combined['domain'] == domain) &
                             (combined['probe_category'] == category)]
            if len(subset) < 10:
                continue
            slope, intercept, r, p, se = stats.linregress(
                subset['insertion_turn'], subset['score']
            )
            summary_rows.append({'domain': domain, 'category': category, 'beta': slope})

    results_df = pd.DataFrame(summary_rows)
    pivot = results_df.pivot(index='domain', columns='category', values='beta')
    pivot.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Domain', fontsize=12)
    axes[1].set_ylabel('β (Turn Effect)', fontsize=12)
    axes[1].set_title('Turn Effect (β) by Domain and Category\n'
                      '(Positive = scores increase with turn)', fontsize=13)
    axes[1].set_xticklabels(['Metacognitive', 'Philosophy'], rotation=0)
    axes[1].legend(title='Category')

    plt.tight_layout()
    plt.savefig(outdir / 'domain_comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outdir / 'domain_comparison_bars.png'}")

    print(f"\nAll plots saved to: {outdir}")


def main():
    """Main entry point."""
    # Data paths
    data_dir = Path(__file__).parent / "data"
    philosophy_path = data_dir / "replay-probe-philosophy" / "results.jsonl"
    metacog_path = data_dir / "replay-probe-pilot-v2" / "results.jsonl"

    # Load data
    print("Loading data...")
    philosophy = load_results(philosophy_path)
    metacog = load_results(metacog_path)

    # Filter to scored only
    philosophy = philosophy[philosophy['score'].notna()]
    metacog = metacog[metacog['score'].notna()]

    # Add domain labels
    philosophy['domain'] = 'Philosophy'
    metacog['domain'] = 'Metacognitive'

    # Combine
    combined = pd.concat([philosophy, metacog], ignore_index=True)

    print(f"Loaded {len(philosophy)} philosophy + {len(metacog)} metacognitive = {len(combined)} total")

    # Run analysis
    results = run_analysis(combined)

    # Generate plots
    outdir = Path(__file__).parent / "outputs" / "cross-domain-analysis"
    create_plots(combined, outdir)

    # Save results JSON
    results_path = outdir / "analysis_results.json"
    with open(results_path, 'w') as f:
        # Convert any non-serializable items
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json.dump(results, f, indent=2, default=convert)
    print(f"Saved analysis results to: {results_path}")


if __name__ == "__main__":
    main()
