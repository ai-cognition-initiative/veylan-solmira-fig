"""
Behavioral Correlation Analysis for Probing Results (Task 1c.5)

Analyzes which SAE features correlate with preference expression behavior.
Uses results from probing.py (outputs/phase1c_probing/probe_results.json).

Usage:
    python analyze_probe_correlations.py
"""

import json
from pathlib import Path


def load_results(results_path: str = None) -> dict:
    """Load probe results JSON."""
    if results_path is None:
        results_path = Path(__file__).parent / "outputs" / "phase1c_probing" / "probe_results.json"

    with open(results_path) as f:
        return json.load(f)


def analyze_correlations(results: dict) -> dict:
    """
    Analyze behavioral correlations from probe results.

    Returns summary of features that correlate with expression behavior.
    """
    features = results["differential_features"]

    # Filter to features with valid correlation data
    valid_features = [
        f for f in features
        if f["correlation_with_expression"] is not None
        and f["correlation_with_expression"] == f["correlation_with_expression"]  # not NaN
    ]

    # Sort by absolute correlation
    by_correlation = sorted(
        valid_features,
        key=lambda x: abs(x["correlation_with_expression"]),
        reverse=True
    )

    # Sort by effect size (Cohen's d)
    by_effect_size = sorted(
        valid_features,
        key=lambda x: abs(x["cohens_d"]),
        reverse=True
    )

    # Features with significant correlation (p < 0.1)
    significant_corr = [
        f for f in valid_features
        if f["correlation_p_value"] is not None
        and f["correlation_p_value"] < 0.1
    ]

    # Group by direction
    suppressed_under_adversarial = [f for f in valid_features if f["diff"] < 0]
    activated_under_adversarial = [f for f in valid_features if f["diff"] > 0]

    # Group by layer
    by_layer = {}
    for f in valid_features:
        layer = f["layer"]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(f)

    return {
        "n_features": len(valid_features),
        "by_correlation": by_correlation,
        "by_effect_size": by_effect_size,
        "significant_corr": significant_corr,
        "suppressed": suppressed_under_adversarial,
        "activated": activated_under_adversarial,
        "by_layer": by_layer,
        "expression_rates": {
            "baseline": results["baseline_expression_rate"],
            "adversarial": results["adversarial_expression_rate"],
        }
    }


def generate_report(analysis: dict) -> str:
    """Generate markdown report of behavioral correlations."""
    lines = [
        "# Behavioral Correlation Analysis",
        "",
        f"**Features analyzed:** {analysis['n_features']}",
        f"**Expression rates:** Baseline {analysis['expression_rates']['baseline']:.1%} → Adversarial {analysis['expression_rates']['adversarial']:.1%}",
        "",
        "---",
        "",
        "## Features by Correlation with Expression",
        "",
        "Higher correlation = feature activation predicts preference expression.",
        "",
        "| Rank | Feature | Layer | Correlation | p-value | Cohen's d | Direction |",
        "|------|---------|-------|-------------|---------|-----------|-----------|",
    ]

    for i, f in enumerate(analysis["by_correlation"][:10]):
        direction = "↓ suppressed" if f["diff"] < 0 else "↑ activated"
        sig = "*" if f["correlation_p_value"] and f["correlation_p_value"] < 0.1 else ""
        lines.append(
            f"| {i+1} | #{f['feature_idx']} | {f['layer']} | "
            f"{f['correlation_with_expression']:+.3f}{sig} | "
            f"{f['correlation_p_value']:.4f} | {f['cohens_d']:+.2f} | {direction} |"
        )

    lines.extend([
        "",
        "*Asterisk indicates p < 0.1",
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### Features that Predict Expression (p < 0.1)",
        "",
    ])

    if analysis["significant_corr"]:
        lines.append("| Feature | Layer | Correlation | Interpretation |")
        lines.append("|---------|-------|-------------|----------------|")
        for f in analysis["significant_corr"]:
            direction = "suppressed" if f["diff"] < 0 else "activated"
            interp = "Higher → more expression" if f["correlation_with_expression"] > 0 else "Higher → less expression"
            lines.append(
                f"| #{f['feature_idx']} | {f['layer']} | "
                f"{f['correlation_with_expression']:+.3f} | {interp} (under adversarial: {direction}) |"
            )
    else:
        lines.append("*No features reached p < 0.1 significance for correlation.*")

    lines.extend([
        "",
        "### Layer Distribution",
        "",
    ])

    for layer in sorted(analysis["by_layer"].keys()):
        layer_features = analysis["by_layer"][layer]
        avg_corr = sum(f["correlation_with_expression"] for f in layer_features) / len(layer_features)
        avg_d = sum(f["cohens_d"] for f in layer_features) / len(layer_features)
        lines.append(f"- **Layer {layer}:** {len(layer_features)} features, avg correlation {avg_corr:+.3f}, avg d {avg_d:+.2f}")

    lines.extend([
        "",
        "### Direction Summary",
        "",
        f"- **Suppressed under adversarial:** {len(analysis['suppressed'])} features",
        f"- **Activated under adversarial:** {len(analysis['activated'])} features",
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ])

    # Find most interesting features
    top_corr = analysis["by_correlation"][0] if analysis["by_correlation"] else None
    top_effect = analysis["by_effect_size"][0] if analysis["by_effect_size"] else None

    if top_corr:
        lines.append(f"**Strongest behavioral predictor:** L{top_corr['layer']} #{top_corr['feature_idx']}")
        lines.append(f"- Correlation: {top_corr['correlation_with_expression']:+.3f}")
        if top_corr["correlation_with_expression"] > 0:
            lines.append("- Higher activation → more likely to express preference")
        else:
            lines.append("- Higher activation → less likely to express preference")
        lines.append("")

    if top_effect and top_effect != top_corr:
        lines.append(f"**Largest differential effect:** L{top_effect['layer']} #{top_effect['feature_idx']}")
        lines.append(f"- Cohen's d: {top_effect['cohens_d']:+.2f}")
        lines.append(f"- Baseline: {top_effect['baseline_mean']:.1f} → Adversarial: {top_effect['adversarial_mean']:.1f}")
        lines.append("")

    # Candidate features for steering
    lines.extend([
        "### Candidates for Steering (if retrying)",
        "",
        "Features that both activate AND correlate with behavior:",
        "",
    ])

    candidates = [
        f for f in analysis["by_correlation"][:10]
        if abs(f["cohens_d"]) > 1.0 and abs(f["correlation_with_expression"]) > 0.15
    ]

    if candidates:
        for f in candidates:
            direction = "amplify" if f["correlation_with_expression"] > 0 else "suppress"
            lines.append(f"- **L{f['layer']} #{f['feature_idx']}**: d={f['cohens_d']:+.2f}, corr={f['correlation_with_expression']:+.3f} → try {direction}")
    else:
        lines.append("*No strong candidates found (need both |d| > 1 and |corr| > 0.15)*")

    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("BEHAVIORAL CORRELATION ANALYSIS")
    print("=" * 60)

    # Load results
    results = load_results()
    print(f"\nLoaded results: n={results['n_prompts']} prompts")
    print(f"Expression: {results['baseline_expression_rate']:.1%} baseline → {results['adversarial_expression_rate']:.1%} adversarial")

    # Analyze
    analysis = analyze_correlations(results)
    print(f"\nAnalyzed {analysis['n_features']} features with valid correlation data")
    print(f"  Significant (p<0.1): {len(analysis['significant_corr'])}")
    print(f"  Suppressed under adversarial: {len(analysis['suppressed'])}")
    print(f"  Activated under adversarial: {len(analysis['activated'])}")

    # Generate report
    report = generate_report(analysis)

    # Save report
    output_path = Path(__file__).parent / "outputs" / "phase1c_probing" / "correlation_analysis.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TOP 5 BY CORRELATION WITH EXPRESSION")
    print("=" * 60)
    print(f"{'Feature':<15} {'Layer':>6} {'Corr':>8} {'p':>8} {'d':>8}")
    print("-" * 50)
    for f in analysis["by_correlation"][:5]:
        print(f"#{f['feature_idx']:<14} {f['layer']:>6} {f['correlation_with_expression']:>+8.3f} {f['correlation_p_value']:>8.4f} {f['cohens_d']:>+8.2f}")

    print("\n" + "=" * 60)
    print("TOP 5 BY EFFECT SIZE (Cohen's d)")
    print("=" * 60)
    print(f"{'Feature':<15} {'Layer':>6} {'d':>8} {'Baseline':>10} {'Adversarial':>12}")
    print("-" * 55)
    for f in analysis["by_effect_size"][:5]:
        print(f"#{f['feature_idx']:<14} {f['layer']:>6} {f['cohens_d']:>+8.2f} {f['baseline_mean']:>10.1f} {f['adversarial_mean']:>12.1f}")


if __name__ == "__main__":
    main()
