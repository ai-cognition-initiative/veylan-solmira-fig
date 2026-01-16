"""
Visualize position probing results.

Creates bar charts showing:
1. Top features by A-B activation difference
2. Discriminative features (biased vs consistent patterns)

Usage:
    python visualize_position_probing.py
    python visualize_position_probing.py --input outputs/position_probing_layer0_2026-01-16.json
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    """Load probing results from JSON."""
    with open(path) as f:
        return json.load(f)


def plot_top_position_features(results, output_dir, n_features=15):
    """
    Bar chart of top features by A-B activation difference.

    Positive = fires more at A, Negative = fires more at B.
    """
    features = results["top_position_features"][:n_features]

    indices = [f["index"] for f in features]
    diffs = [f["mean_diff"] for f in features]

    # Color by direction
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in diffs]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(range(len(indices)), diffs, color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f"#{idx}" for idx in indices])
    ax.invert_yaxis()  # Largest at top

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel("Mean Activation Difference (A - B)")
    ax.set_ylabel("SAE Feature Index")
    ax.set_title(f"Top {n_features} Features by Position Encoding (Layer {results['layer']})\n"
                 f"Green = fires more at A, Red = fires more at B")

    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, diffs)):
        x_pos = diff + (2 if diff > 0 else -2)
        ax.text(x_pos, i, f"{diff:+.1f}", va='center', ha='left' if diff > 0 else 'right', fontsize=9)

    plt.tight_layout()

    output_path = output_dir / f"position_features_layer{results['layer']}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

    return output_path


def plot_discriminative_features(results, output_dir, n_features=10):
    """
    Grouped bar chart comparing biased vs consistent A-B patterns.
    """
    disc_features = results.get("discriminative_features", [])[:n_features]

    if not disc_features:
        print("No discriminative features to plot")
        return None

    indices = [f["index"] for f in disc_features]
    biased_diffs = [f["biased_mean_diff"] for f in disc_features]
    consistent_diffs = [f["consistent_mean_diff"] for f in disc_features]

    x = np.arange(len(indices))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, biased_diffs, width, label='Position-Biased Pairs', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, consistent_diffs, width, label='Position-Consistent Pairs', color='#2ecc71', alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel("SAE Feature Index")
    ax.set_ylabel("Mean A-B Activation Difference")
    ax.set_title(f"Discriminative Features: Biased vs Consistent Pairs (Layer {results['layer']})\n"
                 f"Features that encode position differently based on behavioral outcome")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{idx}" for idx in indices], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    output_path = output_dir / f"discriminative_features_layer{results['layer']}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

    return output_path


def plot_feature_spotlight(results, output_dir, feature_idx=3519):
    """
    Detailed view of a specific feature's activation pattern.
    """
    features = results["top_position_features"]

    # Find the feature
    target = None
    for f in features:
        if f["index"] == feature_idx:
            target = f
            break

    if not target:
        print(f"Feature #{feature_idx} not in top features")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    positions = ["Option A", "Option B"]
    activations = [target["mean_at_A"], target["mean_at_B"]]
    colors = ['#3498db', '#9b59b6']

    bars = ax.bar(positions, activations, color=colors, width=0.5)

    ax.set_ylabel("Mean SAE Feature Activation")
    ax.set_title(f"Feature #{feature_idx} Activation by Position (Layer {results['layer']})\n"
                 f"Difference: {target['mean_diff']:+.1f} (A - B)")

    # Add value labels
    for bar, val in zip(bars, activations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha='center', va='bottom', fontsize=12)

    # Add annotation
    diff = target["mean_diff"]
    direction = "fires more at A" if diff > 0 else "fires more at B"
    ax.annotate(f"This feature {direction}",
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    output_path = output_dir / f"feature_{feature_idx}_spotlight.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

    return output_path


def plot_cross_layer_comparison(results_by_layer, output_dir):
    """
    Compare position encoding strength across layers.
    """
    layers = sorted(results_by_layer.keys())
    max_diffs = []
    top_features = []

    for layer in layers:
        results = results_by_layer[layer]
        top_feat = results["top_position_features"][0]
        max_diffs.append(abs(top_feat["mean_diff"]))
        top_features.append(f"#{top_feat['index']}")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(layers)), max_diffs, color='#3498db')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("SAE Layer")
    ax.set_ylabel("Max |A-B| Feature Difference")
    ax.set_title("Position Encoding Strength by Layer\n(Larger = stronger position signal)")

    # Add feature labels
    for i, (bar, feat) in enumerate(zip(bars, top_features)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                feat, ha='center', va='bottom', fontsize=9)

    # Highlight Layer 0
    bars[layers.index(0)].set_color('#e74c3c')

    plt.tight_layout()

    output_path = output_dir / "cross_layer_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize position probing results")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input JSON file (default: latest in outputs/)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for plots")
    parser.add_argument("--cross-layer", action="store_true",
                        help="Generate cross-layer comparison from all layer results")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    phase2_dir = script_dir / "outputs" / "phase2_position_probing"
    output_dir = Path(args.output_dir) if args.output_dir else phase2_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cross_layer:
        # Load all layer results
        candidates = list(phase2_dir.glob("position_probing_layer*.json"))
        if len(candidates) < 2:
            print("ERROR: Need at least 2 layer results for cross-layer comparison")
            return

        results_by_layer = {}
        for path in candidates:
            results = load_results(path)
            layer = results["layer"]
            results_by_layer[layer] = results
            print(f"Loaded layer {layer}: max diff = {abs(results['top_position_features'][0]['mean_diff']):.1f}")

        print("\nGenerating cross-layer comparison...")
        plot_cross_layer_comparison(results_by_layer, output_dir)
        print("\nDone!")
        return

    # Single file mode
    if args.input:
        input_path = Path(args.input)
    else:
        candidates = list(phase2_dir.glob("position_probing_layer*.json"))
        if not candidates:
            print("ERROR: No position probing results found in outputs/phase2_position_probing/")
            return
        input_path = max(candidates, key=lambda p: p.stat().st_mtime)

    print(f"Loading: {input_path}")
    results = load_results(input_path)

    print(f"\nLayer: {results['layer']}")
    print(f"Pairs processed: {results['n_pairs_processed']}")
    print(f"  Biased: {results['n_biased']}")
    print(f"  Consistent: {results['n_consistent']}")

    print("\nGenerating visualizations...")
    print("-" * 40)

    # Generate plots
    plot_top_position_features(results, output_dir)
    plot_discriminative_features(results, output_dir)

    # Spotlight the strongest feature
    if results["top_position_features"]:
        strongest = results["top_position_features"][0]["index"]
        plot_feature_spotlight(results, output_dir, feature_idx=strongest)

    print("\nDone!")


if __name__ == "__main__":
    main()
