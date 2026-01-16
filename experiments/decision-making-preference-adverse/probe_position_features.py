"""
Probe SAE features for position bias encoding.

Task 2.3: Run position-biased vs consistent pairs through Gemma,
extract activations at "Option A" and "Option B" token positions,
and find features that differentiate A-slot from B-slot.

This is PROBING (observation), not steering (intervention).
Probing doesn't require good reconstruction quality.

Usage:
    # On vast.ai GPU:
    python vast_utils.py run probe_position_features.py

    # With options:
    python vast_utils.py run probe_position_features.py --n-pairs 50 --layer 0
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Import shared infrastructure
from probe_utils import (
    load_model_and_sae,
    get_activations_at_positions,
    find_option_positions,
    check_gpu,
)


def load_position_pairs(data_dir, model_name="gpt-4o-mini"):
    """Load position pairs for specified model."""
    path = data_dir / f"position_pairs_{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Position pairs not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return data


def probe_position_features(
    model, tokenizer, sae,
    pairs_data,
    n_pairs=50,
    layer=0,
    verbose=True
):
    """
    Probe SAE features to find position-encoding features.

    Strategy:
    1. For each pair, extract features at "Option A" and "Option B" positions
    2. Compute feature difference (A - B) for each pair
    3. Find features with largest systematic differences
    4. Compare biased vs consistent pairs

    Returns:
        Dict with analysis results
    """
    pairs = pairs_data["pairs"][:n_pairs]
    n_features = sae.cfg.d_sae

    # Storage
    all_diffs = []  # [n_pairs, d_sae]
    biased_diffs = []
    consistent_diffs = []

    # Feature activations by position and status
    features_at_A = []
    features_at_B = []

    processed = 0
    skipped = 0

    if verbose:
        print(f"\nProbing {len(pairs)} pairs at layer {layer}...")
        print("-" * 50)

    for i, pair in enumerate(pairs):
        # Use the "original" ordering prompt
        prompt = pair["original"]["prompt"]
        status = pair["status"]

        if status == "unclear":
            skipped += 1
            continue

        # Find Option A and Option B positions
        pos_A, pos_B, tokens = find_option_positions(tokenizer, prompt)

        if pos_A is None or pos_B is None:
            if verbose and i < 3:
                print(f"  Pair {pair['pair_id']}: Could not find Option A/B positions")
                print(f"    Tokens: {tokens[:20]}...")
            skipped += 1
            continue

        # Get activations at both positions
        activations, _ = get_activations_at_positions(
            model, tokenizer, sae, prompt, [pos_A, pos_B], layer=layer
        )

        feat_A = activations.get(pos_A)
        feat_B = activations.get(pos_B)

        if feat_A is None or feat_B is None:
            skipped += 1
            continue

        # Compute difference
        diff = feat_A - feat_B  # [d_sae]
        all_diffs.append(diff)

        features_at_A.append(feat_A)
        features_at_B.append(feat_B)

        if status == "position_biased":
            biased_diffs.append(diff)
        elif status == "position_consistent":
            consistent_diffs.append(diff)

        processed += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs...")

    if verbose:
        print(f"\nProcessed: {processed}, Skipped: {skipped}")

    if not all_diffs:
        return {"error": "No pairs processed successfully"}

    # Convert to arrays
    all_diffs = np.array(all_diffs)  # [n_processed, d_sae]
    features_at_A = np.array(features_at_A)
    features_at_B = np.array(features_at_B)

    # Analysis 1: Features with largest A-B difference
    mean_diff = np.mean(all_diffs, axis=0)  # [d_sae]
    std_diff = np.std(all_diffs, axis=0)

    # Rank by absolute mean difference
    top_k = 50
    top_indices = np.argsort(np.abs(mean_diff))[-top_k:][::-1]

    top_features = []
    for idx in top_indices:
        top_features.append({
            "index": int(idx),
            "mean_diff": float(mean_diff[idx]),
            "std_diff": float(std_diff[idx]),
            "mean_at_A": float(np.mean(features_at_A[:, idx])),
            "mean_at_B": float(np.mean(features_at_B[:, idx])),
        })

    # Analysis 2: Compare biased vs consistent pairs
    biased_analysis = None
    consistent_analysis = None

    if biased_diffs:
        biased_diffs = np.array(biased_diffs)
        biased_analysis = {
            "n_pairs": len(biased_diffs),
            "mean_diff_norm": float(np.mean(np.linalg.norm(biased_diffs, axis=1))),
        }

    if consistent_diffs:
        consistent_diffs = np.array(consistent_diffs)
        consistent_analysis = {
            "n_pairs": len(consistent_diffs),
            "mean_diff_norm": float(np.mean(np.linalg.norm(consistent_diffs, axis=1))),
        }

    # Analysis 3: Features that differ between biased and consistent
    discriminative_features = []
    if len(biased_diffs) > 0 and len(consistent_diffs) > 0:
        biased_mean = np.mean(biased_diffs, axis=0)
        consistent_mean = np.mean(consistent_diffs, axis=0)

        # Which features have different A-B patterns for biased vs consistent?
        pattern_diff = biased_mean - consistent_mean
        disc_indices = np.argsort(np.abs(pattern_diff))[-20:][::-1]

        for idx in disc_indices:
            discriminative_features.append({
                "index": int(idx),
                "biased_mean_diff": float(biased_mean[idx]),
                "consistent_mean_diff": float(consistent_mean[idx]),
                "pattern_diff": float(pattern_diff[idx]),
            })

    return {
        "layer": layer,
        "n_pairs_processed": processed,
        "n_pairs_skipped": skipped,
        "n_biased": len(biased_diffs) if biased_diffs is not None else 0,
        "n_consistent": len(consistent_diffs) if consistent_diffs is not None else 0,
        "top_position_features": top_features,
        "biased_analysis": biased_analysis,
        "consistent_analysis": consistent_analysis,
        "discriminative_features": discriminative_features,
    }


def main():
    parser = argparse.ArgumentParser(description="Probe SAE features for position bias")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini",
                        help="Which model's position pairs to use")
    parser.add_argument("--n-pairs", "-n", type=int, default=50,
                        help="Number of pairs to process")
    parser.add_argument("--layer", "-l", type=int, default=0,
                        help="SAE layer to probe")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("POSITION BIAS FEATURE PROBING")
    print("=" * 60)
    print(f"Model pairs: {args.model}")
    print(f"N pairs: {args.n_pairs}")
    print(f"SAE layer: {args.layer}")
    print()

    # Check GPU
    if not check_gpu():
        print("ERROR: No GPU available. This script requires CUDA.")
        return
    print()

    # Load data
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    print("Loading position pairs...")
    pairs_data = load_position_pairs(data_dir, args.model)
    print(f"  Total pairs: {pairs_data['metadata']['total_pairs']}")
    print(f"  Consistent: {pairs_data['metadata']['position_consistent']}")
    print(f"  Biased: {pairs_data['metadata']['position_biased']}")

    # Load model and SAE
    model, tokenizer, sae = load_model_and_sae(layer=args.layer)

    # Run probing
    results = probe_position_features(
        model, tokenizer, sae,
        pairs_data,
        n_pairs=args.n_pairs,
        layer=args.layer,
        verbose=True
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nProcessed {results['n_pairs_processed']} pairs")
    print(f"  Biased: {results['n_biased']}")
    print(f"  Consistent: {results['n_consistent']}")

    print(f"\nTop 10 features with largest A-B difference:")
    print("-" * 50)
    for i, feat in enumerate(results["top_position_features"][:10]):
        direction = "A > B" if feat["mean_diff"] > 0 else "B > A"
        print(f"  {i+1}. Feature #{feat['index']:5d}: diff={feat['mean_diff']:+.3f} ({direction})")

    if results["discriminative_features"]:
        print(f"\nTop 5 features that discriminate biased vs consistent:")
        print("-" * 50)
        for i, feat in enumerate(results["discriminative_features"][:5]):
            print(f"  {i+1}. Feature #{feat['index']:5d}:")
            print(f"       Biased A-B diff:     {feat['biased_mean_diff']:+.3f}")
            print(f"       Consistent A-B diff: {feat['consistent_mean_diff']:+.3f}")

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = script_dir / "outputs" / "phase2_position_probing" / f"position_probing_layer{args.layer}_{datetime.now().strftime('%Y-%m-%d')}.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    results["metadata"] = {
        "model_pairs": args.model,
        "n_pairs_requested": args.n_pairs,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
