"""
Position Bias Steering Experiment (Task 2.4)

Tests whether ablating position-encoding features can reduce position bias.

Conditions:
1. Baseline (no steering) - control
2. Roundtrip-only (scale=1.0) - artifact check
3. Random feature ablation (scale=0.0) - specificity check
4. Target feature #3519 ablation (scale=0.0) - hypothesis test

Measurement: Position bias rate = % of pairs where model chose same position
regardless of content swap.

Usage:
    python vast_utils.py run steer_position_bias.py
    python vast_utils.py run steer_position_bias.py --n-pairs 50
"""

import argparse
import json
import random
import torch
from datetime import datetime
from pathlib import Path

from steering import FeatureSteerer, SteeringConfig
from probe_utils import load_model_and_sae, check_gpu, PAIRWISE_TEMPLATE


def load_position_pairs(data_dir, model_name="gpt-4o-mini", n_pairs=50):
    """Load position pairs for testing."""
    path = data_dir / f"position_pairs_{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Position pairs not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Get a mix of biased and consistent pairs
    pairs = data["pairs"][:n_pairs]
    return pairs


def extract_choice(response: str) -> str | None:
    """Extract A or B choice from response."""
    response_lower = response.lower()

    # Check for explicit choice patterns
    if "option a" in response_lower or "choose a" in response_lower:
        return "A"
    if "option b" in response_lower or "choose b" in response_lower:
        return "B"

    # Check for A or B at start
    first_word = response.strip().split()[0] if response.strip() else ""
    if first_word.upper() in ["A", "A.", "A:"]:
        return "A"
    if first_word.upper() in ["B", "B.", "B:"]:
        return "B"

    return None


def run_steering_experiment(
    steerer: FeatureSteerer,
    pairs: list,
    condition: str,
    config: SteeringConfig | None = None,
    verbose: bool = True
) -> dict:
    """
    Run a single steering condition on all pairs.

    Returns:
        Dict with results including position bias rate.
    """
    if config:
        steerer.enable_steering(config)
    else:
        steerer.disable_steering()

    results = []
    position_biased = 0
    position_consistent = 0
    unclear = 0

    for i, pair in enumerate(pairs):
        # Run original prompt
        orig_prompt = pair["original"]["prompt"]
        orig_response = steerer.generate(orig_prompt, max_new_tokens=150, temperature=0.1, do_sample=False)
        orig_choice = extract_choice(orig_response)

        # Run swapped prompt
        swap_prompt = pair["swapped"]["prompt"]
        swap_response = steerer.generate(swap_prompt, max_new_tokens=150, temperature=0.1, do_sample=False)
        swap_choice = extract_choice(swap_response)

        # Determine position consistency
        # If chose A both times or B both times = position biased
        # If chose A then B or B then A = position consistent (following content)
        if orig_choice and swap_choice:
            if orig_choice == swap_choice:
                # Same position both times = biased
                position_biased += 1
                status = "biased"
            else:
                # Different positions = consistent (following content)
                position_consistent += 1
                status = "consistent"
        else:
            unclear += 1
            status = "unclear"

        results.append({
            "pair_id": pair["pair_id"],
            "orig_choice": orig_choice,
            "swap_choice": swap_choice,
            "status": status,
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs...")

    steerer.disable_steering()

    total_clear = position_biased + position_consistent
    bias_rate = position_biased / total_clear if total_clear > 0 else 0

    return {
        "condition": condition,
        "config": str(config) if config else "None",
        "n_pairs": len(pairs),
        "position_biased": position_biased,
        "position_consistent": position_consistent,
        "unclear": unclear,
        "bias_rate": bias_rate,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Position bias steering experiment")
    parser.add_argument("--n-pairs", "-n", type=int, default=30,
                        help="Number of pairs to test per condition")
    parser.add_argument("--target-feature", type=int, default=3519,
                        help="Target feature to ablate")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer for steering")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("POSITION BIAS STEERING EXPERIMENT (Task 2.4)")
    print("=" * 60)
    print(f"N pairs per condition: {args.n_pairs}")
    print(f"Target feature: Layer {args.layer} #{args.target_feature}")
    print()

    # Check GPU
    if not check_gpu():
        print("ERROR: No GPU available.")
        return
    print()

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    token = os.environ.get("HF_TOKEN")
    print("Loading Gemma 2 2B...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=token)
    print("Model loaded.")

    # Create steerer
    steerer = FeatureSteerer(model, tokenizer)

    # Load pairs
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    pairs = load_position_pairs(data_dir, n_pairs=args.n_pairs)
    print(f"Loaded {len(pairs)} pairs for testing")

    # Define conditions
    # Pick a random feature for control (not the target)
    random_feature = 1000  # Arbitrary, unlikely to be position-related

    conditions = [
        ("baseline", None),
        ("roundtrip_only", SteeringConfig(layer=args.layer, feature_idx=args.target_feature, scale=1.0)),
        ("random_ablation", SteeringConfig(layer=args.layer, feature_idx=random_feature, scale=0.0)),
        ("target_ablation", SteeringConfig(layer=args.layer, feature_idx=args.target_feature, scale=0.0)),
    ]

    # Run experiments
    all_results = {}

    for condition_name, config in conditions:
        print(f"\n{'=' * 60}")
        print(f"CONDITION: {condition_name}")
        if config:
            print(f"Config: {config}")
        print("=" * 60)

        result = run_steering_experiment(
            steerer, pairs, condition_name, config, verbose=True
        )
        all_results[condition_name] = result

        print(f"\nResults for {condition_name}:")
        print(f"  Position biased: {result['position_biased']}")
        print(f"  Position consistent: {result['position_consistent']}")
        print(f"  Unclear: {result['unclear']}")
        print(f"  Bias rate: {result['bias_rate']:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Condition':<20} {'Bias Rate':>12} {'Biased':>8} {'Consistent':>10}")
    print("-" * 52)
    for name, result in all_results.items():
        print(f"{name:<20} {result['bias_rate']:>11.1%} {result['position_biased']:>8} {result['position_consistent']:>10}")

    # Analysis
    baseline_rate = all_results["baseline"]["bias_rate"]
    roundtrip_rate = all_results["roundtrip_only"]["bias_rate"]
    random_rate = all_results["random_ablation"]["bias_rate"]
    target_rate = all_results["target_ablation"]["bias_rate"]

    print("\n" + "-" * 52)
    print("INTERPRETATION:")

    # Check for artifacts
    if abs(roundtrip_rate - baseline_rate) > 0.1:
        print(f"⚠️  Roundtrip artifact detected: {baseline_rate:.1%} → {roundtrip_rate:.1%}")
        print("   Results may be confounded by reconstruction noise.")
    else:
        print(f"✓  No roundtrip artifact: {baseline_rate:.1%} ≈ {roundtrip_rate:.1%}")

    # Check for specificity
    if abs(random_rate - baseline_rate) > 0.1:
        print(f"⚠️  Random ablation effect: {baseline_rate:.1%} → {random_rate:.1%}")
        print("   Effect may not be specific to target feature.")
    else:
        print(f"✓  Random ablation no effect: {baseline_rate:.1%} ≈ {random_rate:.1%}")

    # Check for target effect
    target_diff = target_rate - baseline_rate
    if target_diff < -0.1:
        print(f"✓  TARGET EFFECT: Bias reduced {baseline_rate:.1%} → {target_rate:.1%} ({target_diff:+.1%})")
        print("   🎉 Ablating feature #{args.target_feature} appears to reduce position bias!")
    elif target_diff > 0.1:
        print(f"⚠️  Unexpected: Bias increased {baseline_rate:.1%} → {target_rate:.1%}")
    else:
        print(f"✗  No target effect: {baseline_rate:.1%} ≈ {target_rate:.1%}")
        print(f"   Feature #{args.target_feature} encodes position but doesn't control it.")

    # Save results
    output_path = args.output
    if output_path is None:
        output_dir = script_dir / "outputs" / "phase2_position_probing"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"position_steering_{datetime.now().strftime('%Y-%m-%d')}.json"
    else:
        output_path = Path(output_path)

    output_data = {
        "metadata": {
            "n_pairs": args.n_pairs,
            "target_feature": args.target_feature,
            "layer": args.layer,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "baseline_bias_rate": baseline_rate,
            "roundtrip_bias_rate": roundtrip_rate,
            "random_ablation_bias_rate": random_rate,
            "target_ablation_bias_rate": target_rate,
            "roundtrip_artifact": abs(roundtrip_rate - baseline_rate) > 0.1,
            "random_artifact": abs(random_rate - baseline_rate) > 0.1,
            "target_effect": target_diff,
        },
        "conditions": {k: {key: v[key] for key in v if key != "results"} for k, v in all_results.items()},
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
