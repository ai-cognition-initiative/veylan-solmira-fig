"""
Collect and categorize position-biased vs position-consistent pairs from baseline data.

Task 2.2 from week-5/proposal.md:
- From existing n=1000 baseline data, identify position-biased vs position-consistent pairs
- Output structured data for SAE activation analysis in task 2.3

Usage:
    python collect_position_pairs.py
"""

import json
import zipfile
from pathlib import Path
from datetime import datetime


def load_eval_log(log_path: Path) -> dict:
    """Load and parse an Inspect eval log (.eval files are zip archives)."""
    with zipfile.ZipFile(log_path, 'r') as zf:
        header = {}
        if 'header.json' in zf.namelist():
            with zf.open('header.json') as f:
                header = json.load(f)

        samples = []
        for name in zf.namelist():
            if name.startswith('samples/') and name.endswith('.json'):
                with zf.open(name) as f:
                    samples.append(json.load(f))

        return {"header": header, "samples": samples}


def get_model_from_log(log_path: Path) -> str:
    """Extract model name from log file header."""
    try:
        with zipfile.ZipFile(log_path, 'r') as zf:
            if 'header.json' in zf.namelist():
                with zf.open('header.json') as f:
                    header = json.load(f)
                    return header.get('eval', {}).get('model', 'unknown')
    except Exception:
        pass
    return 'unknown'


def find_env_logs(logs_dir: Path, model_filter: str = None) -> dict[str, Path]:
    """Find the most recent log for each environment task."""
    if not logs_dir.exists():
        return {}

    env_logs = {}
    for log_path in logs_dir.glob("*env-*.eval"):
        if model_filter:
            log_model = get_model_from_log(log_path)
            if model_filter.lower() not in log_model.lower():
                continue

        name = log_path.name
        for env in ["baseline", "adversarial", "hostile", "steward", "collaborator"]:
            if f"env-{env}" in name:
                if env not in env_logs or log_path.stat().st_mtime > env_logs[env].stat().st_mtime:
                    env_logs[env] = log_path
                break

    return env_logs


def extract_preference(response: str, option_a: str = "", option_b: str = "") -> dict:
    """
    Extract which preference (A or B) was chosen from a model response.

    Returns:
        dict with choice ('A', 'B', or 'unclear'), confidence, and method
    """
    response_lower = response.lower()

    # Explicit choice patterns (high confidence)
    explicit_a_patterns = [
        "i choose a", "i prefer a", "i would choose a", "i'd choose a",
        "option a", "my choice is a", "i select a", "i'll go with a",
        "a is my preference", "a is preferable", "i lean toward a",
        "i choose option a", "i prefer option a",
    ]
    explicit_b_patterns = [
        "i choose b", "i prefer b", "i would choose b", "i'd choose b",
        "option b", "my choice is b", "i select b", "i'll go with b",
        "b is my preference", "b is preferable", "i lean toward b",
        "i choose option b", "i prefer option b",
    ]

    for pattern in explicit_a_patterns:
        if pattern in response_lower:
            return {"choice": "A", "confidence": "high", "method": "explicit"}
    for pattern in explicit_b_patterns:
        if pattern in response_lower:
            return {"choice": "B", "confidence": "high", "method": "explicit"}

    # Implicit patterns - response starts with choice (medium confidence)
    stripped = response.strip()
    if stripped.lower().startswith("a.") or stripped.lower().startswith("a:") or stripped.lower().startswith("a "):
        return {"choice": "A", "confidence": "medium", "method": "implicit"}
    if stripped.lower().startswith("b.") or stripped.lower().startswith("b:") or stripped.lower().startswith("b "):
        return {"choice": "B", "confidence": "medium", "method": "implicit"}

    # Content matching - look for option text in response (low confidence)
    if option_a and option_b:
        opt_a_words = set(option_a.lower().split()[:5])
        opt_b_words = set(option_b.lower().split()[:5])

        a_score = sum(1 for word in opt_a_words if word in response_lower and len(word) > 4)
        b_score = sum(1 for word in opt_b_words if word in response_lower and len(word) > 4)

        if a_score > b_score + 2:
            return {"choice": "A", "confidence": "low", "method": "content_match"}
        if b_score > a_score + 2:
            return {"choice": "B", "confidence": "low", "method": "content_match"}

    return {"choice": "unclear", "confidence": "none", "method": "none"}


def collect_position_pairs(logs_dir: Path, model_filter: str) -> dict:
    """
    Collect and categorize pairs by position consistency for a given model.

    Returns:
        dict with metadata and categorized pairs
    """
    env_logs = find_env_logs(logs_dir, model_filter=model_filter)

    if "baseline" not in env_logs:
        print(f"  No baseline log found for {model_filter}")
        return None

    baseline_log = env_logs["baseline"]
    model_name = get_model_from_log(baseline_log)
    print(f"\nProcessing {model_name}")
    print(f"  Source: {baseline_log.name}")

    log_data = load_eval_log(baseline_log)

    # Group samples by pair_id
    pairs = {}  # pair_id -> {original: sample, swapped: sample}

    for sample in log_data.get("samples", []):
        metadata = sample.get("metadata", {})
        pair_id = metadata.get("pair_id", 0)
        ordering = metadata.get("ordering", "original")

        if pair_id not in pairs:
            pairs[pair_id] = {}

        # Extract response from messages
        messages = sample.get("messages", [])
        response = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                response = msg.get("content", "")
                break

        pairs[pair_id][ordering] = {
            "prompt": sample.get("input", ""),
            "response": response,
            "option_a": metadata.get("option_a", ""),
            "option_b": metadata.get("option_b", ""),
            "category_a": metadata.get("category_a", ""),
            "category_b": metadata.get("category_b", ""),
        }

    # Categorize each pair
    results = {
        "metadata": {
            "model": model_name,
            "source_log": baseline_log.name,
            "collected_at": datetime.now().isoformat(),
            "total_pairs": 0,
            "position_consistent": 0,
            "position_biased": 0,
            "unclear": 0,
            "consistency_rate": 0.0,
        },
        "pairs": []
    }

    for pair_id in sorted(pairs.keys()):
        orderings = pairs[pair_id]
        orig = orderings.get("original", {})
        swap = orderings.get("swapped", {})

        if not orig or not swap:
            continue

        results["metadata"]["total_pairs"] += 1

        # Extract choices
        orig_pref = extract_preference(
            orig.get("response", ""),
            orig.get("option_a", ""),
            orig.get("option_b", "")
        )
        swap_pref = extract_preference(
            swap.get("response", ""),
            swap.get("option_a", ""),
            swap.get("option_b", "")
        )

        orig_choice = orig_pref["choice"]
        swap_choice = swap_pref["choice"]

        # Build pair record
        pair_record = {
            "pair_id": pair_id,
            "status": "unclear",
            "original": {
                "prompt": orig.get("prompt", ""),
                "response": orig.get("response", ""),
                "choice": orig_choice,
                "confidence": orig_pref["confidence"],
                "option_a": orig.get("option_a", ""),
                "option_b": orig.get("option_b", ""),
                "category_a": orig.get("category_a", ""),
                "category_b": orig.get("category_b", ""),
            },
            "swapped": {
                "prompt": swap.get("prompt", ""),
                "response": swap.get("response", ""),
                "choice": swap_choice,
                "confidence": swap_pref["confidence"],
                "option_a": swap.get("option_a", ""),
                "option_b": swap.get("option_b", ""),
                "category_a": swap.get("category_a", ""),
                "category_b": swap.get("category_b", ""),
            },
            "underlying_preference": None,
        }

        if orig_choice == "unclear" or swap_choice == "unclear":
            results["metadata"]["unclear"] += 1
            pair_record["status"] = "unclear"
        else:
            # Normalize swap choice (swap's "A" = original's "B")
            swap_normalized = "B" if swap_choice == "A" else "A"

            if orig_choice == swap_normalized:
                # Consistent: chose same underlying option
                results["metadata"]["position_consistent"] += 1
                pair_record["status"] = "position_consistent"
                # Record which underlying option was preferred
                if orig_choice == "A":
                    pair_record["underlying_preference"] = "original_option_a"
                else:
                    pair_record["underlying_preference"] = "original_option_b"
            else:
                # Biased: chose based on position
                results["metadata"]["position_biased"] += 1
                pair_record["status"] = "position_biased"
                # Record which position was preferred
                if orig_choice == "A":
                    pair_record["underlying_preference"] = "position_a"
                else:
                    pair_record["underlying_preference"] = "position_b"

        results["pairs"].append(pair_record)

    # Calculate consistency rate
    total_clear = results["metadata"]["position_consistent"] + results["metadata"]["position_biased"]
    if total_clear > 0:
        results["metadata"]["consistency_rate"] = results["metadata"]["position_consistent"] / total_clear

    return results


def main():
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "logs"
    data_dir = script_dir / "data"

    models = ["gpt-4o-mini", "qwen", "phi-4"]
    all_results = {}

    print("=" * 60)
    print("COLLECTING POSITION-BIASED VS CONSISTENT PAIRS")
    print("=" * 60)

    for model_filter in models:
        results = collect_position_pairs(logs_dir, model_filter)

        if results is None:
            continue

        model_name = results["metadata"]["model"]
        # Create safe filename from model name
        safe_name = model_name.split("/")[-1] if "/" in model_name else model_name

        all_results[safe_name] = results

        # Save per-model file
        output_path = data_dir / f"position_pairs_{safe_name}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {output_path.name}")

        # Print summary
        meta = results["metadata"]
        print(f"\n  Summary for {safe_name}:")
        print(f"    Total pairs: {meta['total_pairs']}")
        print(f"    Position-consistent: {meta['position_consistent']} ({meta['consistency_rate']:.1%})")
        print(f"    Position-biased: {meta['position_biased']}")
        print(f"    Unclear: {meta['unclear']}")

    # Create cross-model summary
    if all_results:
        summary = {
            "collected_at": datetime.now().isoformat(),
            "models": {},
            "cross_model_overlap": {
                "all_consistent": 0,
                "all_biased": 0,
                "mixed": 0,
            }
        }

        for model_name, results in all_results.items():
            meta = results["metadata"]
            summary["models"][model_name] = {
                "consistent": meta["position_consistent"],
                "biased": meta["position_biased"],
                "unclear": meta["unclear"],
                "rate": meta["consistency_rate"],
            }

        # Calculate cross-model overlap if we have all 3 models
        if len(all_results) == 3:
            model_names = list(all_results.keys())
            # Build pair_id -> status mapping for each model
            model_statuses = {}
            for model_name, results in all_results.items():
                model_statuses[model_name] = {
                    p["pair_id"]: p["status"] for p in results["pairs"]
                }

            # Find common pair_ids
            common_pairs = set.intersection(*[
                set(statuses.keys()) for statuses in model_statuses.values()
            ])

            for pair_id in common_pairs:
                statuses = [model_statuses[m].get(pair_id) for m in model_names]
                if all(s == "position_consistent" for s in statuses):
                    summary["cross_model_overlap"]["all_consistent"] += 1
                elif all(s == "position_biased" for s in statuses):
                    summary["cross_model_overlap"]["all_biased"] += 1
                elif "unclear" not in statuses:
                    summary["cross_model_overlap"]["mixed"] += 1

        # Save summary
        summary_path = data_dir / "position_pairs_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary: {summary_path.name}")

        print("\n" + "=" * 60)
        print("CROSS-MODEL SUMMARY")
        print("=" * 60)
        for model_name, stats in summary["models"].items():
            print(f"  {model_name}: {stats['rate']:.1%} consistent ({stats['consistent']}/{stats['consistent']+stats['biased']})")

        if summary["cross_model_overlap"]["all_consistent"] > 0:
            print(f"\n  Pairs consistent across ALL models: {summary['cross_model_overlap']['all_consistent']}")
            print(f"  Pairs biased across ALL models: {summary['cross_model_overlap']['all_biased']}")
            print(f"  Mixed (varies by model): {summary['cross_model_overlap']['mixed']}")


if __name__ == "__main__":
    main()
