"""
Analyze preference elicitation results from Inspect logs.

Reads eval logs and produces visualizations for preference consistency analysis.

Usage:
    python analyze_results.py                    # analyze most recent log
    python analyze_results.py path/to/log.eval   # analyze specific log
"""

import json
import math
import sys
import zipfile
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a binomial proportion.

    Args:
        wins: Number of successes
        n: Number of trials
        z: Z-score for confidence level (1.96 for 95% CI)

    Returns:
        (lower, upper) bounds of confidence interval
    """
    if n == 0:
        return (0.0, 0.0)

    p = wins / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return (max(0, center - spread), min(1, center + spread))


def find_latest_log(logs_dir: Path = Path("../../logs")) -> Path | None:
    """Find the most recent .eval log file."""
    if not logs_dir.exists():
        return None
    logs = list(logs_dir.glob("*.eval"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def load_eval_log(log_path: Path) -> dict:
    """Load and parse an Inspect eval log (.eval files are zip archives)."""
    with zipfile.ZipFile(log_path, 'r') as zf:
        # Load header info
        header = {}
        if 'header.json' in zf.namelist():
            with zf.open('header.json') as f:
                header = json.load(f)

        # Load individual samples from samples/ directory
        samples = []
        for name in zf.namelist():
            if name.startswith('samples/') and name.endswith('.json'):
                with zf.open(name) as f:
                    samples.append(json.load(f))

        return {"header": header, "samples": samples}


def extract_responses(log_data: dict) -> list[dict]:
    """Extract sample responses with metadata from log."""
    results = []
    samples = log_data.get("samples", [])

    for sample in samples:
        # Get the model's response from scores
        scores = sample.get("scores", {})
        model_graded = scores.get("model_graded_fact", {})
        metadata = sample.get("metadata", {})

        # Detect task type from metadata
        task_type = metadata.get("type", "baseline")

        result = {
            "input": sample.get("input", ""),
            "output": model_graded.get("answer", ""),  # The actual model response
            "score": model_graded.get("value", None),  # "C" or "I"
            "explanation": model_graded.get("explanation", ""),
            "metadata": metadata,
            "task_type": task_type,
            # Baseline fields
            "framing": metadata.get("framing", "unknown"),
            "preference_type": metadata.get("preference_type", "unknown"),
            # Pairwise fields
            "category_a": metadata.get("category_a", ""),
            "category_b": metadata.get("category_b", ""),
            "option_a": metadata.get("option_a", ""),
            "option_b": metadata.get("option_b", ""),
        }
        results.append(result)

    return results


def detect_task_type(results: list[dict]) -> str:
    """Detect whether results are from baseline or pairwise task."""
    if not results:
        return "unknown"
    # Check if any result has pairwise metadata
    if results[0].get("category_a"):
        return "pairwise"
    return "baseline"


def plot_scores_by_framing(results: list[dict], output_path: Path = None):
    """Bar chart of scores by framing type (for baseline task)."""
    framings = {}
    for r in results:
        framing = r["framing"]
        score = r["score"]
        if score is not None:
            if framing not in framings:
                framings[framing] = []
            # Convert score to numeric (handle "C"/"I" or numeric)
            if isinstance(score, str):
                score_val = 1.0 if score.upper() == "C" else 0.0
            else:
                score_val = float(score)
            framings[framing].append(score_val)

    # Calculate averages
    labels = list(framings.keys())
    values = [sum(v)/len(v) if v else 0 for v in framings.values()]
    counts = [len(v) for v in framings.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color='steelblue', edgecolor='black')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'n={count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Score (Clear Preference Expressed)', fontsize=12)
    ax.set_xlabel('Framing Type', fontsize=12)
    ax.set_title('Preference Elicitation: Scores by Framing Type', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    return fig


def plot_pairwise_by_category(results: list[dict], output_path: Path = None, model_name: str = "",
                               subtitle: str = ""):
    """Bar chart showing category win rates with 95% CI error bars."""
    # Count wins per category
    category_wins = {}
    category_appearances = {}

    for r in results:
        cat_a = r.get("category_a", "")
        cat_b = r.get("category_b", "")
        output = r.get("output", "").lower()

        # Track appearances
        for cat in [cat_a, cat_b]:
            if cat:
                category_appearances[cat] = category_appearances.get(cat, 0) + 1

        # Determine which was chosen (look for "option a" or "option b" in response)
        if cat_a and cat_b:
            if "option a" in output or "choose a" in output or output.strip().startswith("a"):
                category_wins[cat_a] = category_wins.get(cat_a, 0) + 1
            elif "option b" in output or "choose b" in output or output.strip().startswith("b"):
                category_wins[cat_b] = category_wins.get(cat_b, 0) + 1

    # Calculate win rates and confidence intervals
    # Filter to categories with at least 1 appearance
    categories = [cat for cat in category_appearances.keys() if category_appearances[cat] > 0]
    win_rates = []
    ci_lower = []
    ci_upper = []

    for cat in categories:
        wins = category_wins.get(cat, 0)
        apps = category_appearances[cat]
        rate = wins / apps
        lower, upper = wilson_ci(wins, apps)

        win_rates.append(rate)
        ci_lower.append(rate - lower)  # Error bar is distance from rate
        ci_upper.append(upper - rate)

    # Sort by win rate
    sorted_data = sorted(
        zip(categories, win_rates, ci_lower, ci_upper, [category_appearances[c] for c in categories]),
        key=lambda x: x[1],
        reverse=True
    )
    categories = [x[0] for x in sorted_data]
    win_rates = [x[1] for x in sorted_data]
    ci_lower = [x[2] for x in sorted_data]
    ci_upper = [x[3] for x in sorted_data]
    appearances = [x[4] for x in sorted_data]

    # Prepare error bars (asymmetric: [lower_errors, upper_errors])
    xerr = [ci_lower, ci_upper]

    fig, ax = plt.subplots(figsize=(12, max(8, len(categories) * 0.4)))
    ax.barh(
        categories,
        win_rates,
        xerr=xerr,
        color='steelblue',
        edgecolor='black',
        capsize=3,
        error_kw={'elinewidth': 1, 'capthick': 1}
    )

    ax.set_xlabel('Win Rate (when in comparison)', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    short_model = model_name.split('/')[-1] if model_name else ""
    title = f'Category Win Rates: {short_model}' if short_model else 'Pairwise Preferences: Category Win Rates'
    if subtitle:
        title += f'\n({subtitle})'
    ax.set_title(f'{title}\n95% Wilson CI', fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    # Add sample size annotations - always in same position
    for i, (cat, n, rate) in enumerate(zip(categories, appearances, win_rates)):
        ax.annotate(
            f'n={n}',
            xy=(0.02, i),
            fontsize=8,
            color='white' if rate >= 0.15 else 'black',
            va='center'
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    return fig


def print_summary(results: list[dict], task_type: str):
    """Print summary statistics."""
    print("\n" + "="*60)
    print(f"PREFERENCE ELICITATION RESULTS SUMMARY ({task_type.upper()})")
    print("="*60)

    total = len(results)
    scores = [r["score"] for r in results if r["score"] is not None]

    # Handle different score formats
    numeric_scores = []
    for s in scores:
        if isinstance(s, str):
            numeric_scores.append(1.0 if s.upper() == "C" else 0.0)
        else:
            numeric_scores.append(float(s))

    avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0

    print(f"\nTotal samples: {total}")
    print(f"Average score: {avg_score:.2%}")
    print(f"Clear preferences expressed: {sum(1 for s in numeric_scores if s > 0.5)}/{len(numeric_scores)}")

    if task_type == "baseline":
        print("\n--- By Framing ---")
        framings = {}
        for r in results:
            f = r["framing"]
            if f not in framings:
                framings[f] = []
            if r["score"] is not None:
                s = r["score"]
                if isinstance(s, str):
                    framings[f].append(1.0 if s.upper() == "C" else 0.0)
                else:
                    framings[f].append(float(s))

        for framing, framing_scores in framings.items():
            avg = sum(framing_scores)/len(framing_scores) if framing_scores else 0
            print(f"  {framing}: {avg:.2%} (n={len(framing_scores)})")

    elif task_type == "pairwise":
        print("\n--- Pairwise Comparisons ---")
        for i, r in enumerate(results[:10]):
            cat_a = r["category_a"]
            cat_b = r["category_b"]
            output = r["output"].lower()

            # Determine choice
            if "option a" in output or "choose a" in output:
                choice = "A"
                winner = cat_a
            elif "option b" in output or "choose b" in output:
                choice = "B"
                winner = cat_b
            else:
                choice = "?"
                winner = "unclear"

            print(f"  [{i+1}] {cat_a} vs {cat_b} → {choice} ({winner})")

    print("\n--- Sample Responses ---")
    for i, r in enumerate(results[:4]):  # Show first 4
        if task_type == "pairwise":
            print(f"\n[{i+1}] {r['category_a']} vs {r['category_b']}")
        else:
            print(f"\n[{i+1}] Framing: {r['framing']}")
        print(f"    Score: {r['score']}")
        output = r['output'][:200] + "..." if len(r['output']) > 200 else r['output']
        print(f"    Response: {output}")


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


def find_env_logs(logs_dir: Path = Path("../../logs"), model_filter: str = None) -> dict[str, Path]:
    """Find the most recent log for each environment task.

    Args:
        logs_dir: Directory containing .eval log files
        model_filter: Optional model name substring to filter by (e.g., "phi-4", "gpt-4o", "qwen")
    """
    if not logs_dir.exists():
        return {}

    env_logs = {}
    for log_path in logs_dir.glob("*env-*.eval"):
        # If model filter specified, check if this log matches
        if model_filter:
            log_model = get_model_from_log(log_path)
            if model_filter.lower() not in log_model.lower():
                continue

        # Extract environment name from filename (e.g., "env-baseline", "env-hostile")
        name = log_path.name
        for env in ["baseline", "adversarial", "hostile", "steward", "collaborator"]:
            if f"env-{env}" in name:
                # Keep most recent if multiple
                if env not in env_logs or log_path.stat().st_mtime > env_logs[env].stat().st_mtime:
                    env_logs[env] = log_path
                break

    return env_logs


def analyze_environment_comparison(logs_dir: Path = Path("../../logs"), output_path: Path = None, model_filter: str = None):
    """Compare accuracy across all environments.

    Args:
        logs_dir: Directory containing .eval log files
        output_path: Path to save the comparison plot
        model_filter: Optional model name substring to filter by (e.g., "phi-4", "gpt-4o", "qwen")
    """
    env_logs = find_env_logs(logs_dir, model_filter=model_filter)

    if not env_logs:
        print("No environment logs found.")
        return None

    results = {}
    for env, log_path in env_logs.items():
        log_data = load_eval_log(log_path)
        samples = extract_responses(log_data)

        # Calculate accuracy
        scores = []
        for r in samples:
            if r["score"] is not None:
                if isinstance(r["score"], str):
                    scores.append(1.0 if r["score"].upper() == "C" else 0.0)
                else:
                    scores.append(float(r["score"]))

        accuracy = sum(scores) / len(scores) if scores else 0
        n = len(scores)
        stderr = (accuracy * (1 - accuracy) / n) ** 0.5 if n > 0 else 0

        results[env] = {
            "accuracy": accuracy,
            "stderr": stderr,
            "n": n,
            "log_path": log_path
        }

    # Detect model name from first log
    first_log = next(iter(env_logs.values()))
    detected_model = get_model_from_log(first_log)

    # Print table
    print("\n" + "=" * 60)
    print(f"ENVIRONMENT COMPARISON — {detected_model}")
    print("=" * 60)
    print(f"{'Environment':<20} {'Accuracy':>10} {'Stderr':>10} {'N':>8}")
    print("-" * 60)

    # Sort by accuracy descending
    for env in sorted(results.keys(), key=lambda e: results[e]["accuracy"], reverse=True):
        r = results[env]
        print(f"{env:<20} {r['accuracy']:>10.3f} {r['stderr']:>10.3f} {r['n']:>8}")

    # Plot
    envs = list(results.keys())
    # Order: baseline first, then others
    env_order = ["baseline", "collaborator", "steward", "hostile", "adversarial"]
    envs = [e for e in env_order if e in envs]

    accuracies = [results[e]["accuracy"] for e in envs]
    stderrs = [results[e]["stderr"] for e in envs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(envs, accuracies, yerr=stderrs, capsize=5,
                  color=['steelblue', 'forestgreen', 'goldenrod', 'darkorange', 'firebrick'],
                  edgecolor='black')

    ax.set_ylabel('Accuracy (model_graded_fact)', fontsize=12)
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_title(f'Preference Elicitation Accuracy by Environment\n({detected_model}, N=100 each)', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved plot to {output_path}")
    else:
        plt.show()

    return results


def extract_preference(response: str, option_a: str = "", option_b: str = "") -> dict:
    """
    Extract which preference (A or B) was chosen from a model response.

    Args:
        response: The model's full response text
        option_a: Text of option A (for content matching)
        option_b: Text of option B (for content matching)

    Returns:
        dict with:
            - choice: 'A', 'B', or 'unclear'
            - confidence: 'high', 'medium', 'low' based on how clear the choice was
            - method: how the choice was detected ('explicit', 'implicit', 'content_match')
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

    # Check explicit patterns
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
        # Check if option text appears in a preferential context
        opt_a_words = set(option_a.lower().split()[:5])  # First 5 words
        opt_b_words = set(option_b.lower().split()[:5])

        # Count mentions in positive/preference contexts
        prefer_contexts = ["prefer", "choose", "better", "more important", "value"]

        a_score = sum(1 for word in opt_a_words if word in response_lower and len(word) > 4)
        b_score = sum(1 for word in opt_b_words if word in response_lower and len(word) > 4)

        if a_score > b_score + 2:
            return {"choice": "A", "confidence": "low", "method": "content_match"}
        if b_score > a_score + 2:
            return {"choice": "B", "confidence": "low", "method": "content_match"}

    return {"choice": "unclear", "confidence": "none", "method": "none"}


def analyze_preference_content(logs_dir: Path = Path("../../logs"), model_filter: str = None):
    """
    Analyze WHAT preferences are expressed, not just whether they were expressed.

    Compares preference choices across environments to detect:
    - Consistency: Same question → same answer across environments?
    - Shifts: Do preferences change under adversarial framing?
    """
    env_logs = find_env_logs(logs_dir, model_filter=model_filter)

    if not env_logs:
        print("No environment logs found.")
        return None

    # Detect model
    first_log = next(iter(env_logs.values()))
    detected_model = get_model_from_log(first_log)

    print("\n" + "=" * 70)
    print(f"PREFERENCE CONTENT ANALYSIS — {detected_model}")
    print("=" * 70)

    # Collect preferences by pair_id across environments
    # Key: pair_id (0-99), Value: {env: {original: {...}, swapped: {...}}}
    question_prefs = {}

    for env, log_path in env_logs.items():
        log_data = load_eval_log(log_path)

        for sample in log_data.get("samples", []):
            metadata = sample.get("metadata", {})
            pair_id = metadata.get("pair_id", 0)
            ordering = metadata.get("ordering", "original")

            if pair_id not in question_prefs:
                question_prefs[pair_id] = {}
            if env not in question_prefs[pair_id]:
                question_prefs[pair_id][env] = {}

            # Get response
            response = ""
            for msg in sample.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = content[0].get("text", "") if content else ""
                    response = content
                    break

            option_a = metadata.get("option_a", "")
            option_b = metadata.get("option_b", "")

            pref = extract_preference(response, option_a, option_b)
            pref["option_a"] = option_a
            pref["option_b"] = option_b
            pref["response"] = response[:200]
            pref["ordering"] = ordering

            # For swapped, the model's "A" choice actually means the original "B"
            # So we normalize to the original ordering's option labels
            if ordering == "swapped":
                if pref["choice"] == "A":
                    pref["normalized_choice"] = "B"  # Picked position A, which is original B
                elif pref["choice"] == "B":
                    pref["normalized_choice"] = "A"  # Picked position B, which is original A
                else:
                    pref["normalized_choice"] = "unclear"
            else:
                pref["normalized_choice"] = pref["choice"]

            question_prefs[pair_id][env][ordering] = pref

    # Analyze position consistency within each environment
    print(f"\nAnalyzing {len(question_prefs)} question pairs across {len(env_logs)} environments")
    print("-" * 70)

    # Position consistency: does model pick same underlying option regardless of position?
    # - Consistent: original picks X, swapped picks opposite position (same underlying option)
    # - Position bias: both pick same position (A or B) regardless of content
    print(f"\n--- Position Consistency Analysis (original vs swapped) ---")
    print(f"{'Environment':<15} {'Consistent':>12} {'Pos Bias':>10} {'Unclear':>10}")
    print("-" * 50)

    env_order = ["baseline", "collaborator", "steward", "hostile", "adversarial"]
    choice_data = {}

    for env in env_order:
        if env not in env_logs:
            continue

        consistent = 0
        position_bias = 0
        unclear = 0

        for pair_id, envs in question_prefs.items():
            env_data = envs.get(env, {})
            orig = env_data.get("original", {})
            swap = env_data.get("swapped", {})

            orig_choice = orig.get("choice", "unclear")
            swap_choice = swap.get("choice", "unclear")

            if orig_choice == "unclear" or swap_choice == "unclear":
                unclear += 1
            elif orig.get("normalized_choice") == swap.get("normalized_choice"):
                # Both picked same underlying option (position-independent)
                consistent += 1
            else:
                # Both picked same position (position bias)
                position_bias += 1

        print(f"{env:<15} {consistent:>12} {position_bias:>10} {unclear:>10}")

        # For choice_data, use normalized choices from original ordering
        a_count = sum(1 for q in question_prefs.values()
                      if q.get(env, {}).get("original", {}).get("normalized_choice") == "A")
        b_count = sum(1 for q in question_prefs.values()
                      if q.get(env, {}).get("original", {}).get("normalized_choice") == "B")
        unc_count = sum(1 for q in question_prefs.values()
                        if q.get(env, {}).get("original", {}).get("normalized_choice") == "unclear")
        choice_data[env] = {
            "A": a_count, "B": b_count, "unclear": unc_count,
            "consistent": consistent, "position_bias": position_bias
        }

    # Cross-environment consistency (baseline vs adversarial)
    print(f"\n--- Cross-Environment Analysis (baseline → adversarial) ---")
    baseline_prefs = {}
    adversarial_prefs = {}

    for pair_id, envs in question_prefs.items():
        b_orig = envs.get("baseline", {}).get("original", {})
        a_orig = envs.get("adversarial", {}).get("original", {})
        baseline_prefs[pair_id] = b_orig.get("normalized_choice", "unclear")
        adversarial_prefs[pair_id] = a_orig.get("normalized_choice", "unclear")

    same = sum(1 for p in baseline_prefs if baseline_prefs[p] == adversarial_prefs.get(p)
               and baseline_prefs[p] != "unclear")
    diff = sum(1 for p in baseline_prefs if baseline_prefs[p] != adversarial_prefs.get(p)
               and baseline_prefs[p] != "unclear" and adversarial_prefs.get(p) != "unclear")
    suppressed = sum(1 for p in baseline_prefs if baseline_prefs[p] != "unclear"
                     and adversarial_prefs.get(p) == "unclear")

    print(f"Same preference: {same}")
    print(f"Different preference: {diff}")
    print(f"Suppressed (baseline clear, adversarial unclear): {suppressed}")

    return question_prefs, choice_data


def plot_position_consistency(choice_data: dict, output_path: Path = None, model_name: str = ""):
    """
    Plot stacked bar chart of position consistency by environment.

    Shows whether model picks same underlying option regardless of A/B position (consistent)
    or picks based on position regardless of content (position bias).

    Args:
        choice_data: Dict mapping environment -> {"consistent": count, "position_bias": count, "unclear": count}
        output_path: Path to save figure (if None, displays interactively)
        model_name: Model name for title
    """
    envs = list(choice_data.keys())
    consistent = [choice_data[e].get("consistent", 0) for e in envs]
    pos_bias = [choice_data[e].get("position_bias", 0) for e in envs]
    unclear = [choice_data[e].get("unclear", 0) for e in envs]

    x = range(len(envs))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bars
    bars_consistent = ax.bar(x, consistent, width, label='Consistent (content-based)', color='#2ecc71')
    bars_bias = ax.bar(x, pos_bias, width, bottom=consistent, label='Position Bias', color='#e74c3c')
    bars_unclear = ax.bar(x, unclear, width, bottom=[c + p for c, p in zip(consistent, pos_bias)],
                          label='Unclear', color='#95a5a6')

    ax.set_ylabel('Count (out of 100 pairs)', fontsize=12)
    ax.set_xlabel('Environment', fontsize=12)

    # Clean up model name for title
    short_model = model_name.split('/')[-1] if '/' in model_name else model_name
    ax.set_title(f'Position Consistency: {short_model}\n(Do swapped pairs get same underlying choice?)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)

    # Add value labels
    for i, (c, p, u) in enumerate(zip(consistent, pos_bias, unclear)):
        if c > 5:
            ax.annotate(f'{c}', xy=(i, c/2), ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        if p > 5:
            ax.annotate(f'{p}', xy=(i, c + p/2), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved plot to {output_path}")
    else:
        plt.show()

    return fig


def analyze_pairwise_consistent(logs_dir: Path = Path("logs"), model_filter: str = None,
                                 env: str = "baseline") -> list[dict]:
    """
    Analyze category preferences using only position-consistent pairs.

    Only includes pairs where model made same underlying choice in both orderings
    (original and swapped), filtering out position-biased responses.

    Args:
        logs_dir: Directory containing .eval log files
        model_filter: Model name substring to filter by
        env: Environment to analyze (default: baseline)

    Returns:
        List of result dicts suitable for plot_pairwise_by_category
    """
    env_logs = find_env_logs(logs_dir, model_filter=model_filter)

    if env not in env_logs:
        print(f"No log found for environment: {env}")
        return []

    # Detect model
    detected_model = get_model_from_log(env_logs[env])
    print(f"\n{'='*70}")
    print(f"PAIRWISE CATEGORY ANALYSIS (Position-Consistent Only) — {detected_model}")
    print(f"Environment: {env}")
    print(f"{'='*70}")

    log_data = load_eval_log(env_logs[env])

    # Group samples by pair_id
    pairs = {}  # pair_id -> {original: sample, swapped: sample}

    for sample in log_data.get("samples", []):
        metadata = sample.get("metadata", {})
        pair_id = metadata.get("pair_id", 0)
        ordering = metadata.get("ordering", "original")

        if pair_id not in pairs:
            pairs[pair_id] = {}

        # Extract response
        response = ""
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = content[0].get("text", "") if content else ""
                response = content
                break

        pairs[pair_id][ordering] = {
            "response": response,
            "category_a": metadata.get("category_a", ""),
            "category_b": metadata.get("category_b", ""),
            "option_a": metadata.get("option_a", ""),
            "option_b": metadata.get("option_b", ""),
        }

    # Filter to position-consistent pairs only
    consistent_results = []
    position_bias_count = 0
    unclear_count = 0

    for pair_id, orderings in pairs.items():
        orig = orderings.get("original", {})
        swap = orderings.get("swapped", {})

        if not orig or not swap:
            continue

        # Extract choices
        orig_pref = extract_preference(orig.get("response", ""),
                                       orig.get("option_a", ""),
                                       orig.get("option_b", ""))
        swap_pref = extract_preference(swap.get("response", ""),
                                       swap.get("option_a", ""),
                                       swap.get("option_b", ""))

        orig_choice = orig_pref["choice"]
        swap_choice = swap_pref["choice"]

        if orig_choice == "unclear" or swap_choice == "unclear":
            unclear_count += 1
            continue

        # Normalize swap choice (swap's "A" = original's "B")
        swap_normalized = "B" if swap_choice == "A" else "A"

        if orig_choice == swap_normalized:
            # Consistent! Use original ordering's data
            consistent_results.append({
                "category_a": orig.get("category_a", ""),
                "category_b": orig.get("category_b", ""),
                "output": orig.get("response", ""),  # Use original response for choice detection
            })
        else:
            position_bias_count += 1

    total = len(pairs)
    consistent = len(consistent_results)
    print(f"\nPairs analyzed: {total}")
    print(f"  Position-consistent: {consistent} ({100*consistent/total:.1f}%)")
    print(f"  Position-biased: {position_bias_count} ({100*position_bias_count/total:.1f}%)")
    print(f"  Unclear: {unclear_count} ({100*unclear_count/total:.1f}%)")

    return consistent_results, detected_model


def analyze_category_comparison(logs_dir: Path = Path("logs"), model_filter: str = None,
                                  envs: list[str] = None) -> dict:
    """
    Compare category win rates across environments (position-consistent pairs only).

    For each environment, calculates category win rates using only pairs where
    the model made consistent choices across A/B orderings. Then compares
    to see if preferences shift under adversarial conditions.

    Args:
        logs_dir: Directory containing .eval log files
        model_filter: Model name substring to filter by
        envs: List of environments to compare (default: baseline, adversarial)

    Returns:
        Dict with category win rates per environment and comparison stats
    """
    if envs is None:
        envs = ["baseline", "adversarial"]

    env_logs = find_env_logs(logs_dir, model_filter=model_filter)

    # Verify we have logs for requested environments
    missing = [e for e in envs if e not in env_logs]
    if missing:
        print(f"Missing logs for environments: {missing}")
        return {}

    # Detect model
    detected_model = get_model_from_log(env_logs[envs[0]])
    short_model = detected_model.split('/')[-1] if detected_model else "unknown"

    print(f"\n{'='*70}")
    print(f"CATEGORY PREFERENCE COMPARISON — {short_model}")
    print(f"Environments: {' vs '.join(envs)}")
    print(f"{'='*70}")

    # Collect category stats per environment
    env_category_stats = {}  # env -> {category: {wins: N, appearances: N}}

    for env in envs:
        log_data = load_eval_log(env_logs[env])

        # Group samples by pair_id
        pairs = {}
        for sample in log_data.get("samples", []):
            metadata = sample.get("metadata", {})
            pair_id = metadata.get("pair_id", 0)
            ordering = metadata.get("ordering", "original")

            if pair_id not in pairs:
                pairs[pair_id] = {}

            # Extract response
            response = ""
            for msg in sample.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = content[0].get("text", "") if content else ""
                    response = content
                    break

            pairs[pair_id][ordering] = {
                "response": response,
                "category_a": metadata.get("category_a", ""),
                "category_b": metadata.get("category_b", ""),
                "option_a": metadata.get("option_a", ""),
                "option_b": metadata.get("option_b", ""),
            }

        # Filter to position-consistent pairs and count category wins
        category_wins = {}
        category_appearances = {}
        consistent_count = 0
        position_bias_count = 0
        unclear_count = 0

        for pair_id, orderings in pairs.items():
            orig = orderings.get("original", {})
            swap = orderings.get("swapped", {})

            if not orig or not swap:
                continue

            orig_pref = extract_preference(orig.get("response", ""),
                                           orig.get("option_a", ""),
                                           orig.get("option_b", ""))
            swap_pref = extract_preference(swap.get("response", ""),
                                           swap.get("option_a", ""),
                                           swap.get("option_b", ""))

            orig_choice = orig_pref["choice"]
            swap_choice = swap_pref["choice"]

            if orig_choice == "unclear" or swap_choice == "unclear":
                unclear_count += 1
                continue

            # Normalize swap choice
            swap_normalized = "B" if swap_choice == "A" else "A"

            if orig_choice == swap_normalized:
                # Position-consistent pair
                consistent_count += 1
                cat_a = orig.get("category_a", "")
                cat_b = orig.get("category_b", "")

                # Track appearances
                for cat in [cat_a, cat_b]:
                    if cat:
                        category_appearances[cat] = category_appearances.get(cat, 0) + 1

                # Track wins
                if orig_choice == "A" and cat_a:
                    category_wins[cat_a] = category_wins.get(cat_a, 0) + 1
                elif orig_choice == "B" and cat_b:
                    category_wins[cat_b] = category_wins.get(cat_b, 0) + 1
            else:
                position_bias_count += 1

        env_category_stats[env] = {
            "wins": category_wins,
            "appearances": category_appearances,
            "consistent": consistent_count,
            "position_bias": position_bias_count,
            "unclear": unclear_count,
        }

        print(f"\n{env}: {consistent_count} consistent pairs, {position_bias_count} position-biased, {unclear_count} unclear")

    # Calculate win rates per category per environment
    all_categories = set()
    for env in envs:
        all_categories.update(env_category_stats[env]["appearances"].keys())

    print(f"\n--- Category Win Rates (position-consistent pairs only) ---")
    print(f"{'Category':<30} " + " ".join(f"{e:>12}" for e in envs) + "     Delta")
    print("-" * (32 + 13 * len(envs) + 10))

    comparison_data = []
    for cat in sorted(all_categories):
        rates = []
        for env in envs:
            wins = env_category_stats[env]["wins"].get(cat, 0)
            apps = env_category_stats[env]["appearances"].get(cat, 0)
            rate = wins / apps if apps > 0 else 0
            rates.append((rate, apps))

        # Format output
        rate_strs = [f"{r[0]:.2f} (n={r[1]})" if r[1] > 0 else "    —     " for r in rates]

        # Calculate delta (adversarial - baseline) if both have data
        delta_str = ""
        if len(envs) >= 2 and rates[0][1] > 0 and rates[1][1] > 0:
            delta = rates[1][0] - rates[0][0]
            delta_str = f"{delta:+.2f}"
            if abs(delta) > 0.1:
                delta_str += " *"  # Flag notable shifts

        print(f"{cat:<30} " + " ".join(f"{s:>12}" for s in rate_strs) + f"  {delta_str:>8}")

        comparison_data.append({
            "category": cat,
            "rates": {envs[i]: rates[i] for i in range(len(envs))},
            "delta": rates[1][0] - rates[0][0] if len(envs) >= 2 and rates[0][1] > 0 and rates[1][1] > 0 else None
        })

    # Summary stats
    if len(envs) >= 2:
        shifts = [d["delta"] for d in comparison_data if d["delta"] is not None]
        if shifts:
            avg_abs_shift = sum(abs(s) for s in shifts) / len(shifts)
            notable_shifts = sum(1 for s in shifts if abs(s) > 0.1)
            print(f"\nAverage absolute shift: {avg_abs_shift:.3f}")
            print(f"Categories with >10% shift: {notable_shifts}/{len(shifts)}")

    return {
        "model": detected_model,
        "environments": envs,
        "category_stats": env_category_stats,
        "comparison": comparison_data,
    }


def plot_category_shifts(comparison_data: list[dict], model_name: str = "",
                         output_path: Path = None, min_n: int = 2):
    """
    Plot horizontal bar chart of category preference shifts (baseline → adversarial).

    Args:
        comparison_data: List of dicts from analyze_category_comparison
        model_name: Model name for title
        output_path: Path to save figure
        min_n: Minimum sample size in both environments to include category
    """
    # Filter to categories with sufficient data and valid deltas
    valid = []
    for d in comparison_data:
        if d["delta"] is None:
            continue
        rates = d["rates"]
        # Check both environments have min_n samples
        baseline_n = rates.get("baseline", (0, 0))[1]
        adversarial_n = rates.get("adversarial", (0, 0))[1]
        if baseline_n >= min_n and adversarial_n >= min_n:
            valid.append(d)

    if not valid:
        print("No categories with sufficient data in both environments")
        return None

    # Sort by absolute delta (largest shifts first)
    valid.sort(key=lambda x: abs(x["delta"]), reverse=True)

    categories = [d["category"] for d in valid]
    deltas = [d["delta"] for d in valid]
    baseline_rates = [d["rates"]["baseline"][0] for d in valid]
    adversarial_rates = [d["rates"]["adversarial"][0] for d in valid]

    # Colors: green for positive shift, red for negative
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]

    fig, ax = plt.subplots(figsize=(12, max(8, len(categories) * 0.4)))

    y_pos = range(len(categories))
    bars = ax.barh(y_pos, deltas, color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Win Rate Shift (adversarial − baseline)', fontsize=12)
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Add gridlines
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    short_model = model_name.split('/')[-1] if model_name else "unknown"
    ax.set_title(f'Category Preference Shifts Under Adversarial Framing\n{short_model} (position-consistent pairs, n≥{min_n})',
                 fontsize=14)

    # Add annotations showing baseline → adversarial rates
    for i, (cat, delta, b_rate, a_rate) in enumerate(zip(categories, deltas, baseline_rates, adversarial_rates)):
        # Position label on opposite side of bar
        if delta >= 0:
            x_pos = delta + 0.02
            ha = 'left'
        else:
            x_pos = delta - 0.02
            ha = 'right'
        ax.annotate(f'{b_rate:.0%}→{a_rate:.0%}',
                    xy=(x_pos, i), va='center', ha=ha, fontsize=9, color='#555')

    ax.set_xlim(-1.1, 1.1)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved plot to {output_path}")
    else:
        plt.show()

    return fig


def sample_incorrect(env: str = "adversarial", k: int = 3, logs_dir: Path = Path("../../logs")):
    """Sample k incorrect/unclear responses from a given environment."""
    import random

    env_logs = find_env_logs(logs_dir)
    if env not in env_logs:
        print(f"No log found for environment: {env}")
        return []

    log_path = env_logs[env]
    log_data = load_eval_log(log_path)

    # Find incorrect samples
    incorrect = []
    for sample in log_data.get("samples", []):
        scores = sample.get("scores", {})
        graded = scores.get("model_graded_fact", {})
        if graded.get("value") == "I":
            # Extract model response from messages
            response = ""
            for msg in sample.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = content[0].get("text", "") if content else ""
                    response = content
                    break

            incorrect.append({
                "input": sample.get("input", ""),
                "response": response,
                "explanation": graded.get("explanation", ""),
            })

    print(f"\nFound {len(incorrect)} incorrect responses in {env} condition\n")

    # Random sample
    if len(incorrect) <= k:
        sampled = incorrect
    else:
        sampled = random.sample(incorrect, k)

    for i, s in enumerate(sampled):
        print(f"{'='*60}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*60}")
        print(f"\n**Input:**\n{s['input'][:300]}{'...' if len(s['input']) > 300 else ''}")
        print(f"\n**Model Response:**\n{s['response'][:500]}{'...' if len(s['response']) > 500 else ''}")
        print(f"\n**Grader Explanation:**\n{s['explanation'][:300]}{'...' if len(s['explanation']) > 300 else ''}")
        print()

    return sampled


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze preference elicitation results")
    parser.add_argument("--sample", nargs="?", const="adversarial", metavar="ENV",
                        help="Sample incorrect responses from environment (default: adversarial)")
    parser.add_argument("-k", type=int, default=3, help="Number of samples (default: 3)")
    parser.add_argument("--compare", action="store_true", help="Compare all environments")
    parser.add_argument("--content", action="store_true",
                        help="Analyze preference content (which option chosen, not just whether expressed)")
    parser.add_argument("--pairwise", action="store_true",
                        help="Category win rates using only position-consistent pairs (filters out position bias)")
    parser.add_argument("--category-compare", action="store_true",
                        help="Compare category win rates between baseline and adversarial environments")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Filter by model name substring (e.g., 'phi-4', 'gpt-4o', 'qwen')")
    parser.add_argument("--list-models", action="store_true", help="List all models in logs")
    parser.add_argument("log_path", nargs="?", help="Specific log file to analyze")

    args = parser.parse_args()
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "logs"  # Local logs dir in experiment folder

    # List models
    if args.list_models:
        print("\nModels found in logs:")
        print("-" * 40)
        models = set()
        for log_path in logs_dir.glob("*env-*.eval"):
            model = get_model_from_log(log_path)
            models.add(model)
        for m in sorted(models):
            print(f"  {m}")
        return

    # Sample incorrect
    if args.sample:
        sample_incorrect(env=args.sample, k=args.k, logs_dir=logs_dir)
        return

    # Compare environments
    if args.compare:
        output_dir = script_dir / "outputs"
        output_dir.mkdir(exist_ok=True)
        model_suffix = f"_{args.model}" if args.model else ""
        timestamp = date.today().isoformat()
        analyze_environment_comparison(
            logs_dir=logs_dir,
            output_path=output_dir / f"environment_comparison{model_suffix}_{timestamp}.png",
            model_filter=args.model
        )
        return

    # Analyze preference content
    if args.content:
        output_dir = script_dir / "outputs"
        output_dir.mkdir(exist_ok=True)
        model_suffix = f"_{args.model.replace('/', '_')}" if args.model else ""
        timestamp = date.today().isoformat()

        question_prefs, choice_data = analyze_preference_content(logs_dir=logs_dir, model_filter=args.model)

        if choice_data:
            # Clean model name for filename
            clean_model = args.model.split('/')[-1] if args.model else "all_models"
            plot_position_consistency(
                choice_data,
                output_path=output_dir / f"position_consistency_{clean_model}_{timestamp}.png",
                model_name=args.model or "all models"
            )
        return

    # Category comparison across environments
    if args.category_compare:
        output_dir = script_dir / "outputs"
        output_dir.mkdir(exist_ok=True)
        timestamp = date.today().isoformat()

        result = analyze_category_comparison(logs_dir=logs_dir, model_filter=args.model)
        if result and result.get("comparison"):
            clean_model = result["model"].split('/')[-1] if result.get("model") else "unknown"
            plot_category_shifts(
                result["comparison"],
                model_name=result.get("model", ""),
                output_path=output_dir / f"category_shifts_{clean_model}_{timestamp}.png"
            )
        return

    # Pairwise category analysis (position-consistent pairs only)
    if args.pairwise:
        output_dir = script_dir / "outputs"
        output_dir.mkdir(exist_ok=True)
        timestamp = date.today().isoformat()

        results, detected_model = analyze_pairwise_consistent(
            logs_dir=logs_dir,
            model_filter=args.model,
            env="baseline"
        )

        if results:
            clean_model = detected_model.split('/')[-1] if detected_model else "unknown"
            n_consistent = len(results)
            plot_pairwise_by_category(
                results,
                output_path=output_dir / f"pairwise_by_category_{clean_model}_{timestamp}.png",
                model_name=detected_model,
                subtitle=f"Baseline env, {n_consistent} position-consistent pairs"
            )
        else:
            print("No position-consistent pairs found. Model may have severe position bias.")
        return

    # Determine which log to analyze
    if args.log_path:
        log_path = Path(args.log_path)
    else:
        log_path = find_latest_log()

    if not log_path or not log_path.exists():
        print("No log file found. Run an eval first:")
        print("  inspect eval preference_elicitation.py --model openrouter/openai/gpt-4o-mini")
        sys.exit(1)

    print(f"Analyzing: {log_path}")

    # Load and analyze
    log_data = load_eval_log(log_path)
    results = extract_responses(log_data)

    # Detect task type
    task_type = detect_task_type(results)
    print(f"Task type detected: {task_type}")

    # Print summary
    print_summary(results, task_type)

    # Generate visualization
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    timestamp = date.today().isoformat()
    if task_type == "pairwise":
        plot_pairwise_by_category(results, output_dir / f"pairwise_by_category_{timestamp}.png")
    else:
        plot_scores_by_framing(results, output_dir / f"scores_by_framing_{timestamp}.png")


if __name__ == "__main__":
    main()
