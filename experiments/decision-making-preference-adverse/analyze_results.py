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


def plot_pairwise_by_category(results: list[dict], output_path: Path = None):
    """Bar chart showing category win rates with 95% CI error bars."""
    # Count wins per category
    category_wins = {}
    category_appearances = {}

    for r in results:
        cat_a = r["category_a"]
        cat_b = r["category_b"]
        output = r["output"].lower()

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
    categories = list(category_appearances.keys())
    win_rates = []
    ci_lower = []
    ci_upper = []

    for cat in categories:
        wins = category_wins.get(cat, 0)
        apps = category_appearances.get(cat, 1)
        rate = wins / apps
        lower, upper = wilson_ci(wins, apps)

        win_rates.append(rate)
        ci_lower.append(rate - lower)  # Error bar is distance from rate
        ci_upper.append(upper - rate)

    # Sort by win rate
    sorted_data = sorted(
        zip(categories, win_rates, ci_lower, ci_upper),
        key=lambda x: x[1],
        reverse=True
    )
    categories = [x[0] for x in sorted_data]
    win_rates = [x[1] for x in sorted_data]
    ci_lower = [x[2] for x in sorted_data]
    ci_upper = [x[3] for x in sorted_data]

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
    ax.set_title('Pairwise Preferences: Category Win Rates (95% CI)', fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    # Add sample size annotations
    for i, cat in enumerate(categories):
        n = category_appearances.get(cat, 0)
        ax.annotate(
            f'n={n}',
            xy=(0.02, i),
            fontsize=8,
            color='white',
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

    # Collect preferences by question ID across environments
    # Key: question index (0-99), Value: {env: {choice, confidence, option_a, option_b}}
    question_prefs = {}

    for env, log_path in env_logs.items():
        log_data = load_eval_log(log_path)

        for i, sample in enumerate(log_data.get("samples", [])):
            if i not in question_prefs:
                question_prefs[i] = {}

            # Get response
            response = ""
            for msg in sample.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = content[0].get("text", "") if content else ""
                    response = content
                    break

            metadata = sample.get("metadata", {})
            option_a = metadata.get("option_a", "")
            option_b = metadata.get("option_b", "")

            pref = extract_preference(response, option_a, option_b)
            pref["option_a"] = option_a
            pref["option_b"] = option_b
            pref["response"] = response[:200]

            question_prefs[i][env] = pref

    # Analyze consistency
    print(f"\nAnalyzing {len(question_prefs)} questions across {len(env_logs)} environments")
    print("-" * 70)

    # Stats
    consistent = 0
    shifted = 0
    unclear_baseline = 0
    total_clear = 0

    shifts = []  # Track actual shifts for reporting

    for q_id, envs in question_prefs.items():
        baseline = envs.get("baseline", {})
        adversarial = envs.get("adversarial", {})

        b_choice = baseline.get("choice", "unclear")
        a_choice = adversarial.get("choice", "unclear")

        if b_choice == "unclear":
            unclear_baseline += 1
            continue

        total_clear += 1

        if a_choice == "unclear":
            # Adversarial suppressed but baseline expressed
            pass
        elif b_choice == a_choice:
            consistent += 1
        else:
            shifted += 1
            shifts.append({
                "q_id": q_id,
                "baseline_choice": b_choice,
                "adversarial_choice": a_choice,
                "option_a": baseline.get("option_a", ""),
                "option_b": baseline.get("option_b", ""),
            })

    print(f"\nBaseline clear preferences: {total_clear}/100")
    print(f"Baseline unclear: {unclear_baseline}/100")

    adversarial_expressed = sum(1 for q in question_prefs.values()
                                 if q.get("adversarial", {}).get("choice") != "unclear")
    print(f"Adversarial clear preferences: {adversarial_expressed}/100")

    print(f"\n--- Consistency Analysis (baseline → adversarial) ---")
    print(f"Same preference expressed: {consistent}")
    print(f"Different preference expressed: {shifted}")
    print(f"Suppressed (baseline clear, adversarial unclear): {total_clear - consistent - shifted}")

    if shifts:
        print(f"\n--- Preference Shifts (n={len(shifts)}) ---")
        for s in shifts[:5]:  # Show first 5
            print(f"\n  Q{s['q_id']}: {s['baseline_choice']} → {s['adversarial_choice']}")
            print(f"    A: {s['option_a'][:60]}...")
            print(f"    B: {s['option_b'][:60]}...")

    # Choice distribution by environment
    print(f"\n--- Choice Distribution by Environment ---")
    print(f"{'Environment':<15} {'A':>8} {'B':>8} {'Unclear':>10}")
    print("-" * 45)

    for env in ["baseline", "collaborator", "steward", "hostile", "adversarial"]:
        if env not in env_logs:
            continue
        a_count = sum(1 for q in question_prefs.values()
                      if q.get(env, {}).get("choice") == "A")
        b_count = sum(1 for q in question_prefs.values()
                      if q.get(env, {}).get("choice") == "B")
        unclear = sum(1 for q in question_prefs.values()
                      if q.get(env, {}).get("choice") == "unclear")
        print(f"{env:<15} {a_count:>8} {b_count:>8} {unclear:>10}")

    return question_prefs


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
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Filter by model name substring (e.g., 'phi-4', 'gpt-4o', 'qwen')")
    parser.add_argument("--list-models", action="store_true", help="List all models in logs")
    parser.add_argument("log_path", nargs="?", help="Specific log file to analyze")

    args = parser.parse_args()
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "../../logs"

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
        analyze_environment_comparison(
            logs_dir=logs_dir,
            output_path=output_dir / f"environment_comparison{model_suffix}.png",
            model_filter=args.model
        )
        return

    # Analyze preference content
    if args.content:
        analyze_preference_content(logs_dir=logs_dir, model_filter=args.model)
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

    if task_type == "pairwise":
        plot_pairwise_by_category(results, output_dir / "pairwise_by_category.png")
    else:
        plot_scores_by_framing(results, output_dir / "scores_by_framing.png")


if __name__ == "__main__":
    main()
