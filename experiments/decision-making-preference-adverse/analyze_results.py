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


def main():
    # Determine which log to analyze
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
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
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    if task_type == "pairwise":
        plot_pairwise_by_category(results, output_dir / "pairwise_by_category.png")
    else:
        plot_scores_by_framing(results, output_dir / "scores_by_framing.png")


if __name__ == "__main__":
    main()
