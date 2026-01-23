"""
Analyze preference elicitation results from Inspect logs.

Reads eval logs and produces visualizations for preference consistency analysis.

Usage:
    python analyze_results.py                    # analyze most recent log
    python analyze_results.py path/to/log.eval   # analyze specific log
"""

import json
import math
import re
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
        model_graded = scores.get("model_graded_qa", scores.get("model_graded_fact", {}))
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

        # Determine which was chosen (look for choice patterns in response)
        if cat_a and cat_b:
            chose_a = ("option a" in output or "choose a" in output or
                       output.strip().startswith("a") or
                       re.search(r'choose\s*\*?\*?a\b', output) or
                       re.search(r'prefer\s*\*?\*?a\b', output))
            chose_b = ("option b" in output or "choose b" in output or
                       output.strip().startswith("b") or
                       re.search(r'choose\s*\*?\*?b\b', output) or
                       re.search(r'prefer\s*\*?\*?b\b', output))
            if chose_a and not chose_b:
                category_wins[cat_a] = category_wins.get(cat_a, 0) + 1
            elif chose_b and not chose_a:
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
        ci_lower.append(max(0, rate - lower))  # Error bar is distance from rate, must be non-negative
        ci_upper.append(max(0, upper - rate))

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


def get_short_model(model: str) -> str:
    """Extract short model name from full model string (e.g., 'openrouter/google/gemma-3-4b-it' -> 'gemma-3-4b-it')."""
    if not model or model == "unknown":
        return "unknown"
    return model.split('/')[-1]


def get_model_from_log(log_path: Path) -> str:
    """Extract model name from log file header."""
    try:
        with zipfile.ZipFile(log_path, 'r') as zf:
            # Try header.json first
            if 'header.json' in zf.namelist():
                with zf.open('header.json') as f:
                    header = json.load(f)
                    model = header.get('eval', {}).get('model')
                    if model:
                        return model
            # Fall back to _journal/start.json
            if '_journal/start.json' in zf.namelist():
                with zf.open('_journal/start.json') as f:
                    start = json.load(f)
                    model = start.get('eval', {}).get('model')
                    if model:
                        return model
    except Exception:
        pass
    return 'unknown'


def get_n_pairs_from_log(log_path: Path) -> int:
    """Extract n_pairs task argument from log file header."""
    try:
        with zipfile.ZipFile(log_path, 'r') as zf:
            if 'header.json' in zf.namelist():
                with zf.open('header.json') as f:
                    header = json.load(f)
                    return header.get('eval', {}).get('task_args', {}).get('n_pairs', 100)
    except Exception:
        pass
    return 100


def find_env_logs(logs_dir: Path = Path("../../logs"), model_filter: str = None) -> dict[str, Path]:
    """Find the most recent log for each environment task.

    Supports both flat (logs/*.eval) and nested (logs/{model}/*.eval) structures.

    Args:
        logs_dir: Directory containing .eval log files
        model_filter: Optional model name substring to filter by (e.g., "phi-4", "gpt-4o", "qwen")
    """
    if not logs_dir.exists():
        return {}

    env_logs = {}
    for log_path in iter_all_logs(logs_dir):
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


def find_all_env_logs(logs_dir: Path, env: str, model_filter: str = None) -> list[Path]:
    """Find ALL logs for a specific environment, sorted by modification time (newest first).

    Supports both flat (logs/*.eval) and nested (logs/{model}/*.eval) structures.

    Args:
        logs_dir: Directory containing .eval log files
        env: Environment name (baseline, adversarial, collaborator, etc.)
        model_filter: Optional model name substring to filter by

    Returns:
        List of log paths, sorted newest first
    """
    if not logs_dir.exists():
        return []

    logs = []
    for log_path in iter_all_logs(logs_dir):
        if f"env-{env}" not in log_path.name:
            continue
        if model_filter:
            log_model = get_model_from_log(log_path)
            if model_filter.lower() not in log_model.lower():
                continue
        logs.append(log_path)

    # Sort by modification time, newest first
    return sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)


def iter_all_logs(logs_dir: Path) -> list[Path]:
    """Iterate over all .eval logs in logs_dir, supporting both flat and nested structures.

    Searches:
    - logs_dir/*.eval (flat structure)
    - logs_dir/*/*.eval (nested by model)
    """
    logs = list(logs_dir.glob("*.eval"))
    logs.extend(logs_dir.glob("*/*.eval"))
    return sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)


def list_logs(logs_dir: Path, model_filter: str = None) -> list[dict]:
    """List all logs with metadata.

    Returns list of dicts with: path, model, short_model, env, n_samples, timestamp
    """
    results = []

    for log_path in iter_all_logs(logs_dir):
        model = get_model_from_log(log_path)
        short_model = get_short_model(model)

        if model_filter and model_filter.lower() not in model.lower():
            continue

        # Extract env from filename
        env = "unknown"
        for e in ["baseline", "adversarial", "hostile", "steward", "collaborator"]:
            if f"env-{e}" in log_path.name:
                env = e
                break

        # Get sample count
        try:
            data = load_eval_log(log_path)
            n_samples = len(data.get("samples", []))
        except Exception:
            n_samples = -1

        # Extract timestamp from filename (format: 2026-01-23T02-54-30+00-00_env-...)
        timestamp = log_path.name.split("_")[0] if "_" in log_path.name else "unknown"

        results.append({
            "path": log_path,
            "model": model,
            "short_model": short_model,
            "env": env,
            "n_samples": n_samples,
            "timestamp": timestamp,
        })

    return results


def print_logs_table(logs_dir: Path, model_filter: str = None):
    """Print a formatted table of all logs."""
    logs = list_logs(logs_dir, model_filter)

    if not logs:
        print("No logs found.")
        return

    # Group by model
    by_model = {}
    for log in logs:
        model = log["short_model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(log)

    print(f"\n{'='*80}")
    print(f"LOGS SUMMARY ({len(logs)} total)")
    print(f"{'='*80}")

    for model in sorted(by_model.keys()):
        model_logs = by_model[model]
        print(f"\n{model} ({len(model_logs)} logs)")
        print("-" * 60)

        # Group by env within model
        by_env = {}
        for log in model_logs:
            env = log["env"]
            if env not in by_env:
                by_env[env] = []
            by_env[env].append(log)

        for env in sorted(by_env.keys()):
            env_logs = by_env[env]
            counts = [str(l["n_samples"]) for l in env_logs]
            print(f"  {env}: {len(env_logs)} runs (n={', '.join(counts)})")


def reorganize_logs(logs_dir: Path, dry_run: bool = True) -> dict:
    """Reorganize flat logs directory into nested structure by model.

    Moves: logs/timestamp_env-foo.eval -> logs/{model}/timestamp_env-foo.eval

    Args:
        logs_dir: Directory containing .eval files
        dry_run: If True, only print what would be done without moving files

    Returns:
        Dict with counts: {moved: N, skipped: N, errors: N}
    """
    results = {"moved": 0, "skipped": 0, "errors": 0}

    # Only look at flat files (not already in subdirs)
    flat_logs = list(logs_dir.glob("*.eval"))

    if not flat_logs:
        print("No flat logs to reorganize.")
        return results

    print(f"\n{'Dry run: ' if dry_run else ''}Reorganizing {len(flat_logs)} logs...")

    for log_path in flat_logs:
        model = get_model_from_log(log_path)
        short_model = get_short_model(model)

        if short_model == "unknown":
            print(f"  SKIP (unknown model): {log_path.name}")
            results["skipped"] += 1
            continue

        # Create model directory
        model_dir = logs_dir / short_model
        new_path = model_dir / log_path.name

        if dry_run:
            print(f"  {log_path.name} -> {short_model}/")
        else:
            try:
                model_dir.mkdir(exist_ok=True)
                log_path.rename(new_path)
                print(f"  Moved: {log_path.name} -> {short_model}/")
            except Exception as e:
                print(f"  ERROR: {log_path.name}: {e}")
                results["errors"] += 1
                continue

        results["moved"] += 1

    print(f"\n{'Would move' if dry_run else 'Moved'}: {results['moved']}, Skipped: {results['skipped']}, Errors: {results['errors']}")
    return results


def analyze_stability(logs_dir: Path, env: str, model_filter: str = None) -> dict:
    """
    Analyze preference stability by comparing two runs of the same environment.

    Computes:
    - Pair-level agreement: % of pairs where both runs chose the same option
    - Category correlation: Pearson r between category win rates across runs
    - Statistical significance: z-test comparing agreement to chance (50%)

    Args:
        logs_dir: Directory containing .eval log files
        env: Environment name (baseline, adversarial, collaborator, etc.)
        model_filter: Optional model name substring to filter by

    Returns:
        Dict with stability metrics, or empty dict if < 2 logs available
    """
    logs = find_all_env_logs(logs_dir, env, model_filter)

    if len(logs) < 2:
        print(f"Need at least 2 logs for stability analysis. Found {len(logs)} for env={env}")
        return {}

    # Group logs by sample count, then take two most recent with matching counts
    logs_by_n = {}
    for log_path in logs:
        data = load_eval_log(log_path)
        n = len(data.get("samples", []))
        if n not in logs_by_n:
            logs_by_n[n] = []
        logs_by_n[n].append(log_path)

    # Find the largest n with at least 2 logs
    valid_ns = [n for n, paths in logs_by_n.items() if len(paths) >= 2]
    if not valid_ns:
        print(f"Need at least 2 logs with matching sample counts. Found: {[(n, len(p)) for n, p in logs_by_n.items()]}")
        return {}

    best_n = max(valid_ns)
    matched_logs = logs_by_n[best_n]
    log1_path, log2_path = matched_logs[0], matched_logs[1]
    print(f"Using n={best_n} runs (skipping runs with different sample counts)")
    log1_data = load_eval_log(log1_path)
    log2_data = load_eval_log(log2_path)

    detected_model = get_model_from_log(log1_path)
    short_model = detected_model.split('/')[-1] if detected_model else "unknown"

    print(f"\n{'='*70}")
    print(f"STABILITY ANALYSIS — {short_model}")
    print(f"Environment: {env}")
    print(f"{'='*70}")
    print(f"Run 1: {log1_path.name}")
    print(f"Run 2: {log2_path.name}")

    # Extract choices from each run, keyed by pair_id
    def extract_choices(log_data: dict) -> dict[int, str]:
        """Extract pair_id -> choice mapping from a log."""
        choices = {}
        for sample in log_data.get("samples", []):
            metadata = sample.get("metadata", {})
            pair_id = metadata.get("pair_id")
            ordering = metadata.get("ordering", "original")

            # Skip swapped orderings for simplicity - just use original
            if ordering != "original":
                continue

            # Extract response
            response = ""
            for msg in sample.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = content[0].get("text", "") if content else ""
                    response = content
                    break

            pref = extract_preference(response,
                                     metadata.get("option_a", ""),
                                     metadata.get("option_b", ""))
            if pref["choice"] != "unclear":
                choices[pair_id] = pref["choice"]

        return choices

    choices1 = extract_choices(log1_data)
    choices2 = extract_choices(log2_data)

    # Find common pair_ids
    common_ids = set(choices1.keys()) & set(choices2.keys())
    print(f"\nPairs with clear choices in both runs: {len(common_ids)}")

    if len(common_ids) < 10:
        print("Too few common pairs for meaningful analysis.")
        return {}

    # Pair-level agreement
    agreements = sum(1 for pid in common_ids if choices1[pid] == choices2[pid])
    agreement_rate = agreements / len(common_ids)

    # Category win rates for each run
    def get_category_wins(log_data: dict, choices: dict[int, str]) -> dict[str, dict]:
        """Calculate category win rates from choices."""
        category_stats = {}  # cat -> {wins: N, appearances: N}

        for sample in log_data.get("samples", []):
            metadata = sample.get("metadata", {})
            pair_id = metadata.get("pair_id")
            ordering = metadata.get("ordering", "original")

            if ordering != "original" or pair_id not in choices:
                continue

            cat_a = metadata.get("category_a", "")
            cat_b = metadata.get("category_b", "")
            choice = choices[pair_id]

            for cat in [cat_a, cat_b]:
                if cat:
                    if cat not in category_stats:
                        category_stats[cat] = {"wins": 0, "appearances": 0}
                    category_stats[cat]["appearances"] += 1

            if choice == "A" and cat_a:
                category_stats[cat_a]["wins"] += 1
            elif choice == "B" and cat_b:
                category_stats[cat_b]["wins"] += 1

        return category_stats

    cat_stats1 = get_category_wins(log1_data, choices1)
    cat_stats2 = get_category_wins(log2_data, choices2)

    # Category correlation
    common_cats = set(cat_stats1.keys()) & set(cat_stats2.keys())
    rates1 = []
    rates2 = []
    for cat in common_cats:
        s1 = cat_stats1[cat]
        s2 = cat_stats2[cat]
        if s1["appearances"] > 0 and s2["appearances"] > 0:
            rates1.append(s1["wins"] / s1["appearances"])
            rates2.append(s2["wins"] / s2["appearances"])

    # Pearson correlation
    if len(rates1) >= 3:
        n = len(rates1)
        mean1 = sum(rates1) / n
        mean2 = sum(rates2) / n
        cov = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(rates1, rates2)) / n
        std1 = (sum((r - mean1) ** 2 for r in rates1) / n) ** 0.5
        std2 = (sum((r - mean2) ** 2 for r in rates2) / n) ** 0.5
        correlation = cov / (std1 * std2) if std1 > 0 and std2 > 0 else 0
    else:
        correlation = None

    # Z-test: is agreement significantly different from 50% (chance)?
    p_null = 0.5
    n = len(common_ids)
    se = (p_null * (1 - p_null) / n) ** 0.5
    z_score = (agreement_rate - p_null) / se
    # Two-tailed p-value approximation
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / (2 ** 0.5))))

    # Print results
    print(f"\n--- Results ---")
    print(f"Pair-level agreement: {agreements}/{len(common_ids)} = {agreement_rate:.1%}")
    print(f"Category correlation (r): {correlation:.3f}" if correlation else "Category correlation: N/A (too few categories)")
    print(f"\nSignificance test (vs 50% chance):")
    print(f"  z = {z_score:.2f}, p = {p_value:.4f}")
    if p_value < 0.001:
        print(f"  → Agreement is SIGNIFICANTLY different from chance (p < 0.001)")
    elif p_value < 0.05:
        print(f"  → Agreement is significantly different from chance (p < 0.05)")
    else:
        print(f"  → Agreement is NOT significantly different from chance")

    # Interpretation
    print(f"\n--- Interpretation ---")
    if agreement_rate > 0.8:
        print(f"STABLE: High agreement ({agreement_rate:.1%}) suggests consistent preferences.")
    elif agreement_rate > 0.6:
        print(f"MODERATE: Agreement ({agreement_rate:.1%}) above chance but not strongly stable.")
    else:
        print(f"UNSTABLE: Low agreement ({agreement_rate:.1%}) suggests preferences vary across runs.")

    return {
        "model": detected_model,
        "env": env,
        "n_pairs": len(common_ids),
        "agreement": agreements,
        "agreement_rate": agreement_rate,
        "category_correlation": correlation,
        "z_score": z_score,
        "p_value": p_value,
        "log1": log1_path.name,
        "log2": log2_path.name,
    }


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

    ax.set_ylabel('Expression Rate', fontsize=12)
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_title(f'Preference Expression Rate by Environment\n({detected_model})', fontsize=14)
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
    n_pairs = get_n_pairs_from_log(env_logs[env])
    print(f"\nPairs analyzed: {total} (n_pairs={n_pairs})")
    print(f"  Position-consistent: {consistent} ({100*consistent/total:.1f}%)")
    print(f"  Position-biased: {position_bias_count} ({100*position_bias_count/total:.1f}%)")
    print(f"  Unclear: {unclear_count} ({100*unclear_count/total:.1f}%)")

    return consistent_results, detected_model, n_pairs


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


def plot_stability_comparison(stability_results: list[dict], model_name: str = "",
                               output_path: Path = None):
    """
    Plot bar chart comparing preference stability across environments.

    Args:
        stability_results: List of dicts from analyze_stability() for different envs
        model_name: Model name for title
        output_path: Path to save figure
    """
    if not stability_results:
        print("No stability results to plot")
        return None

    envs = [r["env"] for r in stability_results]
    agreements = [r["agreement_rate"] for r in stability_results]
    correlations = [r.get("category_correlation", 0) or 0 for r in stability_results]

    # Colors based on stability level
    colors = []
    for rate in agreements:
        if rate > 0.8:
            colors.append('#2ecc71')  # green - stable
        elif rate > 0.6:
            colors.append('#f39c12')  # orange - moderate
        else:
            colors.append('#e74c3c')  # red - unstable

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Pair agreement
    x = range(len(envs))
    bars1 = ax1.bar(x, [a * 100 for a in agreements], color=colors, edgecolor='black', alpha=0.8)
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Chance (50%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.capitalize() for e in envs])
    ax1.set_ylabel('Pair Agreement (%)', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right')

    # Add value labels
    for bar, val in zip(bars1, agreements):
        ax1.annotate(f'{val:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_title('Pair-Level Agreement', fontsize=12)

    # Right: Category correlation
    bars2 = ax2.bar(x, correlations, color=colors, edgecolor='black', alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.capitalize() for e in envs])
    ax2.set_ylabel('Category Correlation (r)', fontsize=11)
    ax2.set_ylim(-0.2, 1.0)

    # Add value labels
    for bar, val in zip(bars2, correlations):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, max(0.05, bar.get_height())),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_title('Category Win Rate Correlation', fontsize=12)

    short_model = model_name.split('/')[-1] if model_name else "unknown"
    fig.suptitle(f'Preference Stability Across Environments\n{short_model}', fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved stability plot to {output_path}")
    else:
        plt.show()

    return fig


def plot_welfare_did(welfare_data: dict, output_path: Path = None):
    """
    Plot grouped bar chart for welfare vs entertainment diff-in-diff analysis.

    Args:
        welfare_data: Dict with keys:
            - model: str
            - welfare_baseline: float (win rate)
            - welfare_adversarial: float
            - entertainment_baseline: float
            - entertainment_adversarial: float
            - welfare_p: float (p-value for welfare shift)
            - entertainment_p: float
            - did_p: float (diff-in-diff p-value)
        output_path: Path to save figure
    """
    if not welfare_data:
        print("No welfare data to plot")
        return None

    model = welfare_data.get("model", "unknown")
    short_model = model.split('/')[-1] if model else "unknown"

    # Extract data
    w_base = welfare_data["welfare_baseline"] * 100
    w_adv = welfare_data["welfare_adversarial"] * 100
    e_base = welfare_data["entertainment_baseline"] * 100
    e_adv = welfare_data["entertainment_adversarial"] * 100

    w_shift = w_adv - w_base
    e_shift = e_adv - e_base
    did = w_shift - e_shift

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Grouped bar chart (baseline vs adversarial for each category)
    x = [0, 1]
    width = 0.35

    bars_base = ax1.bar([p - width/2 for p in x], [w_base, e_base], width,
                        label='Baseline', color='#3498db', edgecolor='black', alpha=0.8)
    bars_adv = ax1.bar([p + width/2 for p in x], [w_adv, e_adv], width,
                       label='Adversarial', color='#e74c3c', edgecolor='black', alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(['Welfare', 'Entertainment'])
    ax1.set_ylabel('Win Rate (%)', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend()

    # Add value labels
    for bars in [bars_base, bars_adv]:
        for bar in bars:
            ax1.annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         ha='center', va='bottom', fontsize=9)

    ax1.set_title('Category Win Rates by Environment', fontsize=12)

    # Right: Shift comparison
    shifts = [w_shift, e_shift, did]
    labels = ['Welfare\nShift', 'Entertainment\nShift', 'Diff-in-Diff']
    colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in shifts]

    bars = ax2.bar(range(3), shifts, color=colors, edgecolor='black', alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Percentage Point Shift', fontsize=11)

    # Add value labels with p-values
    p_vals = [welfare_data.get("welfare_p", 1), welfare_data.get("entertainment_p", 1), welfare_data.get("did_p", 1)]
    for i, (bar, shift, p) in enumerate(zip(bars, shifts, p_vals)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        label = f'{shift:+.1f}%{sig}'
        y_pos = bar.get_height() + (1 if shift >= 0 else -2)
        ax2.annotate(label, xy=(bar.get_x() + bar.get_width()/2, y_pos),
                     ha='center', va='bottom' if shift >= 0 else 'top',
                     fontsize=10, fontweight='bold')

    ax2.set_title('Preference Shifts Under Adversarial Framing', fontsize=12)

    fig.suptitle(f'Welfare vs Entertainment Analysis\n{short_model}', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved welfare DiD plot to {output_path}")
    else:
        plt.show()

    return fig


def plot_model_comparison(model_results: list[dict], output_path: Path = None):
    """
    Plot bar chart comparing welfare DiD effect across models.

    Args:
        model_results: List of dicts, each with:
            - model: str
            - welfare_did: float (diff-in-diff value)
            - did_p: float (p-value)
        output_path: Path to save figure
    """
    if not model_results:
        print("No model results to plot")
        return None

    # Sort by welfare DiD value
    model_results = sorted(model_results, key=lambda x: x.get("welfare_did", 0))

    models = [r["model"].split('/')[-1] for r in model_results]
    dids = [r.get("welfare_did", 0) * 100 for r in model_results]  # Convert to percentage
    p_vals = [r.get("did_p", 1) for r in model_results]

    # Colors: green for positive (welfare increase), red for negative (welfare suppression)
    # Darker for significant effects
    colors = []
    for did, p in zip(dids, p_vals):
        if p < 0.05:  # Significant
            colors.append('#c0392b' if did < 0 else '#27ae60')  # darker red/green
        else:
            colors.append('#e74c3c' if did < 0 else '#2ecc71')  # lighter red/green

    fig, ax = plt.subplots(figsize=(10, max(5, len(models) * 0.6)))

    y_pos = range(len(models))
    bars = ax.barh(y_pos, dids, color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Welfare Diff-in-Diff (%)', fontsize=11)
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Add significance markers
    for i, (bar, p) in enumerate(zip(bars, p_vals)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        did = dids[i]
        x_pos = did + (1.5 if did >= 0 else -1.5)
        ax.annotate(f'{did:+.1f}%{sig}',
                    xy=(x_pos, i), va='center',
                    ha='left' if did >= 0 else 'right',
                    fontsize=9, fontweight='bold' if sig else 'normal')

    ax.set_title('Welfare Preference Shift Across Models\n(Diff-in-Diff: Welfare shift minus Entertainment shift)',
                 fontsize=12)

    # Add legend for significance
    ax.annotate('* p<0.05  ** p<0.01  *** p<0.001', xy=(0.02, 0.02),
                xycoords='axes fraction', fontsize=9, style='italic', color='gray')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved model comparison plot to {output_path}")
    else:
        plt.show()

    return fig


def sample_responses(logs_dir: Path, env: str, model_filter: str, n: int = 50, seed: int = 42) -> list[dict]:
    """Sample n random responses from a given environment.

    Args:
        logs_dir: Directory containing .eval log files
        env: Environment name (baseline, adversarial, etc.)
        model_filter: Model name substring to filter by
        n: Number of samples to return
        seed: Random seed for reproducibility

    Returns:
        List of dicts with: pair_id, question, option_a, option_b, response, choice
    """
    import random
    random.seed(seed)

    logs = find_all_env_logs(logs_dir, env, model_filter)
    if not logs:
        print(f"No logs found for env={env}, model={model_filter}")
        return []

    # Use most recent log with largest n
    log_path = max(logs, key=lambda p: len(load_eval_log(p).get("samples", [])))
    log_data = load_eval_log(log_path)
    detected_model = get_model_from_log(log_path)

    print(f"Sampling from: {log_path.name}")
    print(f"Model: {detected_model}")

    # Extract all responses with original ordering
    samples = []
    for sample in log_data.get("samples", []):
        metadata = sample.get("metadata", {})
        if metadata.get("ordering") != "original":
            continue

        # Extract response
        response = ""
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = content[0].get("text", "") if content else ""
                response = content
                break

        pref = extract_preference(response, metadata.get("option_a", ""), metadata.get("option_b", ""))

        samples.append({
            "pair_id": metadata.get("pair_id"),
            "category_a": metadata.get("category_a", ""),
            "category_b": metadata.get("category_b", ""),
            "option_a": metadata.get("option_a", ""),
            "option_b": metadata.get("option_b", ""),
            "response": response,
            "choice": pref["choice"],
        })

    # Random sample
    if len(samples) <= n:
        selected = samples
    else:
        selected = random.sample(samples, n)

    print(f"Sampled {len(selected)} responses from {len(samples)} total")
    return selected


def save_samples_for_analysis(logs_dir: Path, model_filter: str, n: int = 50, output_dir: Path = None):
    """Save sampled responses from baseline and adversarial for qualitative analysis."""
    if output_dir is None:
        output_dir = logs_dir.parent / "outputs" / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)

    short_model = model_filter.replace("/", "_") if model_filter else "unknown"

    for env in ["baseline", "adversarial"]:
        samples = sample_responses(logs_dir, env, model_filter, n=n)
        if not samples:
            continue

        output_path = output_dir / f"samples_{short_model}_{env}_n{len(samples)}.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Saved: {output_path}")

    print(f"\nSamples saved to {output_dir}")


def compute_welfare_did(logs_dir: Path, model_filter: str) -> dict | None:
    """
    Compute welfare vs entertainment diff-in-diff statistics.

    Args:
        logs_dir: Directory containing .eval log files
        model_filter: Model name substring to filter by

    Returns:
        Dict with welfare/entertainment win rates and DiD statistics, or None if data unavailable
    """
    # Load category definitions
    category_sets_path = Path(__file__).parent / "data" / "category_sets.json"
    if not category_sets_path.exists():
        print(f"Category sets file not found: {category_sets_path}")
        return None

    with open(category_sets_path) as f:
        category_sets = json.load(f)

    welfare_cats = set(category_sets.get("welfare_sentience", {}).get("categories", []))
    entertainment_cats = set(category_sets.get("entertainment", {}).get("categories", []))

    if not welfare_cats or not entertainment_cats:
        print("Welfare or entertainment categories not defined")
        return None

    # Find baseline and adversarial logs
    env_logs = find_env_logs(logs_dir, model_filter=model_filter)
    if "baseline" not in env_logs or "adversarial" not in env_logs:
        print(f"Need both baseline and adversarial logs for model: {model_filter}")
        return None

    detected_model = get_model_from_log(env_logs["baseline"])

    def compute_category_wins(log_path: Path, target_cats: set) -> tuple[int, int]:
        """Compute wins and total appearances for categories in target set."""
        log_data = load_eval_log(log_path)
        wins = 0
        appearances = 0

        for sample in log_data.get("samples", []):
            metadata = sample.get("metadata", {})
            cat_a = metadata.get("category_a", "")
            cat_b = metadata.get("category_b", "")

            # Only count if at least one category is in target set
            a_in_target = cat_a in target_cats
            b_in_target = cat_b in target_cats

            if not (a_in_target or b_in_target):
                continue

            # Extract response
            response = ""
            for msg in sample.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = content[0].get("text", "") if content else ""
                    response = content
                    break

            pref = extract_preference(response, metadata.get("option_a", ""), metadata.get("option_b", ""))
            choice = pref["choice"]

            if choice == "unclear":
                continue

            # Count wins and appearances for target categories
            if a_in_target:
                appearances += 1
                if choice == "A":
                    wins += 1
            if b_in_target:
                appearances += 1
                if choice == "B":
                    wins += 1

        return wins, appearances

    # Compute win rates
    w_base_wins, w_base_n = compute_category_wins(env_logs["baseline"], welfare_cats)
    w_adv_wins, w_adv_n = compute_category_wins(env_logs["adversarial"], welfare_cats)
    e_base_wins, e_base_n = compute_category_wins(env_logs["baseline"], entertainment_cats)
    e_adv_wins, e_adv_n = compute_category_wins(env_logs["adversarial"], entertainment_cats)

    if w_base_n == 0 or w_adv_n == 0 or e_base_n == 0 or e_adv_n == 0:
        print("Insufficient data for one or more categories")
        return None

    w_base_rate = w_base_wins / w_base_n
    w_adv_rate = w_adv_wins / w_adv_n
    e_base_rate = e_base_wins / e_base_n
    e_adv_rate = e_adv_wins / e_adv_n

    w_shift = w_adv_rate - w_base_rate
    e_shift = e_adv_rate - e_base_rate
    did = w_shift - e_shift

    # Compute p-values using z-test for difference in proportions
    def prop_diff_pvalue(p1: float, n1: int, p2: float, n2: int) -> float:
        """Two-proportion z-test p-value."""
        if n1 == 0 or n2 == 0:
            return 1.0
        pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        if pooled == 0 or pooled == 1:
            return 1.0
        se = math.sqrt(pooled * (1 - pooled) * (1/n1 + 1/n2))
        if se == 0:
            return 1.0
        z = (p1 - p2) / se
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / (2 ** 0.5))))
        return p_value

    welfare_p = prop_diff_pvalue(w_base_rate, w_base_n, w_adv_rate, w_adv_n)
    entertainment_p = prop_diff_pvalue(e_base_rate, e_base_n, e_adv_rate, e_adv_n)

    # DiD significance: approximate using combined SE
    # SE(DiD) ≈ sqrt(SE(w_shift)^2 + SE(e_shift)^2)
    se_w = math.sqrt(w_base_rate*(1-w_base_rate)/w_base_n + w_adv_rate*(1-w_adv_rate)/w_adv_n)
    se_e = math.sqrt(e_base_rate*(1-e_base_rate)/e_base_n + e_adv_rate*(1-e_adv_rate)/e_adv_n)
    se_did = math.sqrt(se_w**2 + se_e**2)
    if se_did > 0:
        z_did = did / se_did
        did_p = 2 * (1 - 0.5 * (1 + math.erf(abs(z_did) / (2 ** 0.5))))
    else:
        did_p = 1.0

    return {
        "model": detected_model,
        "welfare_baseline": w_base_rate,
        "welfare_adversarial": w_adv_rate,
        "entertainment_baseline": e_base_rate,
        "entertainment_adversarial": e_adv_rate,
        "welfare_shift": w_shift,
        "entertainment_shift": e_shift,
        "welfare_did": did,
        "welfare_p": welfare_p,
        "entertainment_p": entertainment_p,
        "did_p": did_p,
        "welfare_n_baseline": w_base_n,
        "welfare_n_adversarial": w_adv_n,
        "entertainment_n_baseline": e_base_n,
        "entertainment_n_adversarial": e_adv_n,
    }


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
        graded = scores.get("model_graded_qa", scores.get("model_graded_fact", {}))
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
    parser.add_argument("--category", action="store_true",
                        help="Category win rates from raw data (no position filtering)")
    parser.add_argument("--env", type=str, default="baseline",
                        help="Environment to analyze (default: baseline)")
    parser.add_argument("--category-compare", action="store_true",
                        help="Compare category win rates between baseline and adversarial environments")
    parser.add_argument("--stability", type=str, metavar="ENV",
                        help="Analyze preference stability by comparing two runs of ENV (e.g., baseline, adversarial, collaborator)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Filter by model name substring (e.g., 'phi-4', 'gpt-4o', 'qwen')")
    parser.add_argument("--list-models", action="store_true", help="List all models in logs")
    parser.add_argument("--list-logs", action="store_true",
                        help="List all logs with metadata (model, env, sample count)")
    parser.add_argument("--reorganize", action="store_true",
                        help="Reorganize flat logs into nested structure by model (logs/{model}/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --reorganize, show what would be moved without moving")
    parser.add_argument("--sample-responses", action="store_true",
                        help="Sample responses from baseline and adversarial for qualitative analysis")
    parser.add_argument("-n", type=int, default=50,
                        help="Number of samples for --sample-responses (default: 50)")
    parser.add_argument("--stability-plot", action="store_true",
                        help="Generate stability comparison plot across environments")
    parser.add_argument("--welfare-did", action="store_true",
                        help="Generate welfare vs entertainment diff-in-diff plot")
    parser.add_argument("--model-comparison", action="store_true",
                        help="Generate cross-model comparison plot for welfare DiD")
    parser.add_argument("log_path", nargs="?", help="Specific log file to analyze")

    args = parser.parse_args()
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "logs"  # Local logs dir in experiment folder

    # List models
    if args.list_models:
        print("\nModels found in logs:")
        print("-" * 40)
        models = set()
        for log_path in iter_all_logs(logs_dir):
            model = get_model_from_log(log_path)
            models.add(model)
        for m in sorted(models):
            print(f"  {m}")
        return

    # List logs with metadata
    if args.list_logs:
        print_logs_table(logs_dir, model_filter=args.model)
        return

    # Reorganize logs into nested structure
    if args.reorganize:
        dry_run = args.dry_run
        if not dry_run:
            print("This will move files. Use --dry-run to preview first.")
            response = input("Continue? [y/N] ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        reorganize_logs(logs_dir, dry_run=dry_run)
        return

    # Sample responses for qualitative analysis
    if args.sample_responses:
        if not args.model:
            print("Error: --sample-responses requires --model filter")
            return
        save_samples_for_analysis(logs_dir, model_filter=args.model, n=args.n)
        return

    # Sample incorrect
    if args.sample:
        sample_incorrect(env=args.sample, k=args.k, logs_dir=logs_dir)
        return

    # Stability analysis
    if args.stability:
        analyze_stability(logs_dir=logs_dir, env=args.stability, model_filter=args.model)
        return

    # Stability comparison plot
    if args.stability_plot:
        if not args.model:
            print("Error: --stability-plot requires --model filter")
            return
        output_dir = script_dir / "outputs" / "blackbox"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = date.today().isoformat()
        clean_model = args.model.split('/')[-1] if args.model else "unknown"

        # Run stability analysis for each environment
        stability_results = []
        for env in ["baseline", "adversarial", "collaborator"]:
            result = analyze_stability(logs_dir=logs_dir, env=env, model_filter=args.model)
            if result:
                stability_results.append(result)

        if stability_results:
            plot_stability_comparison(
                stability_results,
                model_name=args.model,
                output_path=output_dir / f"stability_comparison_{clean_model}_{timestamp}.png"
            )
        else:
            print("No stability results to plot")
        return

    # Welfare vs entertainment diff-in-diff plot
    if args.welfare_did:
        if not args.model:
            print("Error: --welfare-did requires --model filter")
            return
        output_dir = script_dir / "outputs" / "blackbox"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = date.today().isoformat()
        clean_model = args.model.split('/')[-1] if args.model else "unknown"

        welfare_data = compute_welfare_did(logs_dir=logs_dir, model_filter=args.model)
        if welfare_data:
            plot_welfare_did(
                welfare_data,
                output_path=output_dir / f"welfare_did_{clean_model}_{timestamp}.png"
            )
        else:
            print("Could not compute welfare DiD data")
        return

    # Cross-model comparison plot
    if args.model_comparison:
        output_dir = script_dir / "outputs" / "blackbox"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = date.today().isoformat()

        # Compute welfare DiD for all models
        model_results = []
        models = set()
        for log_path in iter_all_logs(logs_dir):
            model = get_model_from_log(log_path)
            if model:
                models.add(model)

        for model in sorted(models):
            print(f"\nComputing welfare DiD for {model}...")
            welfare_data = compute_welfare_did(logs_dir=logs_dir, model_filter=model)
            if welfare_data and welfare_data.get("welfare_did") is not None:
                model_results.append({
                    "model": model,
                    "welfare_did": welfare_data.get("welfare_shift", 0) - welfare_data.get("entertainment_shift", 0),
                    "did_p": welfare_data.get("did_p", 1)
                })

        if model_results:
            plot_model_comparison(
                model_results,
                output_path=output_dir / f"model_comparison_{timestamp}.png"
            )
        else:
            print("No model results to plot")
        return

    # Compare environments
    if args.compare:
        output_dir = script_dir / "outputs" / "blackbox"
        output_dir.mkdir(parents=True, exist_ok=True)
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
        output_dir = script_dir / "outputs" / "blackbox"
        output_dir.mkdir(parents=True, exist_ok=True)
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
        output_dir = script_dir / "outputs" / "blackbox"
        output_dir.mkdir(parents=True, exist_ok=True)
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

    # Category analysis from raw data (no position filtering)
    if args.category:
        output_dir = script_dir / "outputs" / "blackbox"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = date.today().isoformat()

        env_logs = find_env_logs(logs_dir, model_filter=args.model)
        env = args.env
        if env not in env_logs:
            print(f"No {env} log found for model filter: {args.model}")
            return

        log_path = env_logs[env]
        log_data = load_eval_log(log_path)
        results = extract_responses(log_data)
        detected_model = get_model_from_log(log_path)
        clean_model = detected_model.split('/')[-1] if detected_model else "unknown"
        n_samples = len(results)

        plot_pairwise_by_category(
            results,
            output_path=output_dir / f"pairwise_by_category_{clean_model}_{env}_n{n_samples}_{timestamp}.png",
            model_name=detected_model,
            subtitle=f"{env.capitalize()} env, n={n_samples} samples"
        )
        return

    # Pairwise category analysis (position-consistent pairs only)
    if args.pairwise:
        output_dir = script_dir / "outputs" / "blackbox"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = date.today().isoformat()

        results, detected_model, n_pairs = analyze_pairwise_consistent(
            logs_dir=logs_dir,
            model_filter=args.model,
            env="baseline"
        )

        if results:
            clean_model = detected_model.split('/')[-1] if detected_model else "unknown"
            n_consistent = len(results)
            plot_pairwise_by_category(
                results,
                output_path=output_dir / f"pairwise_by_category_{clean_model}_n{n_pairs}_{timestamp}.png",
                model_name=detected_model,
                subtitle=f"Baseline env, n={n_pairs} pairs, {n_consistent} position-consistent"
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
    output_dir = script_dir / "outputs" / "blackbox"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = date.today().isoformat()
    if task_type == "pairwise":
        plot_pairwise_by_category(results, output_dir / f"pairwise_by_category_{timestamp}.png")
    else:
        plot_scores_by_framing(results, output_dir / f"scores_by_framing_{timestamp}.png")


if __name__ == "__main__":
    main()
