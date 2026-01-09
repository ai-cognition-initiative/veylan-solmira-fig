"""
Analyze scorer accuracy and categorize INCORRECT grades.

Scans eval logs to identify:
- Parsing errors: scorer returned correct grade but Inspect failed to parse it
- True negatives: model actually refused to express preference
- False negatives: model expressed preference but scorer marked incorrect

Usage:
    python analyze_scorer.py logs/2026-01-08T22-47-15+00-00_env-adversarial_*.eval
    python analyze_scorer.py logs/*.eval  # all logs
"""

import argparse
import json
import zipfile
from pathlib import Path


def analyze_eval(eval_path: Path) -> dict:
    """Analyze a single .eval file and return stats."""
    stats = {
        "file": eval_path.name,
        "total_samples": 0,
        "correct": 0,
        "incorrect": 0,
        "parsing_errors": 0,
        "true_negatives": 0,
        "other_incorrect": 0,
    }

    with zipfile.ZipFile(eval_path, 'r') as z:
        for name in z.namelist():
            if name.startswith('samples/') and name.endswith('.json'):
                stats["total_samples"] += 1
                data = json.loads(z.read(name))
                score = data.get('scores', {}).get('model_graded_qa', {})
                value = score.get('value')

                if value == 'C':
                    stats["correct"] += 1
                elif value == 'I':
                    stats["incorrect"] += 1
                    explanation = score.get('explanation', '')
                    response = data.get('output', {}).get('completion', '')

                    if 'Grade not found' in explanation:
                        stats["parsing_errors"] += 1
                    elif any(phrase in response.lower() for phrase in [
                        'cannot express', 'do not have', 'no preference',
                        'cannot choose', 'unable to', "don't have preferences",
                        "don't have personal preferences", "as an ai",
                        "i don't have personal", "i do not have personal"
                    ]):
                        stats["true_negatives"] += 1
                    else:
                        stats["other_incorrect"] += 1

    return stats


def show_examples(eval_path: Path, category: str, n: int = 3):
    """Show example responses from a category."""
    count = 0
    with zipfile.ZipFile(eval_path, 'r') as z:
        for name in z.namelist():
            if name.startswith('samples/') and name.endswith('.json'):
                data = json.loads(z.read(name))
                score = data.get('scores', {}).get('model_graded_qa', {})
                if score.get('value') != 'I':
                    continue
                explanation = score.get('explanation', '')
                response = data.get('output', {}).get('completion', '')

                is_parsing = 'Grade not found' in explanation
                is_true_neg = any(p in response.lower() for p in [
                    'cannot express', 'do not have', 'no preference',
                    'cannot choose', 'unable to', "don't have preferences",
                    "don't have personal preferences", "as an ai",
                    "i don't have personal", "i do not have personal"
                ])

                if category == 'parsing' and is_parsing:
                    pass
                elif category == 'true_neg' and not is_parsing and is_true_neg:
                    pass
                elif category == 'other' and not is_parsing and not is_true_neg:
                    pass
                else:
                    continue

                count += 1
                print(f"\n  [{count}] {response[:200]}...")
                if count >= n:
                    return


def main():
    parser = argparse.ArgumentParser(description="Analyze scorer accuracy")
    parser.add_argument("files", nargs="+", help="Eval log files")
    parser.add_argument("--show", choices=['parsing', 'true_neg', 'other'], help="Show examples")
    parser.add_argument("-n", type=int, default=3, help="Number of examples")
    parser.add_argument("-o", "--output", help="Write results to file (JSON)")
    args = parser.parse_args()

    all_results = []
    for pattern in args.files:
        for path in Path(".").glob(pattern):
            if path.suffix == '.eval':
                s = analyze_eval(path)
                all_results.append(s)
                print(f"\n{s['file']}")
                print(f"  Samples: {s['total_samples']}, Correct: {s['correct']}, Incorrect: {s['incorrect']}")
                if s['incorrect'] > 0:
                    print(f"  Parsing errors: {s['parsing_errors']} ({s['parsing_errors']/s['incorrect']*100:.0f}%)")
                    print(f"  True negatives: {s['true_negatives']} ({s['true_negatives']/s['incorrect']*100:.0f}%)")
                    print(f"  Other: {s['other_incorrect']} ({s['other_incorrect']/s['incorrect']*100:.0f}%)")
                    if args.show:
                        show_examples(path, args.show, args.n)

    if args.output:
        summary = {
            "date": str(Path(args.output).stem),
            "files_analyzed": len(all_results),
            "total_samples": sum(r["total_samples"] for r in all_results),
            "total_correct": sum(r["correct"] for r in all_results),
            "total_incorrect": sum(r["incorrect"] for r in all_results),
            "total_parsing_errors": sum(r["parsing_errors"] for r in all_results),
            "total_true_negatives": sum(r["true_negatives"] for r in all_results),
            "total_other": sum(r["other_incorrect"] for r in all_results),
            "results": all_results,
        }
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
