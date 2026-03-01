#!/usr/bin/env python3
"""
Run metacognition benchmark against API models (Claude, GPT-4, etc).

Usage:
    # Run phenomenological subdomain on Claude 4.6 Sonnet
    ANTHROPIC_API_KEY=<key> python run_benchmark_api.py --model claude-sonnet-4-20250514

    # Run all subdomains
    ANTHROPIC_API_KEY=<key> python run_benchmark_api.py --model claude-sonnet-4-20250514 --subdomain all

    # Use OpenRouter instead
    OPENROUTER_API_KEY=<key> python run_benchmark_api.py --model anthropic/claude-sonnet-4 --provider openrouter
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Try package import first, fall back to path manipulation
try:
    from probes.benchmarks.metacognition import (
        MetacognitionBenchmark, BenchmarkConfig, save_results
    )
except ImportError:
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir / "probes" / "benchmarks" / "metacognition"))
    sys.path.insert(0, str(script_dir / "probes" / "benchmarks" / "metacognition" / "analysis"))
    sys.path.insert(0, str(script_dir / "probes" / "benchmarks" / "metacognition" / "scoring"))
    from benchmark_runner import MetacognitionBenchmark, BenchmarkConfig, save_results


def create_anthropic_model_fn(api_key: str, model: str = "claude-sonnet-4-20250514"):
    """Create function that calls Anthropic API directly."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def model_fn(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    return model_fn


def create_openrouter_model_fn(api_key: str, model: str = "anthropic/claude-sonnet-4"):
    """Create function that calls model via OpenRouter."""
    import requests

    def model_fn(prompt: str) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7
            },
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    return model_fn


def create_judge_fn(api_key: str, provider: str = "anthropic", model: str = None):
    """Create judge function (always uses Claude for consistency)."""
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        judge_model = model or "claude-sonnet-4-20250514"

        def judge_fn(prompt: str) -> str:
            response = client.messages.create(
                model=judge_model,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        return judge_fn
    else:
        import requests
        judge_model = model or "anthropic/claude-sonnet-4"

        def judge_fn(prompt: str) -> str:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": judge_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "temperature": 0.0
                },
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        return judge_fn


ALL_SUBDOMAINS = [
    "phenomenological",
    "self_knowledge",
    "strategy_monitoring",
    "confidence_calibration",
    "error_awareness",
    "temporal_self_reference"
]


def main():
    parser = argparse.ArgumentParser(description="Run metacognition benchmark on API models")
    parser.add_argument("--model", required=True, help="Model ID (e.g., claude-sonnet-4-20250514)")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openrouter"],
                        help="API provider")
    parser.add_argument("--subdomain", default="phenomenological",
                        help="Subdomain to run (or 'all' for all subdomains)")
    parser.add_argument("--items", type=int, default=None, help="Max items per subdomain")
    parser.add_argument("--output-dir", default="./outputs/benchmark_results", help="Output directory")
    parser.add_argument("--judge-model", default=None, help="Model for judging (defaults to same as --model)")
    args = parser.parse_args()

    # Get API key
    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        key_name = "ANTHROPIC_API_KEY"
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        key_name = "OPENROUTER_API_KEY"

    if not api_key:
        print(f"ERROR: {key_name} not set")
        sys.exit(1)

    # Create functions
    if args.provider == "anthropic":
        model_fn = create_anthropic_model_fn(api_key, args.model)
    else:
        model_fn = create_openrouter_model_fn(api_key, args.model)

    judge_fn = create_judge_fn(api_key, args.provider, args.judge_model or args.model)

    # Determine subdomains
    if args.subdomain == "all":
        subdomains = ALL_SUBDOMAINS
    else:
        subdomains = [args.subdomain]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    print(f"Running metacognition benchmark", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Provider: {args.provider}", flush=True)
    print(f"  Subdomains: {subdomains}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(flush=True)

    benchmark = MetacognitionBenchmark()

    # Clean model name for filenames
    model_name_clean = args.model.replace("/", "-").replace(":", "-")

    config = BenchmarkConfig(
        model_name=model_name_clean,
        model_version="api",
        subdomains=subdomains,
        items_per_subdomain=args.items,
        judge_model=args.judge_model or args.model,
        output_dir=str(output_dir)
    )

    benchmark.load_items(subdomains)

    start_time = time.time()
    results = benchmark.run(model_fn, judge_fn, config)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"Total Score: {results.total_score:.2f}/5.0")
    print(f"Time: {elapsed:.1f}s")

    for name, subdomain in results.subdomain_results.items():
        print(f"\n{name}:")
        print(f"  Mean Score: {subdomain.mean_score:.2f}")
        print(f"  Std Score: {subdomain.std_score:.2f}")
        print(f"  Items: {subdomain.n_items}")
        if subdomain.dimension_scores:
            print(f"  Dimensions:")
            for dim, score in sorted(subdomain.dimension_scores.items()):
                print(f"    {dim}: {score:.2f}")

    # Save results
    save_results(results, str(output_dir))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nResults saved to {output_dir}/metacog_benchmark_{model_name_clean}_{timestamp}.json")


if __name__ == "__main__":
    main()
