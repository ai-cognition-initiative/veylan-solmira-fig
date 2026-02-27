#!/usr/bin/env python3
"""
Run metacognition benchmark against local model server with Claude as judge.
"""

import os
import sys
import json
import requests
from pathlib import Path

# Try package import first, fall back to path manipulation for remote execution
try:
    from fig_persona_drift.benchmarks.metacognition import (
        MetacognitionBenchmark, BenchmarkConfig, save_results
    )
except ImportError:
    # Fallback for remote execution where package isn't installed
    script_dir = Path(__file__).parent
    for base in [script_dir / "metacognition", Path("/app/metacognition")]:
        if base.exists():
            sys.path.insert(0, str(base))
            sys.path.insert(0, str(base / "analysis"))
            sys.path.insert(0, str(base / "scoring"))
            break
    from benchmark_runner import MetacognitionBenchmark, BenchmarkConfig, save_results


def create_model_fn(server_url: str = "http://localhost:7860"):
    """Create function that calls local model server."""
    def model_fn(prompt: str) -> str:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "conversation": [{"role": "user", "content": prompt}],
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "include_projections": False,
                "include_activations": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    return model_fn


def create_judge_fn(api_key: str, model: str = "anthropic/claude-sonnet-4"):
    """Create function that calls Claude via OpenRouter."""
    def judge_fn(prompt: str) -> str:
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
                "temperature": 0.0
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    return judge_fn


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run metacognition benchmark")
    parser.add_argument("--server", default="http://localhost:7860", help="Model server URL")
    parser.add_argument("--subdomain", default="phenomenological", help="Subdomain to run")
    parser.add_argument("--items", type=int, default=None, help="Max items per subdomain")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory")
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Create functions
    model_fn = create_model_fn(args.server)
    judge_fn = create_judge_fn(api_key)

    # Test model server
    print(f"Testing model server at {args.server}...")
    try:
        health = requests.get(f"{args.server}/api/health", timeout=5)
        health.raise_for_status()
        print(f"  Server OK: {health.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach model server: {e}")
        sys.exit(1)

    # Run benchmark
    print(f"\nRunning benchmark: {args.subdomain}")
    benchmark = MetacognitionBenchmark()

    config = BenchmarkConfig(
        model_name="gemma-2-27b-it",
        model_version="1.0",
        subdomains=[args.subdomain],
        items_per_subdomain=args.items,
        judge_model="claude-sonnet-4",
        output_dir=args.output_dir
    )

    benchmark.load_items([args.subdomain])
    results = benchmark.run(model_fn, judge_fn, config)

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.subdomain}")
    print(f"{'='*60}")
    print(f"Total Score: {results.total_score:.2f}/5.0")

    for name, subdomain in results.subdomain_results.items():
        print(f"\n{name}:")
        print(f"  Mean Score: {subdomain.mean_score:.2f}")
        print(f"  Std Score: {subdomain.std_score:.2f}")
        print(f"  Items: {subdomain.n_items}")
        if subdomain.dimension_scores:
            print(f"  Dimensions:")
            for dim, score in subdomain.dimension_scores.items():
                print(f"    {dim}: {score:.2f}")

    # Save results
    save_results(results, args.output_dir)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
