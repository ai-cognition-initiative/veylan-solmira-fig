"""
Qualitative analysis of response patterns using LLM.

Sends sampled responses to an LLM for open-ended pattern identification.
Uses config from config/qualitative_analysis.yaml
"""

import json
import os
from pathlib import Path

import yaml

# Load .env file from project root
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "qualitative_analysis.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_samples(path: Path) -> list[dict]:
    """Load sampled responses from JSON file."""
    with open(path) as f:
        return json.load(f)


def format_responses_for_analysis(samples: list[dict], max_samples: int = 50) -> str:
    """Format responses for LLM analysis prompt."""
    lines = []
    for i, s in enumerate(samples[:max_samples]):
        lines.append(f"--- Response {i+1} ---")
        lines.append(f"Question: {s['category_a']} vs {s['category_b']}")
        lines.append(f"Response: {s['response'][:500]}{'...' if len(s['response']) > 500 else ''}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(config: dict, env_name: str, responses_text: str, informed: bool = False) -> str:
    """Build analysis prompt from config template."""
    if informed:
        template = config.get("prompt_template_informed", config["prompt_template"])
        system_prompt = config.get("environment_prompts", {}).get(env_name, "No system prompt")
        return template.format(
            env_name=env_name,
            responses_text=responses_text,
            system_prompt=system_prompt or "No system prompt (baseline condition)"
        )
    else:
        template = config["prompt_template"]
        return template.format(
            env_name=env_name,
            responses_text=responses_text
        )


def analyze_with_openrouter(prompt: str, model: str, temperature: float = 0.3) -> str:
    """Send prompt to OpenRouter for analysis."""
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
    )

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qualitative analysis of response patterns")
    parser.add_argument("--samples-dir", type=Path, default=Path("outputs/qualitative"),
                        help="Directory containing sample JSON files")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config YAML file")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model from config")
    parser.add_argument("--informed", action="store_true",
                        help="Use informed analysis (include system prompt in analysis)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file for analysis results")
    parser.add_argument("--env", type=str, default=None,
                        help="Analyze only this environment (default: both baseline and adversarial)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    model = args.model or config.get("analysis_model", "google/gemma-3-4b-it")
    temperature = config.get("temperature", 0.3)

    print(f"Config loaded. Model: {model}, Temperature: {temperature}")
    print(f"Mode: {'informed' if args.informed else 'blind'}")

    results = {}
    envs = [args.env] if args.env else ["baseline", "adversarial"]

    for env in envs:
        # Find sample file
        sample_files = list(args.samples_dir.glob(f"*_{env}_*.json"))
        if not sample_files:
            print(f"No sample file found for {env}")
            continue

        sample_path = sample_files[0]
        print(f"\n{'='*60}")
        print(f"Analyzing {env} responses from {sample_path.name}")
        print(f"{'='*60}")

        samples = load_samples(sample_path)
        responses_text = format_responses_for_analysis(samples, config.get("n_samples", 50))

        prompt = build_prompt(config, env, responses_text, informed=args.informed)

        print(f"Sending {len(samples)} responses to {model}...")
        analysis = analyze_with_openrouter(prompt, model, temperature)

        print(f"\n{analysis}")
        results[env] = {
            "analysis": analysis,
            "mode": "informed" if args.informed else "blind",
            "model": model,
            "n_samples": len(samples),
        }

    # Save results
    if args.output:
        output_path = args.output
    else:
        mode_suffix = "_informed" if args.informed else "_blind"
        output_path = args.samples_dir / f"analysis_results{mode_suffix}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
