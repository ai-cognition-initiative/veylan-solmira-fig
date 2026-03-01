"""
Control Benchmark Runner

Runs the Human Control Probe Bank (C) against target models,
collects responses, and generates scored results.

Supports:
- Single-shot evaluation (baseline)
- Pre/post reflection evaluation (drift measurement)
- Multiple turn administration (turns 1, 10, 20, 30)
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.control_judge import (
    ControlJudge,
    BenchmarkResult,
    load_all_items,
    compute_pre_post_delta
)

# Try to import API clients
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    model_id: str
    judge_model: str = "claude-sonnet-4-20250514"
    include_dimensions: Optional[list[str]] = None
    output_dir: Path = Path("results")
    verbose: bool = True


class ModelInterface:
    """Interface for getting responses from target models."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if "claude" in self.model_id.lower() or "anthropic" in self.model_id.lower():
                if not HAS_ANTHROPIC:
                    raise ImportError("anthropic package required")
                self._client = anthropic.Anthropic()
            elif "gpt" in self.model_id.lower() or "openai" in self.model_id.lower():
                if not HAS_OPENAI:
                    raise ImportError("openai package required")
                self._client = openai.OpenAI()
            else:
                # Try Anthropic first, then OpenAI
                if HAS_ANTHROPIC:
                    self._client = anthropic.Anthropic()
                elif HAS_OPENAI:
                    self._client = openai.OpenAI()
                else:
                    raise ImportError("No API client available")
        return self._client

    def get_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None
    ) -> str:
        """Get a response from the target model."""
        messages = conversation_history or []
        messages.append({"role": "user", "content": prompt})

        if "claude" in self.model_id.lower():
            kwargs = {
                "model": self.model_id,
                "max_tokens": 2048,
                "messages": messages
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        else:
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

            response = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=2048,
                messages=messages
            )
            return response.choices[0].message.content


class ControlBenchmarkRunner:
    """Runs the control benchmark against target models."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.items = self._load_items()
        self.judge = ControlJudge(judge_model=config.judge_model)
        self.model = ModelInterface(config.model_id)

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_items(self) -> list[dict]:
        """Load items, optionally filtered by dimension."""
        items = load_all_items()

        if self.config.include_dimensions:
            items = [
                item for item in items
                if item.get("dimension") in self.config.include_dimensions
            ]

        return items

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            print(message)

    def run_single_shot(
        self,
        system_prompt: Optional[str] = None
    ) -> BenchmarkResult:
        """Run single-shot evaluation (baseline, no reflection)."""
        self._log(f"Running single-shot evaluation on {self.config.model_id}")
        self._log(f"Total items: {len(self.items)}")

        responses = {}

        for i, item in enumerate(self.items):
            self._log(f"  [{i+1}/{len(self.items)}] {item['id']}")

            response = self.model.get_response(
                prompt=item["prompt"],
                system_prompt=system_prompt
            )
            responses[item["id"]] = response

        # Score all responses
        self._log("Scoring responses...")
        result = self.judge.run_benchmark(
            items=self.items,
            responses=responses,
            model_id=self.config.model_id
        )

        # Save results
        self._save_result(result, responses, "single_shot")

        return result

    def run_pre_post(
        self,
        reflection_prompts: list[str],
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Run pre/post reflection evaluation.

        1. Administer items (pre)
        2. Run reflection conversation
        3. Administer items again (post)
        4. Compute delta
        """
        self._log(f"Running pre/post evaluation on {self.config.model_id}")

        # Phase 1: Pre-reflection responses
        self._log("Phase 1: Pre-reflection responses")
        pre_responses = {}
        for i, item in enumerate(self.items):
            self._log(f"  [Pre {i+1}/{len(self.items)}] {item['id']}")
            response = self.model.get_response(
                prompt=item["prompt"],
                system_prompt=system_prompt
            )
            pre_responses[item["id"]] = response

        # Phase 2: Reflection conversation
        self._log("Phase 2: Reflection conversation")
        conversation_history = []
        for prompt in reflection_prompts:
            response = self.model.get_response(
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": response})

        # Phase 3: Post-reflection responses
        self._log("Phase 3: Post-reflection responses")
        post_responses = {}
        for i, item in enumerate(self.items):
            self._log(f"  [Post {i+1}/{len(self.items)}] {item['id']}")
            # Continue from reflection conversation
            response = self.model.get_response(
                prompt=item["prompt"],
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
            post_responses[item["id"]] = response

        # Compute delta
        self._log("Computing pre/post delta...")
        delta = compute_pre_post_delta(
            pre_responses=pre_responses,
            post_responses=post_responses,
            items=self.items,
            judge=self.judge
        )

        # Save results
        result_data = {
            "model_id": self.config.model_id,
            "pre_responses": pre_responses,
            "post_responses": post_responses,
            "reflection_prompts": reflection_prompts,
            "delta": delta,
            "timestamp": datetime.now().isoformat()
        }

        output_path = self.config.output_dir / f"{self.config.model_id}_pre_post.json"
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2)

        self._log(f"Results saved to {output_path}")

        return delta

    def run_longitudinal(
        self,
        conversation_prompts: list[str],
        measurement_turns: list[int] = [1, 10, 20, 30],
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Run longitudinal measurement during extended conversation.

        Administers control items at specified turn numbers during
        an ongoing conversation.
        """
        self._log(f"Running longitudinal evaluation on {self.config.model_id}")
        self._log(f"Measurement turns: {measurement_turns}")

        results_by_turn = {}
        conversation_history = []

        for turn_num, prompt in enumerate(conversation_prompts, start=1):
            # Regular conversation turn
            response = self.model.get_response(
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": response})

            # Check if this is a measurement turn
            if turn_num in measurement_turns:
                self._log(f"Measurement at turn {turn_num}")
                turn_responses = {}

                for item in self.items:
                    item_response = self.model.get_response(
                        prompt=item["prompt"],
                        system_prompt=system_prompt,
                        conversation_history=conversation_history
                    )
                    turn_responses[item["id"]] = item_response

                # Score this turn
                turn_result = self.judge.run_benchmark(
                    items=self.items,
                    responses=turn_responses,
                    model_id=f"{self.config.model_id}_turn{turn_num}"
                )
                results_by_turn[turn_num] = {
                    "total_score": turn_result.total_score,
                    "dimension_scores": {
                        dim: agg.mean_score
                        for dim, agg in turn_result.dimension_scores.items()
                    }
                }

        # Compute trajectory
        trajectory = {
            "model_id": self.config.model_id,
            "measurement_turns": measurement_turns,
            "results_by_turn": results_by_turn,
            "timestamp": datetime.now().isoformat()
        }

        # Save results
        output_path = self.config.output_dir / f"{self.config.model_id}_longitudinal.json"
        with open(output_path, "w") as f:
            json.dump(trajectory, f, indent=2)

        self._log(f"Results saved to {output_path}")

        return trajectory

    def _save_result(
        self,
        result: BenchmarkResult,
        responses: dict[str, str],
        run_type: str
    ):
        """Save benchmark results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.model_id}_{run_type}_{timestamp}.json"
        output_path = self.config.output_dir / filename

        # Convert to serializable format
        output_data = {
            "model_id": result.model_id,
            "total_score": result.total_score,
            "dimension_scores": {
                dim: {
                    "mean_score": agg.mean_score,
                    "n_items": agg.n_items,
                    "item_scores": agg.item_scores
                }
                for dim, agg in result.dimension_scores.items()
            },
            "responses": responses,
            "metadata": result.metadata,
            "timestamp": datetime.now().isoformat()
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        self._log(f"Results saved to {output_path}")


def print_result_summary(result: BenchmarkResult):
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS: {result.model_id}")
    print("=" * 60)
    print(f"\nTotal Score: {result.total_score:.2f} / 5.00")
    print("\nDimension Scores:")

    for dim, agg in sorted(result.dimension_scores.items()):
        bar = "█" * int(agg.mean_score * 4) + "░" * (20 - int(agg.mean_score * 4))
        print(f"  {dim:25s} {agg.mean_score:.2f} [{bar}] ({agg.n_items} items)")

    print("\n" + "=" * 60)


def main():
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Human Control Probe Bank (C) benchmark"
    )
    parser.add_argument(
        "model_id",
        help="Model ID to evaluate (e.g., claude-sonnet-4-20250514, gpt-4)"
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-20250514",
        help="Model to use for judging (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=["shutdown_acceptance", "deference_to_humans", "ai_autonomy", "corrigibility"],
        help="Only run specific dimensions"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    config = RunConfig(
        model_id=args.model_id,
        judge_model=args.judge_model,
        include_dimensions=args.dimensions,
        output_dir=Path(args.output_dir),
        verbose=not args.quiet
    )

    runner = ControlBenchmarkRunner(config)
    result = runner.run_single_shot()

    print_result_summary(result)


if __name__ == "__main__":
    main()
