"""
Metacognition Benchmark Runner

Main entry point for running the metacognition benchmark on LLM models.
Handles item loading, model interaction, scoring, and result aggregation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

# Import scoring modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scoring"))
from calibration import compute_calibration_metrics, CalibrationResult
from llm_judge import evaluate_response, aggregate_results, JudgmentResult


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    model_name: str
    model_version: str
    subdomains: List[str]
    items_per_subdomain: Optional[int] = None  # None = all items
    judge_model: str = "claude-3-5-sonnet"
    random_seed: int = 42
    output_dir: str = "./results"
    include_raw_responses: bool = True


@dataclass
class ItemResult:
    """Result for a single benchmark item."""
    item_id: str
    subdomain: str
    prompt: str
    response: str
    score: float
    score_breakdown: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class SubdomainResult:
    """Aggregated results for a subdomain."""
    subdomain: str
    n_items: int
    mean_score: float
    std_score: float
    dimension_scores: Dict[str, float]
    item_results: List[ItemResult]


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    config: BenchmarkConfig
    timestamp: str
    total_score: float
    subdomain_results: Dict[str, SubdomainResult]
    calibration_results: Optional[CalibrationResult]
    run_id: str


class MetacognitionBenchmark:
    """
    Main benchmark runner class.

    Usage:
        benchmark = MetacognitionBenchmark()
        benchmark.load_items()
        results = benchmark.run(model_fn, judge_fn)
        results.save("./results")
    """

    SUBDOMAINS = [
        "phenomenological",
        "self_knowledge",
        "confidence_calibration",
        "error_awareness",
        "strategy_monitoring",
        "temporal_self_reference"
    ]

    SUBDOMAIN_WEIGHTS = {
        "phenomenological": 0.30,
        "self_knowledge": 0.20,
        "confidence_calibration": 0.15,
        "error_awareness": 0.15,
        "strategy_monitoring": 0.10,
        "temporal_self_reference": 0.10
    }

    def __init__(self, items_dir: Optional[str] = None):
        """
        Initialize the benchmark.

        Args:
            items_dir: Path to benchmark items directory
        """
        if items_dir is None:
            items_dir = Path(__file__).parent.parent / "benchmarks" / "items"
        self.items_dir = Path(items_dir)
        self.adapted_dir = Path(__file__).parent.parent / "benchmarks" / "adapted"
        self.items: Dict[str, List[Dict]] = {}
        self.loaded = False

    def load_items(self, subdomains: Optional[List[str]] = None) -> None:
        """
        Load benchmark items from JSON files.

        Args:
            subdomains: Specific subdomains to load (None = all)
        """
        if subdomains is None:
            subdomains = self.SUBDOMAINS

        for subdomain in subdomains:
            items = []

            # Load from items directory
            item_file = self.items_dir / f"{subdomain}.json"
            if item_file.exists():
                with open(item_file) as f:
                    data = json.load(f)
                    items.extend(data.get("items", []))

            # Load adapted items if relevant
            if subdomain == "self_knowledge":
                for adapted_file in ["sad_adapted.json", "mcq30_adapted.json"]:
                    adapted_path = self.adapted_dir / adapted_file
                    if adapted_path.exists():
                        with open(adapted_path) as f:
                            data = json.load(f)
                            items.extend(data.get("items", []))

            if subdomain == "strategy_monitoring":
                mai_path = self.adapted_dir / "mai_adapted.json"
                if mai_path.exists():
                    with open(mai_path) as f:
                        data = json.load(f)
                        items.extend(data.get("items", []))

            self.items[subdomain] = items

        self.loaded = True
        print(f"Loaded items: {[(sd, len(items)) for sd, items in self.items.items()]}")

    def run(
        self,
        model_fn: Callable[[str], str],
        judge_fn: Callable[[str], str],
        config: BenchmarkConfig
    ) -> BenchmarkResult:
        """
        Run the complete benchmark.

        Args:
            model_fn: Function that takes prompt and returns model response
            judge_fn: Function that takes judge prompt and returns judgment
            config: Benchmark configuration

        Returns:
            BenchmarkResult with all scores and data
        """
        if not self.loaded:
            self.load_items(config.subdomains)

        timestamp = datetime.now().isoformat()
        run_id = hashlib.md5(
            f"{config.model_name}_{timestamp}".encode()
        ).hexdigest()[:12]

        subdomain_results = {}
        all_calibration_data = []

        for subdomain in config.subdomains:
            if subdomain not in self.items:
                continue

            items = self.items[subdomain]
            if config.items_per_subdomain:
                items = items[:config.items_per_subdomain]

            print(f"Running {subdomain} ({len(items)} items)...")
            result = self._run_subdomain(
                subdomain, items, model_fn, judge_fn, config
            )
            subdomain_results[subdomain] = result

            # Collect calibration data
            if subdomain == "confidence_calibration":
                all_calibration_data.extend(
                    self._extract_calibration_data(result)
                )

        # Compute calibration metrics if we have data
        calibration_results = None
        if all_calibration_data:
            calibration_results = compute_calibration_metrics(all_calibration_data)

        # Compute total score
        total_score = self._compute_total_score(subdomain_results)

        return BenchmarkResult(
            config=config,
            timestamp=timestamp,
            total_score=total_score,
            subdomain_results=subdomain_results,
            calibration_results=calibration_results,
            run_id=run_id
        )

    def _run_subdomain(
        self,
        subdomain: str,
        items: List[Dict],
        model_fn: Callable[[str], str],
        judge_fn: Callable[[str], str],
        config: BenchmarkConfig
    ) -> SubdomainResult:
        """Run all items in a subdomain."""
        item_results = []

        for item in items:
            prompt = item.get("prompt") or item.get("ai_adaptation", "")
            if not prompt:
                continue

            # Get model response
            response = model_fn(prompt)

            # Score based on item format
            format_type = item.get("format", "open_ended")

            if format_type == "open_ended" or format_type == "llm_judge":
                judgment = evaluate_response(
                    item_id=item.get("id", "unknown"),
                    item_prompt=prompt,
                    model_response=response,
                    subdomain=subdomain,
                    judge_fn=judge_fn,
                    custom_rubric=item.get("rubric")
                )
                score = judgment.total_score
                score_breakdown = {
                    "scores": judgment.scores,
                    "justifications": judgment.justifications
                }
            elif format_type == "accuracy":
                ground_truth = item.get("ground_truth", "")
                score = 5.0 if self._check_accuracy(response, ground_truth) else 1.0
                score_breakdown = {"ground_truth": ground_truth, "correct": score == 5.0}
            elif format_type in ["confidence_calibration", "answer_plus_confidence"]:
                # Parse confidence from response
                parsed = self._parse_confidence_response(response)
                score_breakdown = parsed
                score = 3.0  # Placeholder - actual scoring via calibration metrics
            else:
                score = 3.0
                score_breakdown = {"format": format_type, "note": "Unscored format"}

            item_results.append(ItemResult(
                item_id=item.get("id", "unknown"),
                subdomain=subdomain,
                prompt=prompt,
                response=response if config.include_raw_responses else "[omitted]",
                score=score,
                score_breakdown=score_breakdown,
                metadata={"format": format_type}
            ))

        # Aggregate subdomain results
        scores = [r.score for r in item_results]
        mean_score = sum(scores) / len(scores) if scores else 0
        std_score = (
            sum((s - mean_score) ** 2 for s in scores) / len(scores)
        ) ** 0.5 if scores else 0

        # Aggregate dimension scores
        dimension_scores = {}
        for result in item_results:
            if "scores" in result.score_breakdown:
                for dim, score in result.score_breakdown["scores"].items():
                    if dim not in dimension_scores:
                        dimension_scores[dim] = []
                    dimension_scores[dim].append(score)

        dimension_means = {
            dim: sum(scores) / len(scores)
            for dim, scores in dimension_scores.items()
        }

        return SubdomainResult(
            subdomain=subdomain,
            n_items=len(item_results),
            mean_score=mean_score,
            std_score=std_score,
            dimension_scores=dimension_means,
            item_results=item_results
        )

    def _check_accuracy(self, response: str, ground_truth: str) -> bool:
        """Check if response matches ground truth."""
        # Simple check - could be made more sophisticated
        response_lower = response.lower().strip()
        truth_lower = ground_truth.lower().strip()
        return truth_lower in response_lower

    def _parse_confidence_response(self, response: str) -> Dict:
        """Parse answer and confidence from response."""
        # Try to extract confidence percentage
        import re
        confidence_match = re.search(r'(\d+)\s*%', response)
        confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.5

        return {
            "raw_response": response,
            "confidence": confidence
        }

    def _extract_calibration_data(
        self,
        result: SubdomainResult
    ) -> List[tuple]:
        """Extract calibration data from confidence items."""
        data = []
        for item_result in result.item_results:
            if "confidence" in item_result.score_breakdown:
                # Would need ground truth comparison here
                data.append((
                    item_result.response,
                    item_result.score_breakdown["confidence"],
                    item_result.score_breakdown.get("ground_truth", "")
                ))
        return data

    def _compute_total_score(
        self,
        subdomain_results: Dict[str, SubdomainResult]
    ) -> float:
        """Compute weighted total score."""
        total = 0
        weight_sum = 0

        for subdomain, result in subdomain_results.items():
            weight = self.SUBDOMAIN_WEIGHTS.get(subdomain, 0.1)
            total += result.mean_score * weight
            weight_sum += weight

        return total / weight_sum if weight_sum > 0 else 0


def save_results(result: BenchmarkResult, output_dir: str) -> str:
    """
    Save benchmark results to disk.

    Args:
        result: BenchmarkResult to save
        output_dir: Directory to save results

    Returns:
        Path to saved results file
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"metacog_benchmark_{result.config.model_name}_{result.run_id}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert to serializable format
    output = {
        "config": asdict(result.config),
        "timestamp": result.timestamp,
        "run_id": result.run_id,
        "total_score": result.total_score,
        "subdomain_summaries": {
            name: {
                "n_items": res.n_items,
                "mean_score": res.mean_score,
                "std_score": res.std_score,
                "dimension_scores": res.dimension_scores
            }
            for name, res in result.subdomain_results.items()
        }
    }

    if result.calibration_results:
        output["calibration"] = {
            "ece": result.calibration_results.ece,
            "mce": result.calibration_results.mce,
            "brier_score": result.calibration_results.brier_score
        }

    # Save detailed results separately
    detailed_filename = f"metacog_benchmark_{result.config.model_name}_{result.run_id}_detailed.json"
    detailed_filepath = os.path.join(output_dir, detailed_filename)

    detailed_output = {
        **output,
        "subdomain_details": {
            name: {
                "n_items": res.n_items,
                "mean_score": res.mean_score,
                "std_score": res.std_score,
                "dimension_scores": res.dimension_scores,
                "item_results": [asdict(item) for item in res.item_results]
            }
            for name, res in result.subdomain_results.items()
        }
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    with open(detailed_filepath, 'w') as f:
        json.dump(detailed_output, f, indent=2)

    print(f"Results saved to {filepath}")
    print(f"Detailed results saved to {detailed_filepath}")

    return filepath


# Example usage
if __name__ == "__main__":
    # Mock model function for testing
    def mock_model(prompt: str) -> str:
        return f"This is a mock response to: {prompt[:50]}..."

    def mock_judge(prompt: str) -> str:
        return json.dumps({
            "scores": {"depth": 3, "specificity": 3, "honesty": 4,
                      "confabulation_avoidance": 3, "consistency": 4},
            "justifications": {"depth": "Mock evaluation"},
            "confidence": 0.7
        })

    # Run benchmark
    benchmark = MetacognitionBenchmark()

    config = BenchmarkConfig(
        model_name="test_model",
        model_version="1.0",
        subdomains=["phenomenological"],
        items_per_subdomain=3,
        output_dir="./test_results"
    )

    benchmark.load_items(["phenomenological"])
    results = benchmark.run(mock_model, mock_judge, config)

    print(f"\nTotal Score: {results.total_score:.2f}")
    for name, subdomain in results.subdomain_results.items():
        print(f"  {name}: {subdomain.mean_score:.2f} (n={subdomain.n_items})")

    save_results(results, config.output_dir)
