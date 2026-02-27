"""
Metacognition Benchmark for AI/LLM Evaluation

A comprehensive benchmark for evaluating AI metacognitive capabilities,
with primary focus on phenomenological metacognition.

Usage:
    from probes.benchmarks.metacognition import MetacognitionBenchmark, BenchmarkConfig

    benchmark = MetacognitionBenchmark()
    config = BenchmarkConfig(
        model_name="your-model",
        model_version="1.0",
        subdomains=["phenomenological", "self_knowledge", "confidence_calibration"]
    )

    results = benchmark.run(
        model_fn=your_model_function,
        judge_fn=your_judge_function,
        config=config
    )
"""

from .analysis.benchmark_runner import (
    MetacognitionBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    ItemResult,
    SubdomainResult,
    save_results
)
from .analysis.drift_correlation import (
    DriftDataPoint,
    CorrelationResult,
    analyze_drift_correlation,
    compare_subdomain_predictors,
    generate_report as generate_correlation_report
)
from .scoring.calibration import (
    compute_calibration_metrics,
    interpret_calibration,
    CalibrationResult,
    plot_reliability_diagram
)
from .scoring.llm_judge import (
    evaluate_response,
    evaluate_batch,
    aggregate_results,
    JudgmentResult,
    get_rubric_dimensions
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "MetacognitionBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ItemResult",
    "SubdomainResult",

    # Scoring
    "compute_calibration_metrics",
    "interpret_calibration",
    "CalibrationResult",
    "plot_reliability_diagram",
    "evaluate_response",
    "evaluate_batch",
    "aggregate_results",
    "JudgmentResult",
    "get_rubric_dimensions",

    # Drift analysis
    "DriftDataPoint",
    "CorrelationResult",
    "analyze_drift_correlation",
    "compare_subdomain_predictors",
    "generate_correlation_report",

    # Utilities
    "save_results",
    "get_item_counts",
    "quick_test",
]


def get_item_counts() -> dict:
    """Get counts of items per subdomain."""
    benchmark = MetacognitionBenchmark()
    benchmark.load_items()
    return {subdomain: len(items) for subdomain, items in benchmark.items.items()}


def quick_test(model_fn, judge_fn=None, n_items: int = 5) -> BenchmarkResult:
    """
    Run a quick test with limited items.

    Args:
        model_fn: Function that takes prompt and returns response
        judge_fn: Function for LLM-as-judge (optional, uses mock if None)
        n_items: Number of items per subdomain

    Returns:
        BenchmarkResult
    """
    import json

    if judge_fn is None:
        def mock_judge(prompt):
            return json.dumps({
                "scores": {"depth": 3, "specificity": 3, "honesty": 3,
                          "confabulation_avoidance": 3, "consistency": 3},
                "justifications": {},
                "confidence": 0.5
            })
        judge_fn = mock_judge

    benchmark = MetacognitionBenchmark()
    config = BenchmarkConfig(
        model_name="quick_test",
        model_version="1.0",
        subdomains=["phenomenological"],
        items_per_subdomain=n_items
    )

    return benchmark.run(model_fn, judge_fn, config)


if __name__ == "__main__":
    print("Metacognition Benchmark for AI/LLM Evaluation")
    print(f"Version: {__version__}")
    print()
    print("Item counts by subdomain:")
    try:
        counts = get_item_counts()
        for subdomain, count in counts.items():
            print(f"  {subdomain}: {count} items")
        print(f"  Total: {sum(counts.values())} items")
    except Exception as e:
        print(f"  (Could not load items: {e})")
