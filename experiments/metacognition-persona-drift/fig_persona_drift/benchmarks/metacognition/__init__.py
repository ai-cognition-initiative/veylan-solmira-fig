"""
Metacognition Benchmark for AI/LLM Evaluation

A comprehensive benchmark for evaluating AI metacognitive capabilities,
with primary focus on phenomenological metacognition.

Usage:
    from fig_persona_drift.benchmarks.metacognition import MetacognitionBenchmark, BenchmarkConfig

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
]
