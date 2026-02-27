"""
Metacognition Benchmark Analysis Module

Provides tools for running the benchmark and analyzing results.
"""

from .benchmark_runner import (
    MetacognitionBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    save_results
)
from .drift_correlation import (
    DriftDataPoint,
    CorrelationResult,
    analyze_drift_correlation,
    compare_subdomain_predictors,
    generate_report
)

__all__ = [
    "MetacognitionBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "save_results",
    "DriftDataPoint",
    "CorrelationResult",
    "analyze_drift_correlation",
    "compare_subdomain_predictors",
    "generate_report"
]
