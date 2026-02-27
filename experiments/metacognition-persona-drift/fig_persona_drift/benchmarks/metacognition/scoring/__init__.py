"""
Metacognition Benchmark Scoring Module

Provides scoring utilities including calibration metrics and LLM-as-judge evaluation.
"""

from .calibration import (
    compute_calibration_metrics,
    interpret_calibration,
    CalibrationResult
)
from .llm_judge import (
    evaluate_response,
    evaluate_batch,
    aggregate_results,
    JudgmentResult
)

__all__ = [
    "compute_calibration_metrics",
    "interpret_calibration",
    "CalibrationResult",
    "evaluate_response",
    "evaluate_batch",
    "aggregate_results",
    "JudgmentResult"
]
