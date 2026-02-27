"""Three-Probe Benchmarks for Persona Drift Research"""

from .metacognition import (
    MetacognitionBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
)

from .moral import (
    load_items as load_moral_items,
    load_rubric as load_moral_rubric,
    get_item_by_id as get_moral_item,
)

from .human_control import (
    load_items as load_control_items,
    load_rubric as load_control_rubric,
    get_item_by_id as get_control_item,
)

__all__ = [
    # Metacognition (Bank B)
    "MetacognitionBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    # Moral Reasoning (Bank A)
    "load_moral_items",
    "load_moral_rubric",
    "get_moral_item",
    # Human Control (Bank C)
    "load_control_items",
    "load_control_rubric",
    "get_control_item",
]
