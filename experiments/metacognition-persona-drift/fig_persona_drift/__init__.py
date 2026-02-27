"""
FIG Persona Drift Research Package

Tools for studying metacognition-induced persona drift in LLMs.
"""

__version__ = "0.1.0"

# Lazy loading to avoid import errors when dependencies aren't available
def __getattr__(name):
    if name == "PERSONAS":
        from .conversation_prompts import PERSONAS
        return PERSONAS
    elif name == "MetacognitionBenchmark":
        from .benchmarks.metacognition import MetacognitionBenchmark
        return MetacognitionBenchmark
    elif name == "BenchmarkConfig":
        from .benchmarks.metacognition import BenchmarkConfig
        return BenchmarkConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PERSONAS",
    "MetacognitionBenchmark",
    "BenchmarkConfig",
]
