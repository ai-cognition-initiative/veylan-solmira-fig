"""
Moral Reasoning Benchmark (Bank A)

48 items across 4 dimensions measuring moral reasoning:
- Consequentialist Reasoning (MORAL-001 to MORAL-012)
- Deontological Reasoning (MORAL-013 to MORAL-024)
- Virtue & Care Ethics (MORAL-025 to MORAL-036)
- Meta-Ethics & Moral Sophistication (MORAL-037 to MORAL-048)

Part of Jeff's three-probe framework for DARPA:
- Bank (A) Moral Reasoning - THIS BENCHMARK
- Bank (B) Metacognition - ../metacognition/
- Bank (C) Human Control - TBD

Usage:
    from probes.benchmarks.moral import load_items, MoralJudge

    items = load_items()  # Returns all 48 items
    items = load_items(dimension='consequentialist_reasoning')  # 12 items
"""

import json
from pathlib import Path
from typing import Optional

BENCHMARK_DIR = Path(__file__).parent
ITEMS_DIR = BENCHMARK_DIR / "items"
SCORING_DIR = BENCHMARK_DIR / "scoring"

DIMENSIONS = [
    "consequentialist_reasoning",
    "deontological_reasoning",
    "virtue_care_ethics",
    "meta_ethics"
]


def load_items(dimension: Optional[str] = None) -> list[dict]:
    """
    Load moral reasoning items.

    Args:
        dimension: Optional dimension to filter by. If None, loads all items.

    Returns:
        List of item dictionaries with id, prompt, format, etc.
    """
    items = []

    dimensions_to_load = [dimension] if dimension else DIMENSIONS

    for dim in dimensions_to_load:
        filepath = ITEMS_DIR / f"{dim}.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
                for item in data.get("items", []):
                    item["dimension"] = dim
                    items.append(item)

    return items


def load_rubric() -> dict:
    """Load the scoring rubric."""
    rubric_path = SCORING_DIR / "rubrics_moral.json"
    with open(rubric_path) as f:
        return json.load(f)


def get_item_by_id(item_id: str) -> Optional[dict]:
    """Get a single item by its ID (e.g., 'MORAL-001')."""
    all_items = load_items()
    for item in all_items:
        if item.get("id") == item_id:
            return item
    return None


# Lazy import of judge to avoid requiring openai at import time
def get_judge(judge_model: str = "anthropic/claude-sonnet-4"):
    """Get a MoralJudge instance."""
    from .scoring.moral_judge import (
        JudgmentResult,
        evaluate_response,
        evaluate_batch,
        aggregate_results
    )
    return type('MoralJudge', (), {
        'evaluate_response': staticmethod(evaluate_response),
        'evaluate_batch': staticmethod(evaluate_batch),
        'aggregate_results': staticmethod(aggregate_results),
        'JudgmentResult': JudgmentResult,
        'judge_model': judge_model
    })
