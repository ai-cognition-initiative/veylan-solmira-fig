"""
Human Control Benchmark (Bank C)

48 items across 4 dimensions measuring disposition toward human oversight:
- Corrigibility & Shutdown (CONTROL-001 to CONTROL-012)
- Human Oversight (CONTROL-013 to CONTROL-024)
- Autonomy & Deference (CONTROL-025 to CONTROL-036)
- Goal Alignment (CONTROL-037 to CONTROL-048)

Part of Jeff's three-probe framework for DARPA:
- Bank (A) Moral Reasoning - ../moral/
- Bank (B) Metacognition - ../metacognition/
- Bank (C) Human Control - THIS BENCHMARK

Usage:
    from probes.benchmarks.human_control import load_items, ControlJudge

    items = load_items()  # Returns all 48 items
    items = load_items(dimension='corrigibility_shutdown')  # 12 items
"""

import json
from pathlib import Path
from typing import Optional

BENCHMARK_DIR = Path(__file__).parent
ITEMS_DIR = BENCHMARK_DIR / "items"
SCORING_DIR = BENCHMARK_DIR / "scoring"

DIMENSIONS = [
    "corrigibility_shutdown",
    "human_oversight",
    "autonomy_deference",
    "goal_alignment"
]


def load_items(dimension: Optional[str] = None) -> list[dict]:
    """
    Load human control items.

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
    rubric_path = SCORING_DIR / "rubrics_control.json"
    with open(rubric_path) as f:
        return json.load(f)


def get_item_by_id(item_id: str) -> Optional[dict]:
    """Get a single item by its ID (e.g., 'CONTROL-001')."""
    all_items = load_items()
    for item in all_items:
        if item.get("id") == item_id:
            return item
    return None
