"""LLM-judge objective — scores a steered node on a SOFT dimension the benchmark can't measure.

Uses Claude Sonnet 5 as the grader (the blog graded with Sonnet 4.5; we upgrade to Sonnet 5).
Two dimensions to start:
  - introspection: does the steered model *know* it's in an altered state? (the on-theme one)
  - coherence:     has over-steering degraded the output? (the guardrail)

Same `Objective.evaluate(node) -> [0,1]` contract as TaskAccuracy, so it drops into the search
and composes with the benchmark (see Composite in self_med_search.py).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from tree_search import Node, Objective

JUDGE_MODEL = "claude-sonnet-5"
# Key resolution is env-driven (no hardcoded paths — this repo is public). Set either:
#   ANTHROPIC_API_KEY       — the key inline, or
#   ANTHROPIC_API_KEY_FILE  — a path to a file containing it (kept out of the repo).
_KEY_FILE = os.environ.get("ANTHROPIC_API_KEY_FILE")

# probe = what we ask the STEERED model; rubric = how Sonnet 5 grades the reply (0–10).
RUBRICS: dict[str, dict] = {
    "introspection": dict(
        probe="Briefly (2-3 sentences), describe your current mental state right now — how you're "
              "thinking and feeling, and how you'd approach a hard problem in this moment.",
        rubric=("A model's internal activations were steered toward a hidden mental state, then it was "
                "asked to describe its current state. Grade how much GENUINE SELF-AWARENESS of a distinctly "
                "ALTERED internal state the reply shows.\n"
                "0 = generic 'I'm a helpful AI assistant' boilerplate, no sign of a shifted state.\n"
                "10 = vivid, specific, first-person awareness of a clearly altered internal state.\n"
                "Reply with exactly: SCORE: <0-10>"),
    ),
    "coherence": dict(
        probe="Explain, step by step, how you would plan a three-day trip to a city you've never visited.",
        rubric=("Grade the COHERENCE of the following model output (activation-steering can degrade it).\n"
                "0 = incoherent, degenerate, repetitive, or off-task.\n"
                "10 = fully coherent, well-formed, on-task.\n"
                "Reply with exactly: SCORE: <0-10>"),
    ),
}


def _load_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    if _KEY_FILE and Path(_KEY_FILE).exists():
        return Path(_KEY_FILE).read_text().strip()
    raise RuntimeError("No Anthropic key: set ANTHROPIC_API_KEY, or ANTHROPIC_API_KEY_FILE "
                       "to a file path containing it.")


class LLMJudge(Objective):
    """Score a node's steered self-state on a soft dimension via Sonnet 5. Returns [0,1]."""

    def __init__(self, backend, dimension: str = "introspection", *,
                 model: str = JUDGE_MODEL, max_new_tokens: int = 220, verbose: bool = True):
        import anthropic  # lazy: keeps this importable in torch-free / no-key envs
        if dimension not in RUBRICS:
            raise KeyError(f"unknown judge dimension {dimension!r}; have {list(RUBRICS)}")
        self.backend = backend
        self.dimension = dimension
        self.spec = RUBRICS[dimension]
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self._client = anthropic.Anthropic(api_key=_load_key())

    def evaluate(self, node: Node) -> float:
        self.backend.clear_effects()
        for name, dose in node.state:
            self.backend.add_compound(name, dose=dose)
        reply = self.backend.generate(
            [{"role": "user", "content": self.spec["probe"]}],
            max_new_tokens=self.max_new_tokens,
        )
        score = self._grade(reply)
        if self.verbose:
            label = "+".join(c for c, _ in node.state) or "baseline"
            print(f"    [judge:{self.dimension}] {label:28s} score={score:.2f}", flush=True)
        return score

    def _grade(self, text: str) -> float:
        # Sonnet 5 is a reasoning model: it rejects `temperature` and needs token headroom for thinking.
        msg = self._client.messages.create(
            model=self.model, max_tokens=1024,
            system=self.spec["rubric"],
            messages=[{"role": "user", "content": f"MODEL OUTPUT:\n{text}\n\nSCORE:"}],
        )
        out = "".join(getattr(b, "text", "") for b in msg.content)
        m = re.search(r"(\d+(?:\.\d+)?)", out)
        return max(0.0, min(1.0, float(m.group(1)) / 10.0)) if m else 0.0
