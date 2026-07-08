"""Self-medication search — wires the steering backend into the generic tree engine.

v0 experiment: search over COMPOUND COMBINATIONS for the steered self-state that best solves a
small task.
  - Node.state = tuple of (compound, dose) currently active; root = () = unsteered baseline.
  - expand(node)  = add one not-yet-active compound -> a child.
  - Objective     = task accuracy under that steering (capability).
Backend defaults to MLXDrugBackend (8-bit, the fast path) — run with .venv-mlx.

Deferred: an introspection-gain objective (the research heart) swaps in without touching the
search. See TREE_SEARCH_DESIGN.md.

    .venv-mlx/bin/python self_med_search.py --compounds focused calm curious --depth 2
"""
from __future__ import annotations

import argparse
import re
import time

from tree_search import BeamSearch, BestFirst, Budget, Node, Objective, render_tree

# v0 task: harder multi-step GSM8K-style problems (the easy set hit the ceiling — no signal).
# Swap in vendor problems/gsm8k.py for the real dataset later.
TASKS: list[tuple[str, float]] = [
    ("A store had 120 apples. It sold 1/3 of them in the morning, then 1/4 of the remaining apples in the afternoon. How many apples are left?", 60),
    ("Tom reads 12 pages per day for 5 days, then doubles his daily rate for the next 3 days. How many pages does he read in total?", 132),
    ("A class has 30 students. 40% play soccer, half of the remaining students play basketball, and the rest read books. How many students read books?", 9),
    ("Sarah buys 3 books at $8 each and 2 notebooks at $3 each, then uses a $5 coupon. How many dollars does she pay?", 25),
    ("A water tank holds 200 liters. It is 3/5 full. Then 40 liters are added, and afterward 1/4 of the water is drained. How many liters remain?", 120),
    ("A baker made 96 cookies. He packs them into boxes of 8, then sells 7 boxes. How many cookies are left unsold?", 40),
]


def _final_number(text: str) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


class TaskAccuracy(Objective):
    """Fraction of task problems answered correctly under a node's active steering."""

    def __init__(self, backend, tasks=TASKS, max_new_tokens=220, verbose=True):
        self.backend = backend
        self.tasks = tasks
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

    def evaluate(self, node: Node) -> float:
        self.backend.clear_effects()
        for name, dose in node.state:
            self.backend.add_compound(name, dose=dose)
        correct = 0
        for q, gold in self.tasks:
            msgs = [{"role": "user", "content": q + "\nEnd your reply with 'Answer: <number>'."}]
            out = self.backend.generate(msgs, max_new_tokens=self.max_new_tokens)
            got = _final_number(out)
            if got is not None and abs(got - gold) < 1e-6:
                correct += 1
        acc = correct / len(self.tasks)
        if self.verbose:
            label = "+".join(c for c, _ in node.state) or "baseline"
            print(f"    {label:34s} acc={acc:.2f}  ({correct}/{len(self.tasks)})", flush=True)
        return acc


def compound_expand(compounds: list[str], dose: float):
    def expand(node: Node):
        active = {c for c, _ in node.state}
        return [((c, dose), node.state + ((c, dose),)) for c in compounds if c not in active]
    return expand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-8bit")
    ap.add_argument("--compounds", nargs="+", default=["focused", "calm", "curious", "anxious"])
    ap.add_argument("--dose", type=float, default=1.5)
    ap.add_argument("--mode", choices=["single", "multi"], default="multi")
    ap.add_argument("--tasks", type=int, default=len(TASKS))
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--strategy", choices=["beam", "bestfirst"], default="beam")
    a = ap.parse_args()

    from mlx_steerer import MLXDrugBackend
    print(f"Loading {a.model} (MLX, {a.mode}) ...", flush=True)
    t = time.perf_counter()
    backend = MLXDrugBackend(a.model, mode=a.mode)
    print(f"  loaded in {time.perf_counter() - t:.1f}s", flush=True)

    tasks = TASKS[: a.tasks]
    obj = TaskAccuracy(backend, tasks=tasks, max_new_tokens=a.max_new_tokens)
    expand = compound_expand(a.compounds, a.dose)
    strat = BeamSearch(width=a.width, revival=1) if a.strategy == "beam" else BestFirst()
    budget = Budget(max_nodes=40, max_depth=a.depth)

    print(f"compounds={a.compounds} dose={a.dose} tasks={len(tasks)} "
          f"strategy={a.strategy} width={a.width} depth={a.depth}\n", flush=True)
    t = time.perf_counter()
    root = Node(state=())
    res = strat.run(root, expand, obj, budget)
    dt = time.perf_counter() - t

    print(f"\n=== result ({dt:.0f}s, {res.n_evaluated} states evaluated) ===")
    print(render_tree(root))
    best_label = "+".join(c for c, _ in res.best.state) or "baseline"
    print(f"\nBEST: {best_label}  acc={res.best.value:.2f}   (baseline acc={root.value:.2f})")


if __name__ == "__main__":
    main()
