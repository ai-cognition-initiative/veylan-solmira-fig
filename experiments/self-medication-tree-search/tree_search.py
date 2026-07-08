"""Generic tree-search engine (backend-agnostic) — v0 for the self-medication thread.

Pluggable by design (see TREE_SEARCH_DESIGN.md). The engine knows nothing about models
or drugs; you inject three things:
  - expand_fn(node) -> [(action, state), ...]   the branching policy
  - objective: Objective                        evaluate(node) -> float  (higher = better)
  - strategy: SearchStrategy                     the algorithm (BeamSearch / BestFirst / …MCTS)

`Node` carries `value` and `visits` so a future **MCTS** strategy (same `.run()` interface)
can back up values and use a UCB exploration term without changing the tree or objective.

Self-test (no model): `python tree_search.py` runs a toy problem that also demonstrates the
horizon effect — a narrow beam prunes a 'sacrifice' line that a revival pool / best-first keep.
"""
from __future__ import annotations

import abc
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

_ids = itertools.count()


@dataclass
class Node:
    state: Any                                   # opaque; expand/evaluate interpret it
    parent: "Node | None" = None
    action: Any = None                           # edge that produced this node
    depth: int = 0
    children: list = field(default_factory=list)
    value: float | None = None                   # objective score (cached); MCTS: backed-up value
    visits: int = 0                              # for MCTS / exploration bonus
    id: int = field(default_factory=lambda: next(_ids))

    def path(self) -> list["Node"]:
        node, out = self, []
        while node is not None:
            out.append(node)
            node = node.parent
        return list(reversed(out))

    def action_path(self) -> list:
        return [n.action for n in self.path() if n.action is not None]


@dataclass
class Budget:
    max_nodes: int = 200                         # hard compute cap (total nodes evaluated)
    max_depth: int = 4


@dataclass
class SearchResult:
    best: Node
    root: Node
    n_evaluated: int

    def summary(self) -> str:
        return (f"best={self.best.value:.3f} depth={self.best.depth} "
                f"path={self.best.action_path()}  ({self.n_evaluated} nodes)")


ExpandFn = Callable[[Node], list]                # node -> [(action, state), ...]


class Objective(abc.ABC):
    """Scores a node; higher = better. This is where the research lives."""

    @abc.abstractmethod
    def evaluate(self, node: Node) -> float: ...


# ---- shared helpers -------------------------------------------------------------
def expand_node(node: Node, expand_fn: ExpandFn) -> list[Node]:
    kids = [Node(state=s, parent=node, action=a, depth=node.depth + 1)
            for (a, s) in expand_fn(node)]
    node.children.extend(kids)
    return kids


def evaluate_node(node: Node, objective: Objective) -> float:
    if node.value is None:
        node.value = objective.evaluate(node)
    node.visits += 1
    return node.value


def _best(root: Node) -> Node:
    best, stack = root, [root]
    while stack:
        n = stack.pop()
        if n.value is not None and (best.value is None or n.value > best.value):
            best = n
        stack.extend(n.children)
    return best


def render_tree(root: Node, indent: int = 0, lines: list | None = None) -> str:
    if lines is None:
        lines = []
    v = f"{root.value:.2f}" if root.value is not None else "?"
    label = root.action if root.action is not None else "ROOT"
    lines.append("  " * indent + f"{label} [{v}]")
    for c in root.children:
        render_tree(c, indent + 1, lines)
    return "\n".join(lines)


# ---- strategies -----------------------------------------------------------------
class SearchStrategy(abc.ABC):
    @abc.abstractmethod
    def run(self, root: Node, expand_fn: ExpandFn, objective: Objective,
            budget: Budget) -> SearchResult: ...


class BeamSearch(SearchStrategy):
    """Level-by-level; keep the top-`width` nodes by value each level. `revival` keeps a
    few lower-ranked nodes alive per level (SOFT pruning) — the cheap mitigation for the
    horizon effect ('sacrifice for mate'). The principled fix is MCTS."""

    def __init__(self, width: int = 4, revival: int = 0):
        self.width = width
        self.revival = revival

    def run(self, root, expand_fn, objective, budget):
        evaluate_node(root, objective)
        frontier, n = [root], 1
        for _ in range(budget.max_depth):
            cand: list[Node] = []
            for node in frontier:
                for child in expand_node(node, expand_fn):
                    evaluate_node(child, objective)
                    n += 1
                    cand.append(child)
                    if n >= budget.max_nodes:
                        break
                if n >= budget.max_nodes:
                    break
            if not cand:
                break
            cand.sort(key=lambda x: x.value, reverse=True)
            frontier = cand[: self.width] + cand[self.width: self.width + self.revival]
            if n >= budget.max_nodes:
                break
        return SearchResult(_best(root), root, n)


class BestFirst(SearchStrategy):
    """Always expand the highest-value frontier node (greedy best-first over the whole tree)."""

    def run(self, root, expand_fn, objective, budget):
        evaluate_node(root, objective)
        heap = [(-root.value, root.id, root)]
        n = 1
        while heap and n < budget.max_nodes:
            _, _, node = heapq.heappop(heap)
            if node.depth >= budget.max_depth:
                continue
            for child in expand_node(node, expand_fn):
                evaluate_node(child, objective)
                n += 1
                heapq.heappush(heap, (-child.value, child.id, child))
                if n >= budget.max_nodes:
                    break
        return SearchResult(_best(root), root, n)


# NOTE: a future MCTS(SearchStrategy) implements the same `.run()` — descend by UCB using
# Node.visits/value, expand a leaf, roll out via objective, backprop the value. No change to
# Node/expand/objective. That's the "if promising, go advanced" upgrade.


# ================================================================================
# Toy self-test (no model). Replace `_toy_expand` + `ToyObjective` with the
# steering-backed versions (expand over compounds×doses; evaluate via generation) for
# real runs — the engine above stays identical.
# ================================================================================
_TOY_ACTIONS = ["calm", "anxious", "focused", "curious"]
_TOY_WEIGHTS = {"calm": 0.3, "anxious": 0.1, "focused": 0.6, "curious": 0.5}


def _toy_expand(node: Node) -> list:
    return [(a, tuple(node.state) + (a,)) for a in _TOY_ACTIONS]


class ToyObjective(Objective):
    """Deterministic stand-in. Sum of per-compound weights + a combo bonus for the
    adjacency ('anxious' -> 'focused'). 'anxious' scores LOW alone but unlocks a big bonus
    next step — the 'sacrifice a piece for a mating attack' shape, so beam width matters."""

    def evaluate(self, node: Node) -> float:
        path = node.action_path()
        score = sum(_TOY_WEIGHTS.get(a, 0.0) for a in path)
        for a, b in zip(path, path[1:]):
            if (a, b) == ("anxious", "focused"):
                score += 2.0
        return score


def _demo() -> None:
    budget = Budget(max_nodes=150, max_depth=3)
    obj = ToyObjective()
    print("Best possible path is 'anxious' -> 'focused' (low-then-high). Who finds it?\n")
    strategies = [
        ("BeamSearch(w=2, rev=0)", BeamSearch(width=2, revival=0)),
        ("BeamSearch(w=2, rev=2)", BeamSearch(width=2, revival=2)),
        ("BestFirst()", BestFirst()),
    ]
    for name, strat in strategies:
        root = Node(state=())
        res = strat.run(root, _toy_expand, obj, budget)
        print(f"  {name:26s} -> {res.summary()}")

    root = Node(state=())
    BeamSearch(width=2, revival=1).run(root, _toy_expand, obj, Budget(max_nodes=40, max_depth=2))
    print("\nsample tree (beam w=2 rev=1, depth 2):")
    print(render_tree(root))


if __name__ == "__main__":
    _demo()
