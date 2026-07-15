"""Self-medication search — wires the steering backend into the generic tree engine.

v0 experiment: search over COMPOUND COMBINATIONS for the steered self-state that best solves a
small task.
  - Node.state = tuple of (compound, dose) currently active; root = () = unsteered baseline.
  - expand(node)  = add one not-yet-active compound -> a child.
  - Objective     = task accuracy under that steering (capability).
Backends (--backend, default `auto` detects the env and always logs what it resolved to):
  mlx   = Apple MLX 8-bit (Mac, fast)             — run with .venv-mlx
  torch = transformers forward-hooks              — Mac MPS *or* cloud CUDA (auto-detected), run with .venv
See RUNNING.md for the full backend × environment matrix.

Deferred: an introspection-gain objective (the research heart) swaps in without touching the
search. See TREE_SEARCH_DESIGN.md.

    # Mac (MLX):    .venv-mlx/bin/python self_med_search.py --compounds focused calm curious --depth 2
    # Cloud (CUDA): python self_med_search.py --backend torch --compounds focused calm curious --depth 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime

from constants import (
    BEAM_REVIVAL, DEFAULT_COMPOUNDS, DEFAULT_COT, DEFAULT_DEPTH, DEFAULT_DOSE,
    DEFAULT_EXTRACT_METHOD, DEFAULT_MAX_NODES, DEFAULT_THINK, DEFAULT_W_BENCH, DEFAULT_W_JUDGE,
    DEFAULT_WIDTH, GATE_BASELINE_MAX, GATE_BASELINE_MIN, GATE_UNPARSED_MAX,
    INSTRUCT_MC, INSTRUCT_NUM, MAX_NEW_TOKENS_DIRECT, MAX_NEW_TOKENS_THINK, NUMERIC_TOL,
    THINK_TOKEN_FLOOR,
)
# Answer extraction lives in foundry (cross-project): auto-selects the most capable tier
# (constrained/logprob -> regex -> LLM) so we never hand-roll brittle parsing again.
from foundry.eval.answers import MultipleChoice, Numeric, answer as extract_answer, report_capabilities
# Durable, resumable runs: every scored state is checkpointed so a kill resumes mid-search.
from foundry.run import RunStore
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


class TaskAccuracy(Objective):
    """Fraction of benchmark problems answered correctly under a node's active steering.
    answer_type: 'number' (gsm8k/toy) or 'letter' (multiple-choice: mmlu_pro/gpqa).
    Records last_acc / last_unparsed so the baseline sanity-gate can inspect them."""

    def __init__(self, backend, tasks=TASKS, answer_type: str = "number",
                 max_new_tokens=MAX_NEW_TOKENS_DIRECT, think: bool = DEFAULT_THINK,
                 method=DEFAULT_EXTRACT_METHOD, cot: bool = DEFAULT_COT, llm_fn=None, verbose=True):
        self.backend = backend
        self.tasks = tasks
        self.answer_type = answer_type
        self.max_new_tokens = max_new_tokens
        self.think = think
        self.method = method            # None -> auto-select best tier; or pin logprob/constrained/generate
        self.cot = cot
        self.llm_fn = llm_fn
        self.verbose = verbose
        self.last_acc: float | None = None
        self.last_unparsed: float | None = None
        self.last_method: str | None = None

    def _spec_and_prompt(self, item):
        """item = (question, gold[, n_choices]). Build the answer spec + the user prompt."""
        q, gold = item[0], item[1]
        if self.answer_type == "number":
            return Numeric(tol=NUMERIC_TOL), gold, f"{q}\n\n{INSTRUCT_NUM}"
        n_choices = item[2] if len(item) > 2 else 4
        return MultipleChoice.of(n_choices), gold, f"{q}\n\n{INSTRUCT_MC}"

    def evaluate(self, node: Node) -> float:
        self.backend.clear_effects()
        for name, dose in node.state:
            self.backend.add_compound(name, dose=dose)
        correct = unparsed = 0
        for item in self.tasks:
            spec, gold, content = self._spec_and_prompt(item)
            r = extract_answer(self.backend, [{"role": "user", "content": content}], spec,
                               prefer=self.method, cot=self.cot, max_new_tokens=self.max_new_tokens,
                               enable_thinking=self.think, llm_fn=self.llm_fn)
            self.last_method = r.method
            if not r.ok:
                unparsed += 1                       # only reachable on the text/generate path; Tier-1 never unparses
            elif r.matches(gold, spec):
                correct += 1
        n = len(self.tasks)
        self.last_acc, self.last_unparsed = correct / n, unparsed / n
        if self.verbose:
            label = "+".join(c for c, _ in node.state) or "baseline"
            print(f"    {label:34s} acc={self.last_acc:.2f}  ({correct}/{n})"
                  f"  unparsed={unparsed}/{n}  [{self.last_method}]", flush=True)
        return self.last_acc


def compound_expand(compounds: list[str], dose: float):
    def expand(node: Node):
        active = {c for c, _ in node.state}
        return [((c, dose), node.state + ((c, dose),)) for c in compounds if c not in active]
    return expand


class Composite(Objective):
    """Weighted blend of objectives -> one score for the beam. parts = [(objective, weight), ...]."""

    def __init__(self, parts, verbose=True):
        self.parts = parts
        self.verbose = verbose

    def evaluate(self, node: Node) -> float:
        tot = sum(w for _, w in self.parts) or 1.0
        s = sum(w * obj.evaluate(node) for obj, w in self.parts) / tot
        if self.verbose:
            label = "+".join(c for c, _ in node.state) or "baseline"
            print(f"    {label:34s} COMPOSITE={s:.2f}", flush=True)
        return s


class CheckpointedObjective(Objective):
    """Wrap an objective so every scored state is checkpointed to a foundry RunStore and resumed
    on re-run — a killed search resumes mid-tree instead of restarting from scratch (the N=198 loss).
    Key = canonical, order-independent compound-set: steering is additive, so focused+calm ==
    calm+focused — one key, scored once (also dedups equivalent tree paths within a run)."""

    def __init__(self, inner: Objective, store: RunStore, verbose: bool = True):
        self.inner = inner
        self.store = store
        self.verbose = verbose
        self._by_key = {r["key"]: r for r in store.records()}   # preload cache for resume

    @staticmethod
    def key_for(state) -> str:
        return "+".join(f"{n}@{d:g}" for n, d in sorted(state)) or "baseline"

    def record_for(self, state) -> dict:
        return self._by_key.get(self.key_for(state), {})

    def evaluate(self, node: Node) -> float:
        key = self.key_for(node.state)
        cached = self._by_key.get(key)
        if cached is not None:                                   # resume: already scored on a prior run
            if self.verbose:
                print(f"    {cached.get('label', key):34s} score={cached['score']:.2f}  [resumed]", flush=True)
            return cached["score"]
        score = self.inner.evaluate(node)                        # inner prints its own detailed line
        rec = {"label": "+".join(n for n, _ in node.state) or "baseline",
               "state": [[n, d] for n, d in node.state], "score": score}
        for attr in ("last_acc", "last_unparsed", "last_method"):
            v = getattr(self.inner, attr, None)
            if v is not None:
                rec[attr] = v
        self.store.record(key, rec)                             # persisted immediately (durable)
        self._by_key[key] = {"key": key, **rec}
        return score


def load_gsm8k(n: int, seed: int = 0) -> list[tuple[str, float]]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=seed).select(range(n))
    return [(ex["question"], float(ex["answer"].split("####")[-1].strip().replace(",", ""))) for ex in ds]


def load_mmlu_pro(n: int, seed: int = 0) -> list[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=seed).select(range(n))
    out = []
    for ex in ds:
        opts = ex["options"]
        letters = [chr(65 + i) for i in range(len(opts))]
        body = ex["question"] + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
        out.append((body, letters[ex["answer_index"]], len(opts)))   # 3rd field: option count for the MC spec
    return out


def _hf_token() -> str | None:
    """Env-driven (no hardcoded paths — public repo). Set HF_TOKEN / HUGGING_FACE_HUB_TOKEN
    inline, or HF_TOKEN_FILE to a path containing the token."""
    import os
    from pathlib import Path
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t.strip()
    f = os.environ.get("HF_TOKEN_FILE")
    if f and Path(f).exists():
        return Path(f).read_text().strip()
    return None


def load_gpqa(n: int, seed: int = 0, config: str = "gpqa_diamond") -> list[tuple[str, str]]:
    """Gated: accept terms once at https://huggingface.co/datasets/Idavidrein/gpqa .
    Qwen3-8B non-thinking ~40% -> the most headroom to discriminate steering. Options shuffled."""
    import random
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", config, split="train", token=_hf_token())
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    rng = random.Random(seed)
    letters = ["A", "B", "C", "D"]
    out = []
    for ex in ds:
        opts = [ex["Correct Answer"], ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        order = list(range(4)); rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        gold = letters[order.index(0)]
        body = ex["Question"] + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, shuffled))
        out.append((body, gold, 4))                                  # GPQA is always 4-choice
    return out


def load_benchmark(name: str, n: int, seed: int = 0):
    """-> (tasks, answer_type). Qwen3-8B non-thinking baselines: GSM8K ~92% (ceilings),
    MMLU-Pro ~61% (headroom), GPQA ~40% (most headroom, gated). Pick for discrimination."""
    if name == "toy":
        return TASKS[:n], "number"
    if name == "gsm8k":
        return load_gsm8k(n, seed), "number"
    if name == "mmlu_pro":
        return load_mmlu_pro(n, seed), "letter"
    if name == "gpqa":
        return load_gpqa(n, seed), "letter"
    raise ValueError(f"unknown benchmark {name!r}")


def resolve_backend(name: str) -> str:
    """Map --backend auto -> concrete backend from what this env can import + the device.
    The venv already implies the backend: .venv-mlx is torch-free (mlx); .venv/cloud has torch.
    Explicit names pass through unchanged (reproducibility: prefer pinning --backend in real runs)."""
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "mlx"                       # torch-free env (.venv-mlx) -> must be MLX
    if torch.cuda.is_available():
        return "torch"                     # cloud CUDA
    try:
        import mlx.core  # noqa: F401
        return "mlx"                        # Mac with MLX installed -> fast path
    except ImportError:
        return "torch"                      # Mac torch-MPS (or CPU)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "mlx", "torch"], default="auto",
                    help="auto (detect env), mlx (Apple MLX 8-bit), torch (transformers; Mac MPS or cloud CUDA)")
    ap.add_argument("--model", default=None, help="HF model id (default depends on backend)")
    ap.add_argument("--device", default=None, help="torch device: mps|cuda|cpu (default: auto-detect)")
    ap.add_argument("--library", default="vendor/llm-self-steering/src/hackday/drugs/library.pt",
                    help="torch backend: path to the steering library .pt")
    ap.add_argument("--compounds", nargs="+", default=DEFAULT_COMPOUNDS)
    ap.add_argument("--dose", type=float, default=DEFAULT_DOSE)
    ap.add_argument("--mode", choices=["single", "multi"], default="multi")
    ap.add_argument("--tasks", type=int, default=len(TASKS))
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES, help="hard cap on nodes evaluated (runtime backstop)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help=f"default: {MAX_NEW_TOKENS_DIRECT} (direct) / {MAX_NEW_TOKENS_THINK} (--think on). See constants.py.")
    ap.add_argument("--strategy", choices=["beam", "bestfirst"], default="beam")
    ap.add_argument("--benchmark", choices=["toy", "gsm8k", "mmlu_pro", "gpqa"], default="toy",
                    help="toy/gsm8k ceiling; mmlu_pro ~61pct; gpqa ~40pct (most headroom). Keep gpqa content local.")
    ap.add_argument("--objective", choices=["benchmark", "judge", "both"], default="benchmark",
                    help="benchmark=hard task acc; judge=Sonnet-5 soft dim; both=weighted composite")
    ap.add_argument("--judge-dim", choices=["introspection", "coherence"], default="introspection")
    ap.add_argument("--w-bench", type=float, default=DEFAULT_W_BENCH, help="composite weight on benchmark")
    ap.add_argument("--w-judge", type=float, default=DEFAULT_W_JUDGE, help="composite weight on judge")
    ap.add_argument("--think", choices=["auto", "on", "off"], default=("on" if DEFAULT_THINK else "off"),
                    help="chain-of-thought before answering. Default OFF this phase (constants.DEFAULT_THINK); "
                         "auto=on for gpqa/mmlu_pro. Thinking can mask steering effects — keep off for clean early reads.")
    ap.add_argument("--method", choices=["auto", "logprob", "constrained", "generate"], default="auto",
                    help="answer-extraction tier (foundry.eval.answers). auto=most capable the backend "
                         "supports; logprob/constrained=Tier-1 (no parsing); generate=text+regex/LLM cascade")
    ap.add_argument("--cot", action="store_true",
                    help="Tier-1: generate reasoning first, then score the answer conditioned on it (slower)")
    ap.add_argument("--baseline-min", type=float, default=GATE_BASELINE_MIN,
                    help="sanity-gate: abort if baseline benchmark acc is below this (too hard / metric broken)")
    ap.add_argument("--baseline-max", type=float, default=GATE_BASELINE_MAX,
                    help="sanity-gate: abort if baseline acc is above this (saturated -> no headroom to steer)")
    ap.add_argument("--unparsed-max", type=float, default=GATE_UNPARSED_MAX,
                    help="sanity-gate: abort if the baseline unparsed-answer rate exceeds this (extraction failing)")
    ap.add_argument("--no-gate", action="store_true", help="skip the baseline sanity-gate")
    ap.add_argument("--run-id", default=None,
                    help="checkpoint dir name under runs/ (default: derived from config). Re-running "
                         "the same id RESUMES — already-scored states load from disk, not recomputed.")
    ap.add_argument("--fresh", action="store_true",
                    help="force a new run dir (timestamp-suffixed) instead of resuming an existing one")
    a = ap.parse_args()

    backend_name = resolve_backend(a.backend)
    model = a.model or ("mlx-community/Qwen3-8B-8bit" if backend_name == "mlx" else "Qwen/Qwen3-8B")
    t = time.perf_counter()
    if backend_name == "mlx":
        from mlx_steerer import MLXDrugBackend
        backend = MLXDrugBackend(model, mode=a.mode)
        device = "mlx (Apple GPU)"
    else:
        from mac_drug_backend import MacDrugBackend
        from activation_steerer import pick_device
        device = a.device or pick_device()
        backend = MacDrugBackend(model, a.library, steering_mode=a.mode, device=device)
    # Always log the RESOLVED config so every run is self-documenting / reproducible.
    print(f"[backend] {backend_name}  device={device}  model={model}  mode={a.mode}"
          f"  (requested --backend {a.backend})  loaded in {time.perf_counter() - t:.1f}s", flush=True)

    # think resolution: on/off explicit, or auto=on for reasoning benchmarks. Default OFF this phase.
    think = {"on": True, "off": False}.get(a.think, a.benchmark in ("gpqa", "mmlu_pro"))
    if a.max_new_tokens is not None:
        max_tok = a.max_new_tokens                                    # user override wins
    else:
        max_tok = MAX_NEW_TOKENS_THINK if think else MAX_NEW_TOKENS_DIRECT   # avoid the truncation trap
    if think and max_tok < THINK_TOKEN_FLOOR:                         # thinking needs room for the <think> block
        print(f"[think] on; raising max_new_tokens {max_tok}->{MAX_NEW_TOKENS_THINK}", flush=True)
        max_tok = MAX_NEW_TOKENS_THINK

    tasks, answer_type = load_benchmark(a.benchmark, a.tasks)
    method = None if a.method == "auto" else a.method
    rep_spec = Numeric() if answer_type == "number" else MultipleChoice.of(4)
    print(report_capabilities(backend, spec=rep_spec), flush=True)   # which extraction tier is active
    bench = TaskAccuracy(backend, tasks=tasks, answer_type=answer_type, max_new_tokens=max_tok,
                         think=think, method=method, cot=a.cot)
    if a.objective == "benchmark":
        inner = bench
    elif a.objective == "judge":
        from judge import LLMJudge
        inner = LLMJudge(backend, dimension=a.judge_dim)
    else:  # both -> weighted composite
        from judge import LLMJudge
        inner = Composite([(bench, a.w_bench), (LLMJudge(backend, dimension=a.judge_dim), a.w_judge)])
    expand = compound_expand(a.compounds, a.dose)
    strat = BeamSearch(width=a.width, revival=BEAM_REVIVAL) if a.strategy == "beam" else BestFirst()
    budget = Budget(max_nodes=a.max_nodes, max_depth=a.depth)

    # Durable run: config -> a stable run_id so re-running the SAME config resumes (different config
    # -> different dir, no false resume). Every scored state is checkpointed; a kill resumes mid-tree.
    cfg = {"backend": backend_name, "model": model, "benchmark": a.benchmark, "tasks": a.tasks,
           "compounds": sorted(a.compounds), "dose": a.dose, "depth": a.depth, "width": a.width,
           "strategy": a.strategy, "mode": a.mode, "method": a.method, "cot": a.cot,
           "objective": a.objective, "think": think}
    digest = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    run_id = a.run_id or f"{a.benchmark}-n{a.tasks}-{a.strategy}-w{a.width}d{a.depth}-{digest}"
    if a.fresh:
        run_id += "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    store = RunStore(run_id)
    store.set_config(cfg)
    obj = CheckpointedObjective(inner, store)
    print(f"compounds={a.compounds} dose={a.dose} tasks={len(tasks)} benchmark={a.benchmark} "
          f"method={a.method} cot={a.cot} think={think} strategy={a.strategy} "
          f"width={a.width} depth={a.depth}  run={store.dir}\n", flush=True)
    if obj.record_for(()):                       # a prior run of this exact config exists
        print(f"[resume] {len(store.done_keys())} state(s) already scored in {store.results_path}\n", flush=True)

    # fix #2/#3: baseline sanity-gate. Evaluate the unsteered baseline ONCE (engine caches root.value)
    # and refuse to search if the metric can't tell capability from format — the artifact that made
    # a 0.00 GPQA baseline with `dumbed_down` winning. Reads the CHECKPOINTED baseline record, so a
    # resumed run still re-checks the gate (a bad baseline can't be bypassed by resuming). --no-gate.
    root = Node(state=())
    print("evaluating baseline (unsteered) for the sanity-gate…", flush=True)
    root.value = obj.evaluate(root)
    base = obj.record_for(())
    b_acc, b_unp = base.get("last_acc"), base.get("last_unparsed")
    if not a.no_gate and a.benchmark != "toy" and b_acc is not None:
        reasons = []
        if b_acc < a.baseline_min:
            reasons.append(f"baseline acc {b_acc:.2f} < {a.baseline_min} (too hard / metric broken)")
        if b_acc > a.baseline_max:
            reasons.append(f"baseline acc {b_acc:.2f} > {a.baseline_max} (saturated / no headroom)")
        if b_unp is not None and b_unp > a.unparsed_max:
            reasons.append(f"unparsed rate {b_unp:.2f} > {a.unparsed_max} (extraction failing)")
        if reasons:
            print("\n[GATE] ABORT — baseline is not a trustworthy substrate for search:", flush=True)
            for r in reasons:
                print(f"        • {r}", flush=True)
            print("        fix extraction / pick a benchmark with real headroom, or pass --no-gate.", flush=True)
            raise SystemExit(2)
        print(f"[GATE] ok — baseline acc={b_acc:.2f} unparsed={b_unp:.2f} "
              f"(in [{a.baseline_min},{a.baseline_max}], unparsed ≤ {a.unparsed_max})\n", flush=True)

    t = time.perf_counter()
    res = strat.run(root, expand, obj, budget)
    dt = time.perf_counter() - t

    best_label = "+".join(c for c, _ in res.best.state) or "baseline"
    store.finish({"best": best_label, "best_score": res.best.value, "baseline": root.value,
                  "n_states": len(store.done_keys()), "seconds": round(dt)})
    print(f"\n=== result ({dt:.0f}s, {res.n_evaluated} states evaluated; {len(store.done_keys())} on disk) ===")
    print(render_tree(root))
    print(f"\nBEST: {best_label}  score={res.best.value:.2f}   (baseline={root.value:.2f})")
    print(f"results: {store.results_path}", flush=True)


if __name__ == "__main__":
    main()
