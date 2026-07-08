# Tree-search design — self-medication over self-instances

Design analysis for the core novel mechanic: a model **tree-searches over steered copies
of itself**. Written to keep v0 simple while staying *upgrade-ready* for MCTS, adversarial
framings, and learned value nets.

## Problem shape
Single-agent search over a tree of **self-configurations**.
- **Node** = a state: steering config (active compounds/doses) + conversation/history + context.
- **Edge (action)** = apply a compound at a dose, or generate a continuation.
- **Goal** = maximize an objective (below).

It is **not** a two-player game by default (no adversary), so classic **minimax / alpha-beta
don't natively fit** — they assume an opponent minimizing your score. (When an adversary *is*
the right frame → see Future options.)

## Architecture: three pluggable components
Decouple so any one can be swapped without touching the others — this is what buys
"simple now, advanced later":

1. **`expand(node) → children`** — branching policy. Branch over compounds×doses, or generate N continuations under a fixed config.
2. **`Objective.evaluate(node) → value`** — the "make number go up" target. **Swappable, possibly multi-objective:**
   - **capability** ("can it do X better") → capability-benchmark score
   - **introspection / self-knowledge** → probe-battery readout (reuse the 251-item battery)
   - **other sentience/welfare ability** → TBD signal
3. **`SearchStrategy.select(frontier) → next`** — the algorithm. v0 = beam/best-first; upgrade = MCTS.

Node representation + tree machinery stay stable; **strategy and objective are injected.**

## Pruning ↔ exploration/exploitation — the "sacrifice for a mating attack" problem
Pruning is a **compute necessity** (throughput budget below). But hard-pruning on *shallow,
immediate* value causes the **horizon effect**: a line that dips before it pays off (the piece
sacrifice) gets cut prematurely.

What preserves those lines (design for these, not naive greedy):
- **Value, not immediate reward.** `evaluate` should estimate *expected future* payoff (rollout or learned value), not the one-step signal. This alone fixes most horizon problems.
- **Lookahead depth** before committing — so the payoff is within the evaluated horizon.
- **Exploration bonus** (UCB-style) — keep *under-explored* nodes alive even at low current estimate.
- **Soft pruning** — deprioritize + keep revivable, rather than delete.

**Do standard algorithms handle it? Partially, and unequally:**
| Algorithm | Horizon-robust? |
|---|---|
| Greedy / plain beam | ❌ weakest — commits top-k, drops the rest. Mitigate: wider / stochastic / diverse beam, or a revival pool. |
| Depth-limited minimax | ✅ if searched deep enough — but needs the adversarial frame |
| **MCTS** | ✅ best for our single-agent case — deep rollouts back up value; UCB revisits under-explored branches |

**Design stance:** soft pruning + value estimates + exploration term. That property is exactly
what MCTS provides — so **beam is the debuggable v0, MCTS is the principled target.**

## v0 — build first
Best-first / beam over the tree, one simple objective, shallow depth. Standard (≈ "Tree of
Thoughts", Yao 2023), ~100 lines, debuggable. Accept its premature-pruning weakness but add the
cheap mitigations (diverse/stochastic beam + a small revival pool) so a promising-later line
isn't permanently lost.

## Future options (keep support-ready; don't build yet)
- **MCTS / PUCT (AlphaGo / MuZero):** principled explore/exploit + value backup. Drops in as a `SearchStrategy`; best fit for the horizon problem. The natural "if promising, go advanced" upgrade.
- **Adversarial two-player:** deliberately introduce an opponent — a monitor / "trip-sitter" critic, or a self-copy — and then **minimax / alpha-beta / self-play** become apt. Right for contexts about robustness, deception, red-vs-blue, or "can a steered self evade a monitor." A design *choice*, per context.
- **Multi-objective / Pareto:** optimize capability + introspection + welfare *jointly* (weighted scalar or Pareto frontier) rather than one number.
- **Learned value / policy net (the real AlphaGo move):** train a value model on tree outcomes to *guide* the search. This is a natural use of the **Tinker credits** (fine-tune a value/policy model) — ties the two threads together.

## Compute framing
Throughput bounds tree size: MLX-8-bit ≈ ~20 tok/s steered → **~3,000 generations/night**
(torch-MPS ~12 → ~1,700). So one night ≈ a beam of width ~5 × depth ~4 over a few compounds,
or a full compound×dose sweep. Pruning is partly *why* — the budget forces it, which is exactly
the tension with exploration.

## The real hard part
The search algorithm is the easy, standardized bit. **The objective (`evaluate`) is the
research** — "what makes one steered self-instance *better*?" (capability vs introspection vs
welfare) is the actual open question. Keep the abstraction clean so the objective can evolve
without rewriting the search.

## v0 build plan — wiring the engine to steering (the next step)
Turn the validated substrate + toy engine into one real end-to-end run (`self_med_search.py`):
- **`Node.state`** = the set of active `(compound, dose)` — root `()` = unsteered baseline.
- **`expand(node)`** = add one not-yet-active compound → a child. So depth-1 = single compounds, depth-2 = pairs, etc. — a **combinatorial tree over compound *combinations***. (The "sacrifice" case can appear: a compound weak alone but good in combination.)
- **v0 `Objective` = task accuracy (capability):** "which steered self-state best solves task X?" Measurable, genuinely interesting (emotion → capability), and on-thread (the paper's task-eval arm). Reuse the vendor `problems/gsm8k.py` harness — v0 uses a small hardcoded GSM8K-style set for robustness; upgrade to the real dataset later.
- **First run:** shallow beam (width 2, depth 2) over a few compounds on **Qwen3-8B-8bit (MLX)**, multi mode, calibrated dose → a **scored tree of compound-combos vs task performance**, with the unsteered baseline as reference.

**Deferred deliberately — the research heart:** an **introspection-gain** objective ("does searching over steered copies of itself *improve* the model's self-knowledge?"). Needs the probe battery + (eventually) position-indexed steering. Because the objective is pluggable, it swaps in **without touching the search**.
