# Self-Medication Experiment — Mac configuration (no vLLM)

**Context.** Dual config: use the Mac for as much as it can handle; push the rest to the cloud. Each experiment gets its **own env** — this one wants `transformers>=5.3` + Inspect, which conflicts with the repo's `transformers<5` Gemma pipeline. That's fine and intended: **different experiment → different env.** This doc is the **Mac side**; a separate session handles the cloud (vLLM) side.

**The one hard cloud dependency: vLLM.** vLLM needs CUDA/Linux — it does not run usefully on macOS. Their steering layer (`vllm-lens`) is a *vLLM plugin*, so **anything that goes through vLLM is cloud-only.** Everything else runs on the Mac. That single fact defines the whole split below.

## What runs where

| Component | Mac (no vLLM) | Cloud (vLLM) |
|---|---|---|
| Model inference | ✅ small models (**Qwen3-8B**) via transformers-MPS / MLX | ✅ **Qwen3-32B** at throughput |
| Activation steering | ✅ transformers forward-hooks (port of vllm-lens's role) | ✅ vllm-lens plugin |
| Inspect harness (tasks / solvers / scorers) | ✅ via Inspect's `hf/` provider | ✅ via `vllm/` provider |
| Vector extraction (`extract.py`) | ✅ on a small model; ⚠️ 32B → cloud | ✅ any model |
| Story generation (`generate_stories.py`) | ✅ (Claude API) | ✅ |
| LLM-judge scoring | ✅ (Anthropic API) | ✅ |
| Tree-search orchestration (our new code) | ✅ pure Python logic | ✅ |
| Analysis / plotting / stats (`.eval` transcripts) | ✅ | — |
| Full free-play / task-eval sweeps at scale | ⚠️ tiny only | ✅ |

**Rule of thumb:** *develop, prototype, and analyse on the Mac; run the big model + high-throughput evals in the cloud.*

## Mac environment (dedicated to this experiment)

Do **not** install `vllm` or `vllm-lens` on the Mac (they won't build/run). Fresh, isolated env:

```bash
cd experiments/self-medication-tree-search
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install \
  inspect-ai inspect-evals \
  "transformers>=5.3" torch \
  datasets huggingface-hub \
  anthropic \
  pandas matplotlib statsmodels \
  nnsight          # or TransformerLens — clean activation intervention in transformers
# optional, faster Mac inference:
# uv pip install mlx mlx-lm
```
Python 3.12 (their requirement). Torch uses the **MPS** backend automatically on Apple silicon.

## Steering without vLLM

`vllm-lens` intercepts activations *inside* vLLM. Reproduce that on the Mac with **transformers forward-hooks**: add the steering ("compound") vector to the residual stream at layers **16–24**, L2-normalised to magnitude **4.0** (their recipe). Two clean routes:
- **Reuse the Assistant-Axis steering code already in this repo** — it already does transformers-based activation addition; adapt it to add a compound vector at the target layers. *(Preferred — maximal reuse.)*
- Or use **nnsight / TransformerLens** for the hook plumbing.

Their committed vectors live at `vendor/llm-self-steering/src/hackday/drugs/library.pt` and `library_qwen3_32b.pt` — **Qwen3-specific**, so directly usable if you run Qwen3; otherwise **re-extract** for the Mac model via their `drugs/extract.py` + `generate_stories.py` recipe (story-gen is a Claude API call → Mac-fine).

## What you can actually do on the Mac now
1. **Smoke test (do this first):** load **Qwen3-8B** (transformers-MPS or MLX), apply one steering vector via a forward-hook, generate → confirm the output visibly shifts. Smallest proof the steering loop works without vLLM.
2. Adapt the **`take_drug` tool + Inspect task defs** from `vendor/llm-self-steering/src/hackday/` to run against the local model via Inspect's `hf/` provider — drop the `vllm-lens` import, wire the hook-based steering.
3. Build the **tree-search-over-self-instances** orchestration (pure Python; model-agnostic — runs anywhere).
4. **Analyse** cloud `.eval` transcripts locally (pandas / matplotlib / statsmodels).

## Deferred to the cloud (other session)
vLLM + vllm-lens at scale, **Qwen3-32B**, full free-play + task-eval sweeps, vector extraction on 32B — anything needing GPU throughput. Runs on vast.ai via the repo's `vast_utils.py`.

## Handoff notes
- The Mac env is **`.venv`** in this experiment folder (per-experiment-env design; separate from the repo's transformers<5 / Gemma env in the metacognition experiment). The cloud/vLLM path runs on vast.ai — not a local env — so there's no same-folder clash and no `-mac` suffix is needed. *(Reconciled 2026-07-05: an earlier draft said `.venv-mac`; the built env is `.venv`.)*
- Read their code from `vendor/`. **`llm-self-steering` is unlicensed → reimplement/adapt, don't copy verbatim.** `vllm-lens` is MIT (safe to use directly, but it's vLLM-only so it doesn't help the Mac path).
- **Design goal for portability:** the Mac and cloud paths should share the *same* Inspect task definitions and the *same* steering **interface**, differing only in the backend (transformers-hook vs vllm-lens) and model size. Put the steering call behind one function so both configs use it — that's what makes "use the laptop as much as I want, cloud when I must" actually work.
