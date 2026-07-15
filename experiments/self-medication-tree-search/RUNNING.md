# Running the self-medication search — backends & environments

## Secrets (env-driven — no keys or paths in the repo)
This repo is public; keys/tokens are resolved from the environment, never hardcoded. Provide a token
inline, or point at a local file by path (recommended). A gitignored `.env` holds the paths:

```bash
# .env (gitignored) — see .env for the template
ANTHROPIC_API_KEY_FILE=/abs/path/to/anthropic-key.txt   # for the Sonnet-5 judge (--objective judge/both)
HF_TOKEN_FILE=/abs/path/to/hf-read-token.txt            # for gated datasets (GPQA)
# then, before running:
set -a; source .env; set +a
```
Inline alternatives: `ANTHROPIC_API_KEY`, `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`.

The search core (`tree_search.py`, `self_med_search.py`, the objective) is **backend-agnostic**.
Steering is applied by a pluggable backend selected with `--backend`. Default is **`auto`**, which
detects the environment and **always logs what it resolved to** (so every run is self-documenting).

> For real/reported runs, prefer pinning `--backend` explicitly — the environment does *not* uniquely
> determine the backend (Mac = MLX **or** torch-MPS; CUDA = torch **or** vLLM), and a result should be
> attributable to a specific backend. `auto` is for convenience; the startup log records the truth either way.

## Backend × environment matrix

| `--backend` | Runs on | Env / image | Model default | Notes |
|---|---|---|---|---|
| `mlx`   | **Mac** (Apple GPU) | `.venv-mlx` (torch-free) | `mlx-community/Qwen3-8B-8bit` | Fastest on Mac; 8-bit. Needs the torch-free vector export (`.npz`). |
| `torch` | **Mac** (MPS) | `.venv` (torch) | `Qwen/Qwen3-8B` | Full precision; slower than MLX. Uses `library.pt` directly. |
| `torch` | **Cloud** (CUDA) | pod: `ghcr.io/veylansolmira/vllm-base` (or `caml-env`) | `Qwen/Qwen3-8B` | **Same code as Mac-torch** — `device` auto-detects `cuda`. |
| `vllm`  | **Cloud** (CUDA) | `vllm-base` pod + `vllm-lens` | — | ⚠️ **Not yet implemented** — throughput port (`vllm_drug_backend.py`). |

## How `auto` resolves
`torch` not importable (torch-free env) → **mlx** · CUDA present → **torch** · Mac with MLX installed → **mlx** · else → **torch** (MPS/CPU).
Startup always prints: `[backend] <name> device=<...> model=<...> mode=<...> (requested --backend <...>)`.

## Commands

```bash
# Mac, MLX (default when run from the mlx venv):
.venv-mlx/bin/python self_med_search.py --compounds focused calm curious --depth 2

# Mac, torch / MPS:
.venv/bin/python self_med_search.py --backend torch --compounds focused calm curious --depth 2

# Cloud, torch / CUDA (on the pod, after code + library.pt are synced):
python self_med_search.py --backend torch --compounds focused calm curious --depth 2
```

## What each environment needs
- **Mac MLX** (`.venv-mlx`): `mlx`, `mlx-lm`, and the torch-free vector export — run `export_library_torchfree.py` once in `.venv`.
- **Mac torch** (`.venv`): `torch`, `transformers`, and `vendor/llm-self-steering/src/hackday/drugs/library.pt`.
- **Cloud torch**: the experiment code + `library.pt` on the pod; the model downloads to the pod on first run. Launch the box with:
  ```bash
  runpod-launch launch --image ghcr.io/veylansolmira/vllm-base:latest \
    --ssh-key ~/.ssh/runpod_ed25519 --gpu powerful --container-disk 40 --volume 20 -y
  ```
  (RunPod SSH + image notes: see `~/.claude/.../feedback_runpod_ssh` memory; launcher lives in `~/Desktop/ai_dev/foundry`.)

## Steering spec (shared across all backends)
Layers **16–24**, each vector L2-normalized to magnitude **4.0**.
`--mode single` = probe layer only, norm-matched (subtler) · `--mode multi` = all extraction layers, raw add / stacks (stronger).
Per-compound doses come from `DEFAULT_DOSES`; effective dose = `--dose` × the compound's default scale.

## Still to do for a fully turnkey cloud run
1. **Deploy wrapper** — one command to `launch → rsync code + library.pt → run --backend torch → fetch results → tear down`.
2. **`vllm_drug_backend.py`** — the vLLM/`vllm-lens` backend for throughput on large searches (the `--backend vllm` slot).
3. (optional) extract the `SteeringBackend` ABC + `DEFAULT_DOSES` into a shared module so torch/mlx/vllm import one contract.

## Benchmark content policy (GPQA)
GPQA is contamination-gated: raw questions/answers must never reach a scrapeable or third-party location. This restricts *only raw content* — every result we present is fine.
- **Only scores/labels leave the machine** — node scores, the tree, aggregate accuracy, figures are free to commit/present.
- **Raw content stays local + gitignored** (`transcripts/`, `*_raw.json`, `run_*.log`).
- **Inference stays local/on-pod; the judge stays on generic probes** — never route GPQA content through an external API.
- Keep the GPQA canary string intact; never quote raw items in commits, sheets, slides, blogs, or artifacts.
(Does not apply to toy / gsm8k / mmlu_pro — open datasets.)
