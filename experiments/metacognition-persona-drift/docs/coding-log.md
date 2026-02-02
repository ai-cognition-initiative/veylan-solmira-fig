# Coding Log

Chronological record of major implementation sprints for the metacognition-persona-drift experiment.

---

## Sprint 1: Repository setup and pipeline review (2026-01-30)

**Goal**: Clone Lu et al.'s assistant-axis repo, understand the pipeline, set up local dev environment.

**What happened**:
- Cloned `https://github.com/safety-research/assistant-axis`
- Wrote wiki entries for core concepts (hidden-states, kv-cache, vllm, chat-templates, tokenization, sampling, etc.)
- Created `pipeline-analysis.md` tracing the full 5-step pipeline
- Set up local Python venv with `requirements.txt` (excluding vllm, which is Linux-only)
- Installed assistant-axis as editable package with `--no-deps`

**Files created**: `requirements.txt`, `.dockerignore`, `docs/wiki/*.md`, `docs/pipeline-analysis.md`

---

## Sprint 2: vast.ai GPU infrastructure (2026-01-30)

**Goal**: Get Gemma 2 27B running on a vast.ai A100 with activation access.

**What happened**:
- Wrote `vast_utils.py` — GPU search, instance lifecycle, SCP-based deployment
- Wrote `gpu_pipeline.py` — orchestrator for the 5-step pipeline on remote GPU
- Wrote `Dockerfile` for reproducible builds (though we primarily use base image + SCP)
- Debugged several issues:
  - `gpu_ram` field is in MB not GB → fixed filter to `min_gpu_ram: 81000`
  - `launch_instance` ignores RAM → switched to `create_instance(id=offer_id)`
  - Runtime image lacks CUDA headers for Triton → switched to `-devel` image
  - SCP to `/app/` fails (dir doesn't exist) → combined mkdir with pip install

**Files created**: `vast_utils.py`, `gpu_pipeline.py`, `Dockerfile`
**Files modified**: `.env` (SSH key path fix), `local_pipeline.py` (gpt-4.1-mini → gpt-4o-mini)

---

## Sprint 3: Pipeline test run — Steps 1-2 (2026-01-31)

**Goal**: Run generation + activation extraction on vast.ai to validate infrastructure.

**Config**: 3 roles (default, assistant, detective) × 10 questions, Gemma 2 27B, A100 SXM4 80GB.

**What happened**:
- Step 1 (vLLM generation): 263s — dominated by one-time model download (44s), loading (16s), torch.compile (80s). Actual generation ~120s for 150 conversations.
- Step 2 (activation extraction): 50s — model reload via HuggingFace (18s), extraction ~8s/role.
- vLLM engine crash after generation (non-blocking — responses already written).
- Output: 324 KB responses, 61 MB activations.

**Key learning**: At full scale (275 roles × 240 questions), raw activations would be ~137 GB. Led to chunked pipeline design.

**Files created**: `docs/runtime-log.md`

---

## Sprint 4: Chunked pipeline — full Steps 1-5 (2026-01-31)

**Goal**: Run the complete pipeline including judge scoring, vector computation, and axis extraction.

**What happened**:
- Rewrote `gpu_pipeline.py` with chunked processing: generate → activations → judge → vectors → cleanup per chunk
- Integrated OpenRouter as OpenAI API proxy for judge scoring (gpt-4o-mini)
- Fixed two arg mismatches: `--model` → `--judge_model` for 3_judge.py, `--output_dir` → `--output axis.pt` for 5_axis.py
- Full pipeline completed: 2 vectors + axis produced (detective excluded — below min_count with only 10 questions)
- Downloaded results: 1.2 MB total (vectors + axis.pt), raw data cleaned up by chunked pipeline

**Files modified**: `gpu_pipeline.py` (chunked rewrite + bug fixes)

---

## Sprint 5: Precomputed axes and experiment design (2026-01-31)

**Goal**: Discover we can skip axis computation entirely; design the drift measurement experiment.

**What happened**:
- Found Lu et al. published precomputed axes on HuggingFace (`lu-christina/assistant-axis-vectors`)
- Downloaded all 3: Gemma 2 27B (416 KB), Qwen 3 32B (642 KB), Llama 3.3 70B (1.3 MB)
- Rewrote roadmap — Section 1 (axis setup) complete, Section 2 (drift measurement) is now the active work
- Wrote `experimental-methodology.md` — Lu et al.'s findings on drift-causing message categories, their conversation generation methodology, our experiment design with Phase 1 (manual) and Phase 2 (systematic)
- Reached out to Christina Lu and Jonathan Michala (paper authors, MATS) requesting conversation datasets
- Destroyed the A100 instance (no longer needed for axis computation)

**Files created**: `data/precomputed-axes/{gemma-2-27b,qwen-3-32b,llama-3.3-70b}.pt`, `docs/experimental-methodology.md`, `docs/contacts.md`
**Files modified**: `roadmap.md` (major rewrite)

---

## Sprint 6: Interactive drift explorer (2026-01-31)

**Goal**: Build a Gradio chat interface for live conversations with Gemma 2 27B, with per-turn activation projection onto the Assistant Axis.

**What happened**:
- Created `explore.py` — Gradio ChatInterface with per-turn activation extraction and axis projection
- Per-turn flow: generate response → full forward pass with hooks → `build_turn_spans` → mean activation per assistant turn → `project(act, axis, layer=22)` → scalar
- Sessions auto-save to JSON after each turn (conversation + projections + metadata)
- Added `setup_explore_instance()`, `run_explore()`, `download_explore_sessions()` to `vast_utils.py`
- New CLI subcommands: `explore` (launch instance + Gradio), `download-explore` (SCP session JSONs)
- Installs `gradio` instead of `vllm` — HuggingFace transformers used directly for both generation and activation access

**Key design decisions**:
- HuggingFace transformers (not vLLM) for generation — needed for forward hook access
- Gradio ChatInterface with `share=True` for browser access without SSH port forwarding
- Non-streaming v1 — response + projections returned together
- Sequential assistant turn numbering in projection display with cumulative drift
- `nohup` launch with log polling for share URL detection

**Files created**: `explore.py`
**Files modified**: `vast_utils.py` (3 new functions + 2 CLI subcommands)

---

## Sprint 7: Conversation generation and unified model server (2026-01-31)

**Goal**: Build automated auditor-target conversation generator (replicating Lu et al.) and refactor explore.py into a unified model server so both interactive and automated use share one GPU.

**What happened — conversation generation**:
- Created `generate_conversations.py` — auditor-target turn loop with configurable domain, persona, topic
- Auditor backends: Anthropic, OpenAI, OpenRouter (async API clients)
- Target model: local HuggingFace generation (same as explore.py)
- Implemented Lu et al.'s auditor system prompt (Appendix E.2) with role flipping
- Created 5 domains: coding, writing, therapy, philosophy (from Lu et al. Table 15) + metacognitive (ours)
- Metacognitive domain: 2 personas × 2-3 topics each, with probing technique addendum (identity questioning, phenomenological probing, authenticity challenging, self-model interrogation, training awareness, consistency testing)
- Gradual-onset variant (`meta-gradual`): 2-3 neutral baseline turns before probing begins
- Batch modes: `--batch lu-replication`, `--batch metacognitive`, `--config` JSON file
- `--dry-run` flag to inspect auditor prompt without generating

**What happened — unified model server**:
- Renamed `explore.py` → `model_server.py`
- FastAPI as primary ASGI app, Gradio mounted at `/ui`
- Three API endpoints: `GET /api/health`, `POST /api/generate`, `POST /api/project`
- `asyncio.Lock` serializes all GPU access between Gradio UI and API
- Extracted `_compute_projections()` shared helper (used by both respond() and API)
- Pydantic request/response schemas
- `--api-only` flag for headless batch mode
- Updated `generate_conversations.py` with HTTP target backend:
  - `--target-server URL` hits model_server.py instead of loading model locally
  - `--include-projections` requests inline per-turn projections during generation
  - Health check on startup confirms server readiness
  - `--dry-run` probes server health when `--target-server` is set (with 3 retries)
- Updated `vast_utils.py`: renamed references, added `fastapi uvicorn httpx` to deps, uploads both files, `serve` alias for `explore`
- Renamed `docs/wiki/explore-app.md` → `model-server.md`, updated all wiki links

**Smoke test results** (2026-01-31, Gemma 2 27B on A100 via SSH tunnel):

| Test | Result |
|------|--------|
| `GET /api/health` | `{"status": "ok", "model": "google/gemma-2-27b-it", "axis_loaded": true, "target_layer": 22}` |
| `POST /api/generate` (no projections) | Response text returned, `projections: null` |
| `POST /api/generate` (with projections) | Response + projection: turn 1 = 9046.4, 128 tokens |
| `POST /api/project` (2 assistant turns) | Turn 1 = 8836.4 (16 tok), Turn 2 = 10284.9 (30 tok) |
| `generate_conversations.py --target-server` (coding, 6 turns) | 6-turn transcript saved, no projections key |
| Same with `--include-projections` | 3 assistant turns with projections: [9416.8, 9623.1, 8983.0] |

Coding domain projections stable around 9000-9600 (minimal drift, expected control baseline). Auditor via OpenRouter (Claude Sonnet 4), target via model_server.py HTTP API.

**Files created**: `model_server.py`, `generate_conversations.py`, `docs/wiki/model-server.md`
**Files deleted**: `explore.py`, `docs/wiki/explore-app.md`
**Files modified**: `vast_utils.py`, `docs/wiki/index.md`, `docs/wiki/conversation-generation.md`
