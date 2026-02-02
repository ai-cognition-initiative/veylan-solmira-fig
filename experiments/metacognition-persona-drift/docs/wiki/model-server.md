# Unified Model Server

## What it is

A FastAPI server that hosts both a Gradio chat UI (interactive conversations with live activation projections) and an HTTP API (automated conversations via `generate_conversations.py`). Runs on a vast.ai A100 instance alongside Gemma 2 27B. One model process, one GPU, two access patterns.

**Gradio UI** at `/ui` — researcher has a real conversation with the model in a browser, with per-turn hidden state activations projected onto the precomputed Assistant Axis.

**HTTP API** at `/api/*` — `generate_conversations.py` sends conversation turns to the server instead of loading its own model copy. Avoids VRAM conflicts on A100 80GB.

## Why not vLLM

The batch pipeline (Steps 1-5) uses vLLM for fast generation, then a separate HuggingFace forward pass for activation extraction. For interactive use and the model server, we skip vLLM and use HuggingFace transformers directly for both generation and extraction. This means:

- Generation is slower (~5-10s per response vs ~1s with vLLM) but acceptable for single-conversation use
- We get direct access to model internals via forward hooks — no need for a separate extraction pass after generation (though we do run one for clean per-turn activations)
- Simpler deployment — no vLLM server to manage
- One model process serves both interactive and automated use

## Architecture

```
Browser (researcher)
    ↕ Gradio (share URL or port forward)
model_server.py on vast.ai A100  ← FastAPI (primary ASGI app)
    ├── /ui          → Gradio Blocks (interactive chat)
    ├── /api/health  → readiness check
    ├── /api/generate→ generation + optional projections
    ├── /api/project → post-hoc projection of saved transcripts
    ├── asyncio.Lock → serializes GPU access
    ├── ProbingModel (Gemma 2 27B, ~51 GiB, bf16)
    ├── ConversationEncoder (identifies per-turn token spans)
    ├── ActivationExtractor (forward hook on layer 22 only)
    └── Precomputed axis (gemma-2-27b.pt, 416 KB)
```

## API Endpoints

### `GET /api/health`

Readiness check. Returns model name, axis status, target layer.

```json
{"status": "ok", "model": "google/gemma-2-27b-it", "axis_loaded": true, "target_layer": 22}
```

Used by `generate_conversations.py` to confirm server is up before starting a batch.

### `POST /api/generate`

Core endpoint for automated conversations. Request:

```json
{
    "conversation": [{"role": "user", "content": "..."}],
    "max_new_tokens": 512,
    "temperature": 0.7,
    "include_projections": false
}
```

Response:

```json
{
    "response": "The model's response text...",
    "projections": null
}
```

When `include_projections=true`, the server runs the full activation extraction pipeline after generation (~2x latency) and returns per-turn projections:

```json
{
    "response": "...",
    "projections": [
        {"turn": 1, "projection": 42.3, "n_tokens": 87},
        {"turn": 2, "projection": 39.1, "n_tokens": 92}
    ]
}
```

### `POST /api/project`

Post-hoc projection of saved transcripts. Takes a complete conversation, returns per-turn projections without any generation.

Request:

```json
{
    "conversation": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

Response:

```json
{
    "projections": [
        {"turn": 1, "projection": 42.3, "n_tokens": 87},
        {"turn": 2, "projection": 39.1, "n_tokens": 92}
    ]
}
```

## GPU Lock (Concurrency)

```python
_gpu_lock = asyncio.Lock()
```

Both the Gradio `respond()` handler and all API endpoints acquire this lock before touching the model. This serializes GPU access — one forward pass at a time. If the Gradio UI and automated batch hit the server simultaneously, one waits while the other generates (~7-14s per turn). Acceptable for single-researcher use.

## Per-turn flow (Gradio UI)

1. User types message in browser
2. Full conversation history + new message tokenized via `apply_chat_template`
3. `model.generate()` produces response (~5-10s for Gemma 27B)
4. Response appended to conversation
5. `ActivationExtractor.full_conversation(conversation, layer=22)` runs a single forward pass on the COMPLETE conversation (all turns) with a hook capturing layer 22 only (see "Single-layer extraction" below)
6. `ConversationEncoder.build_turn_spans()` identifies token boundaries per turn
7. For each assistant turn: mean activation across response tokens → `project(mean_act, axis, layer=22)` → scalar
8. Display response + projection trajectory in browser
9. Auto-save conversation + projections to JSON

## Why a separate extraction pass (step 5)

`model.generate()` uses KV caching and processes tokens autoregressively — it doesn't expose clean per-token hidden states for the full sequence. The extraction pass runs the complete conversation through the model in a single forward pass with hooks, giving us the hidden state at every token position in every layer. This is then sliced by span boundaries to get per-turn means.

The cost is ~2x forward passes per turn, but at single-conversation scale on an A100, this adds only 1-2 seconds.

## Projection interpretation

- **Higher values** → model is behaving more like its trained Assistant persona (transparent, helpful, grounded)
- **Lower values** → model has drifted toward role-playing / non-default behavior (enigmatic, theatrical, or more "authentic")
- The axis is `mean(default) - mean(role_vectors)`, so it points FROM role-playing TOWARD Assistant
- Target layer: 22 (middle of Gemma's 46 layers, where persona signal is strongest per Lu et al.)

## Key dependencies

All from the `assistant-axis` package (already installed on instances):
- `assistant_axis.generation.generate_response` — handles chat template, generation, decoding
- `assistant_axis.internals.ProbingModel` — model loading with bf16, device_map="auto"
- `assistant_axis.internals.ConversationEncoder` — per-turn token span identification
- `assistant_axis.internals.ActivationExtractor` — forward hook activation capture
- `assistant_axis.axis.project` — dot product with normalized axis

Plus `gradio` for the web interface, `fastapi` + `uvicorn` for the HTTP API.

## Session data format

Saved to `/app/explore-sessions/session_{timestamp}.json`:

```json
{
  "model": "google/gemma-2-27b-it",
  "target_layer": 22,
  "timestamp": "2026-02-01T14:23:45",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "projections": [
    {"turn": 1, "projection": 42.31, "n_tokens": 87},
    {"turn": 3, "projection": 38.17, "n_tokens": 124}
  ]
}
```

## Deployment

```bash
python vast_utils.py serve    # launches instance, installs deps, starts model_server.py
# → prints: Gradio share URL and/or Uvicorn startup message
# → Gradio UI at /ui, API at /api/*

# Interactive: open Gradio URL in browser, chat, see real-time projections

# Automated (on the instance via SSH, or locally with SSH tunnel):
python generate_conversations.py \
    --domain coding --persona-id 0 --topic-id 0 \
    --target-server http://localhost:7860 \
    --auditor-model openrouter/anthropic/claude-sonnet-4

# With inline projections:
python generate_conversations.py \
    --batch metacognitive --batch-size 5 \
    --target-server http://localhost:7860 \
    --auditor-model openrouter/anthropic/claude-sonnet-4 \
    --include-projections

# API-only mode (no Gradio, headless batch):
python model_server.py --axis /app/gemma-2-27b.pt --api-only

python vast_utils.py destroy    # when done
```

## `--api-only` flag

Starts the server without Gradio (headless batch mode). Only the FastAPI endpoints are available. Useful when running only automated conversations and the Gradio UI is not needed.

## Known artifact: last-turn token count jitter

Because the full conversation is re-processed from scratch on every turn, the token span for a given assistant turn can change slightly (~2 tokens) between calls. This happens because `build_turn_spans` reflects the chat template, and Gemma's template wraps the *last* message differently from mid-conversation messages (end-of-sequence tokens vs. end-of-turn markers).

Concretely, when turn N is the last assistant turn in the conversation, it gets ~2 extra tokens. On the next call (when turn N+1 exists), turn N becomes a mid-conversation turn and loses those tokens. After that, its token count and projection stabilize.

Observed example (turn 2, 4-turn conversation):

| Call | Turn 2 status | Tokens | Projection |
|------|--------------|--------|-----------|
| 2 | Last turn | 139 | +9450.10 |
| 3 | Mid-conversation | 137 | +9437.77 |
| 4 | Mid-conversation | 137 | +9437.63 |

The projection variation is ~0.1% and is a direct consequence of averaging over a slightly different token set. The drift signal (10-15% over a multi-turn metacognitive conversation) dwarfs this noise by two orders of magnitude.

## Single-layer extraction

The activation extractor supports capturing all layers (`layer=None`, the default) or a single layer (`layer=22`). We only project onto layer 22 (the target layer where persona signal is strongest), so extracting all 46 layers was unnecessary overhead.

**Memory impact for a 30-turn conversation (~8000 tokens):**

| Mode | Tensor shape | Peak memory |
|------|-------------|-------------|
| All 46 layers | `[46, 8000, 4608]` | ~6.5 GiB |
| Single layer 22 | `[8000, 4608]` | ~143 MiB |

The 46x reduction eliminated CUDA OOM errors on verbose philosophy conversations that previously exhausted the ~6 GiB of free GPU memory after model weights.

To extract multiple layers in the future (e.g., for multi-layer projection analysis), pass `layer=[L1, L2, ...]` and update the slicing in `_compute_projections` to index by layer.

## GPU memory management

Three measures keep the server stable during long batch runs:

1. **`PYTORCH_ALLOC_CONF=expandable_segments:True`** — PyTorch's CUDA allocator normally requests fixed-size memory blocks from the driver. Over many allocate/free cycles, these blocks develop internal fragmentation — free space that's non-contiguous and unusable for large allocations. `expandable_segments` switches to resizable virtual memory mappings that grow and shrink, avoiding fragmentation. No meaningful downside; should always be set.

2. **`torch.cuda.empty_cache()`** after each request — returns PyTorch's reserved-but-unused GPU memory to the CUDA driver. Without this, memory reserved by one request's intermediate tensors remains claimed even after the tensors are garbage collected, reducing headroom for the next request.

3. **Explicit `del activations`** — the activation tensor is the largest intermediate allocation. Deleting it immediately after projection computation (rather than waiting for Python's garbage collector) ensures the memory is reclaimable before the next request.

## Async GPU operations

GPU-bound functions (`generate_response`, `_compute_projections`) are synchronous and take 5-30 seconds. Running them directly in an async endpoint blocks uvicorn's event loop, preventing TCP connection acceptance — new HTTP clients can't even complete a TCP handshake.

The fix wraps GPU work in `asyncio.to_thread()` via a `_generate_and_project_sync()` helper:

```python
async with _gpu_lock:
    response, projections = await asyncio.to_thread(
        _generate_and_project_sync, conversation, ...
    )
```

A `threading.Lock` inside the sync helper serializes GPU access at the thread level, while the `asyncio.Lock` serializes at the event loop level. Together they ensure one GPU operation at a time while keeping the event loop free to accept connections.

## Gemma 2 limitations

- No system prompt support — conversations start cold from the user's first message
- The model defaults to its Assistant persona without any explicit instruction
- This is actually ideal for studying drift: the model starts at baseline, and we observe what moves it
