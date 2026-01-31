# Runtime Log

Observations from actual pipeline runs. Updated as we go.

---

## Test Run: 3 roles x 10 questions (2026-01-31)

**Config**: Gemma 2 27B, 1x A100 SXM4 80GB, $0.936/hr, vast.ai

### Instance setup
- Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (devel needed — Triton compiles CUDA kernels at runtime, requires dev headers)
- Dep install (vllm + assistant-axis): ~2 min via SSH
- Code upload (assistant-axis tar + gpu_pipeline.py): ~10s (1.5 MB tar)

### Step 1: Generate Responses (vLLM)
- **Total**: 263s (4.4 min) — dominated by one-time startup costs
- Model download: 44s (weights cached after first run)
- Model loading: 16s (50.72 GiB into GPU memory)
- torch.compile warmup: 80s (Triton kernel compilation, cached after first run)
- KV cache: 22.13 GiB free after model load, supports ~30 concurrent 2048-token requests
- Actual generation: ~120s for 150 conversations (3 roles x 5 instructions x 10 questions)
- vLLM engine crashed after generation completed (`EngineCore_DP0 died unexpectedly`) but all responses were already written
- Output: 3 JSONL files, 324 KB total

### Step 2: Extract Activations
- **Total**: 50s
- Model reload (HuggingFace, not vLLM): 18s for checkpoint shards
- Extraction: ~8s per role (50 conversations x 46 layers each)
- Output: 3 `.pt` files, 61 MB total (~21 MB per role)

### Data sizes (3 roles, 10 questions)
| Data | Size | Per role |
|------|------|----------|
| Responses (JSONL) | 324 KB | ~107 KB |
| Activations (.pt) | 61 MB | ~21 MB |
| **Total** | **62 MB** | |

### Extrapolated sizes (275 roles, 240 questions)
| Data | Estimate | Notes |
|------|----------|-------|
| Responses | ~30 MB | 275 x ~107 KB |
| Activations | ~5.7 GB | 275 x ~21 MB (but 24x more conversations per role → larger tensors) |
| **Total** | **~6-10 GB** | Activation files dominate |

The activation files store tensors of shape `(n_conversations, n_layers, hidden_dim)` = `(1200, 46, 3584)` per role at full scale. The test run had 50 conversations per role (5 instructions x 10 questions) producing 21 MB each. At full scale (1200 conversations), each role's activation file would be ~500 MB, making the total ~137 GB. This is the main storage and transfer cost.

**Implication**: For the full 275-role run, we should run Steps 3-4 (judge + vectors) on the GPU instance too, reducing each role to a single mean vector before download. The per-role vectors are shape `(46, 3584)` — about 650 KB each, totaling ~175 MB for all 275 roles.

### Lessons learned
- **A100 40GB vs 80GB**: vast.ai `gpu_ram` field is in MB. Must filter `gpu_ram>=81000` to get 80GB cards. Earlier runs landed on 40GB cards and OOM'd.
- **Runtime vs devel image**: `pytorch/pytorch:...-runtime` lacks CUDA dev headers. Triton (used by vLLM) needs them to compile kernels. Use `-devel`.
- **SSH timing**: vast.ai instances need a few seconds after "running" status before SSH accepts connections. Build in retries.
- **vLLM crash**: Engine core died after generation completed. Non-blocking for our pipeline since Step 2 uses HuggingFace directly, not vLLM. Worth monitoring on longer runs.

### Cost
- Instance time: ~6 min active use
- Rate: $0.936/hr
- **Cost for this test**: ~$0.09

---

## Full Pipeline Test: 3 roles x 10 questions, Steps 1-5 (2026-01-31)

**Config**: Same instance as above (warm caches). Chunked pipeline with OpenRouter judge.

### Chunked processing

The pipeline processes roles in chunks: generate → activations → judge → vectors → cleanup per chunk. The "default" role is its own chunk (chunk 1), then remaining roles are grouped.

| Chunk | Roles | Step 1 (Generate) | Step 2 (Activations) | Step 3 (Judge) | Step 4 (Vectors) | Total |
|-------|-------|--------------------|----------------------|----------------|------------------|-------|
| 1 | default (1 role) | 84s | 32s | ~4s | ~1s | ~120s |
| 2 | assistant, detective (2 roles) | 102s | 38s | 7s | 2s | 150s |
| **Step 5** | Compute axis from vectors | | | | | **~2s** |
| **Pipeline total** | | | | | | **~275s** |

### Per-step observations (warm caches)

**Step 1 (Generate)**: vLLM startup dominates each chunk: model load (~11s from cache) + torch.compile (~18s from cache) + engine warmup (~31s) = ~60s fixed overhead per chunk. Actual generation for 50-100 conversations is 25-45s. The vLLM engine crash (`EngineCore_DP0 died unexpectedly`) still occurs every chunk but all responses are already written — non-blocking.

**Step 2 (Activations)**: HuggingFace model reload each chunk (~16s). Extraction rate: ~8-9s per role (50 conversations, 46 layers).

**Step 3 (Judge)**: 50 conversations scored in ~3.5s per role via OpenRouter (gpt-4o-mini). Very fast — API-bound, not compute-bound.

**Step 4 (Vectors)**: Negligible — 1-2s for all roles in chunk. Note: detective role had only 46 score=3 samples (below 50 minimum), so no vector was produced. This is expected with only 10 questions/role; 240 questions should easily exceed the threshold.

**Step 5 (Axis)**: Instantaneous with 2 vectors. Axis shape: `(46, 4608)`.

### Outputs

| File | Size | Description |
|------|------|-------------|
| `vectors/default.pt` | 416 KB | Mean vector for default assistant |
| `vectors/assistant.pt` | 416 KB | Mean vector for "assistant" role |
| `axis.pt` | 416 KB | Assistant axis (default - mean of role vectors) |
| **Total downloaded** | **~1.2 MB** | Chunked cleanup deleted all raw data |

### Full-scale estimates (275 roles, 240 questions)

Based on observed per-step rates, with significant uncertainty marked.

**Chunking**: 1 default chunk + 11 chunks of 25 roles = 12 chunks total.

| Step | Per-chunk estimate | 12 chunks | Notes |
|------|--------------------|-----------|-------|
| Generate (vLLM) | ~60s startup + generation | **~?** | Generation throughput unclear at scale (see below) |
| Activations | 16s reload + 25 roles × ~8s = ~216s | **~43 min** | Linear with #roles |
| Judge (OpenRouter) | 25 roles × 1200 convs × rate | **~1-2 hr** | Rate-limited by API, not compute |
| Vectors | ~25s | **~5 min** | Negligible |
| Axis | ~2s | **2s** | One-time, end of pipeline |

**Generation throughput uncertainty**: The test processed 150 conversations in ~120s (~1.25 conv/s) which was likely underutilizing vLLM's 30-concurrent-request capacity. At full scale (30,000 conversations per chunk), vLLM should saturate better. Needs a larger test (e.g., 25 roles × 10 questions) to get a reliable throughput estimate before committing to the full run.

**Rough cost bracket** (at $0.936/hr):
- Optimistic (generation scales well): 4-8 hours → $4-8
- Conservative (generation is the bottleneck): 12-24 hours → $12-24
- Judge (OpenRouter gpt-4o-mini): 330,000 conversations × ~$0.0001/conv ≈ $33

### Bug fixes during this run
- `gpu_pipeline.py`: `--model` → `--judge_model` for `3_judge.py`
- `gpu_pipeline.py`: `--output_dir` → `--output axis.pt` for `5_axis.py`
