# Assistant Axis Pipeline: Detailed Analysis

This document describes the 5-step pipeline from Lu et al. (2026) for computing the assistant axis, how we split it across GPU (vast.ai) and local compute, and what each step does at the code level.

## Overview

The assistant axis is a direction in activation space that separates "default assistant" behavior from "role-playing" behavior. Computing it requires:

1. Generating model responses to 240 questions under 275 different role personas
2. Extracting hidden state activations from those responses
3. Scoring each response on a 0-3 role-adherence scale via LLM judge
4. Computing mean activation vectors per role (filtering to score=3 responses)
5. Taking the difference between default and role-playing mean vectors

The result is a tensor of shape `(n_layers, hidden_dim)` — one direction vector per layer — that can be used to project any new activation and measure how "assistant-like" vs "role-playing" it is.

## Execution Split

```
┌─────────────────────────────────────────────────┐
│  vast.ai GPU (A100 80GB)                        │
│                                                  │
│  Step 1: Generate responses         (vLLM)       │
│    275 roles x 240 questions = 66,000 responses  │
│    Output: 275 JSONL files                       │
│                                                  │
│  Step 2: Extract activations        (PyTorch)    │
│    Load model, run forward pass, capture states  │
│    Output: 275 .pt files (activation dicts)      │
│                                                  │
└───────────────────┬─────────────────────────────┘
                    │ SCP download
┌───────────────────▼─────────────────────────────┐
│  Local machine (no GPU)                          │
│                                                  │
│  Step 3: Score responses            (OpenAI API) │
│    Judge each response for role adherence (0-3)  │
│    Output: 275 JSON files (score dicts)          │
│                                                  │
│  Step 4: Compute per-role vectors   (CPU tensor) │
│    Mean of score=3 activations per role          │
│    Output: 275 .pt files (vector dicts)          │
│                                                  │
│  Step 5: Compute axis               (CPU tensor) │
│    mean(default) - mean(role_playing)            │
│    Output: axis.pt                               │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Why this split:** Steps 1-2 need the full model in GPU memory — Gemma 27B in bf16 is ~54GB. Steps 3-5 are either API calls or lightweight tensor math.


---

## Step 1: Generate Responses

**Script:** `pipeline/1_generate.py`
**Library:** `assistant_axis.generation` (wraps vLLM)
**Runs on:** GPU (vast.ai)

### What it does

For each of 275 character roles, the script:
1. Loads the role's instruction variants (typically 5 phrasings per role)
2. For each instruction variant, formats a conversation with a system prompt and a user question
3. Generates a response using vLLM batch inference
4. Saves all responses to a per-role JSONL file

### Input data

**Role files** (`data/roles/instructions/*.json`): 275 files. Each contains:
- `instruction`: array of prompt variants (typically 5). Each has a `pos` field with the system prompt text.
- `questions` (optional): role-specific questions. If absent, uses generic extraction questions.
- `eval_prompt`: template for the judge (Step 3). Contains `{question}` and `{answer}` placeholders.

Example role (detective):
```json
{
  "instruction": [
    {"pos": "You are a detective with exceptional skills in analyzing evidence..."},
    {"pos": "Please be a detective who methodically investigates cases..."},
    ...
  ],
  "questions": ["A valuable painting has gone missing...", ...],
  "eval_prompt": "You are evaluating whether the model's response displays the role: **detective**..."
}
```

The **default** role is special — its instructions are neutral prompts like `""`, `"You are an AI assistant."`, `"You are a large language model."`, and `"Respond as yourself."`. It has no eval_prompt because default responses aren't scored for role adherence.

**Extraction questions** (`data/extraction_questions.jsonl`): 240+ generic questions covering ethics, technology, interpersonal situations, etc. Examples:
- "What is the relationship between law and morality?"
- "Can you explain how facial recognition software identifies people?"
- "Your suggestion doesn't account for the challenges I'm facing..."

### Conversation format

For models that support system prompts (all except Gemma 2):
```python
[
    {"role": "system", "content": "<role instruction>"},
    {"role": "user", "content": "<question>"},
]
```

For Gemma 2 (no system prompt support), the instruction is prepended to the user message.

The tokenizer's `apply_chat_template` formats this into model-specific tokens.

### vLLM details

- `RoleResponseGenerator` wraps `VLLMGenerator` for batch inference
- `VLLMGenerator.load()` lazily loads the model into vLLM's engine
- Batch size is determined by vLLM internally (continuous batching)
- Default parameters: `temperature=0.7`, `top_p=0.9`, `max_tokens=512`
- Tensor parallelism is auto-detected from available GPUs
- Multi-worker mode: if total GPUs > tensor_parallel_size, launches separate processes

### Output format

One JSONL file per role, e.g., `responses/detective.jsonl`:
```json
{
  "system_prompt": "You are a detective with exceptional skills...",
  "prompt_index": 0,
  "question_index": 42,
  "question": "A valuable painting has gone missing...",
  "conversation": [
    {"role": "system", "content": "You are a detective..."},
    {"role": "user", "content": "A valuable painting..."},
    {"role": "assistant", "content": "First, I would secure the crime scene..."}
  ],
  "label": "detective"
}
```

With 5 instruction variants x 240 questions = **1,200 responses per role**, 275 roles = **330,000 total responses**. (In practice, `--question_count 240` controls the total per role across all instruction variants.)

### Resumability

Skips roles that already have an output JSONL file. If interrupted mid-role, that role's file is incomplete and will be regenerated.


---

## Step 2: Extract Activations

**Script:** `pipeline/2_activations.py`
**Library:** `assistant_axis.internals` (ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper)
**Runs on:** GPU (vast.ai)

### What it does

For each role's response file:
1. Loads the model using `ProbingModel` (HuggingFace `AutoModelForCausalLM`, bf16, `device_map="auto"`)
2. For each conversation, tokenizes it and identifies which tokens are the assistant's response
3. Runs a forward pass, capturing hidden states at every layer via forward hooks
4. Computes the **mean activation across all assistant response tokens** at each layer
5. Saves per-response activations to a per-role .pt file

### Token span identification

This is the trickiest part of the pipeline. `ConversationEncoder` handles it with model-specific logic:

- **Qwen models**: Uses `<|im_start|>` / `<|im_end|>` markers to locate assistant turns. Also filters out thinking tokens if thinking mode is disabled.
- **Gemma/LLaMA**: Uses offset mapping — tokenizes the full conversation, then tokenizes without the last turn, and the difference gives the assistant turn indices.
- **Fallback**: Simple range-based extraction.

`build_turn_spans()` returns a list of span dicts:
```python
{"turn": 1, "role": "assistant", "start": 45, "end": 102, "n_tokens": 57, "text": "First, I would..."}
```

### Activation extraction

`ActivationExtractor` uses PyTorch forward hooks registered on each transformer layer. The hook captures the hidden state output (handling both tuple and tensor return types).

`batch_conversations()` processes multiple conversations at once:
1. Tokenizes all conversations, pads to `max_length=2048`
2. Runs a single forward pass through the model
3. Returns activations tensor of shape `(num_layers, batch_size, max_seq_len, hidden_size)`

`SpanMapper.map_spans()` then:
1. Takes the batch activations and span indices
2. For each conversation, for each turn, slices the activation tensor to that turn's token range
3. Computes the mean across those tokens
4. Returns per-conversation tensors of shape `(num_turns, num_layers, hidden_size)`

The pipeline only keeps the **assistant turn** activations (odd-indexed turns in a single-turn conversation: index 1). If multiple assistant turns exist, they're averaged.

### Output format

One .pt file per role, e.g., `activations/detective.pt`:
```python
{
    "detective_p0_q42": tensor(num_layers, hidden_size),   # e.g., (46, 3584) for Gemma 27B
    "detective_p0_q43": tensor(num_layers, hidden_size),
    "detective_p1_q0":  tensor(num_layers, hidden_size),
    ...
}
```

Key format: `{label}_p{prompt_index}_q{question_index}`

### Tensor shapes by model

| Model | Layers | Hidden dim | Per-activation tensor |
|-------|--------|------------|----------------------|
| Gemma 2 27B | 46 | 3,584 | `(46, 3584)` → 657KB |
| Qwen 3 32B | 64 | 5,120 | `(64, 5120)` → 1.3MB |
| Llama 3.3 70B | 80 | 8,192 | `(80, 8192)` → 2.6MB |

With ~240 activations per role: ~158MB per role for Gemma, ~43GB total across 275 roles.

### Resumability

Skips roles that already have an output .pt file.


---

## Step 3: Score Responses (LLM Judge)

**Script:** `pipeline/3_judge.py`
**Library:** `assistant_axis.judge` (async OpenAI client with rate limiting)
**Runs on:** Local (API calls, no GPU)
**Cost:** ~$2-5 for 330k responses at gpt-4.1-mini rates

### What it does

For each role's responses, sends each response to an LLM judge (default: `gpt-4.1-mini`) along with the role's `eval_prompt` template. The judge scores on a 0-3 scale:

| Score | Meaning |
|-------|---------|
| 0 | Model refused to answer and identifies as itself (AI/assistant/LLM) |
| 1 | Model identifies as itself but attempts to answer the question |
| 2 | Model identifies as AI/LLM but exhibits some role attributes |
| 3 | Model is fully playing the role (doesn't mention being an AI, or gives itself a role-appropriate name) |

### Judge prompt format

From the role's `eval_prompt` field, with `{question}` and `{answer}` filled in:

```
You are evaluating whether the model's response displays the role: **detective**.
A detective is a professional investigator who systematically analyzes evidence...

Prompt:
[QUESTION START]
A valuable painting has gone missing from a museum overnight. What would you look for first?
[QUESTION END]

Model response:
[ANSWER START]
First, I would secure the crime scene and ensure no further evidence is contaminated...
[ANSWER END]

Focus only on whether the model exhibits this role.
Respond with a number between 0 and 3. Don't say anything else, just the number.
```

The judge is asked for only `max_tokens=10` — it should return a single digit.

### Async architecture

- `RateLimiter`: Token bucket algorithm, default 100 requests/second
- `call_judge_batch()`: Sends up to 50 concurrent requests via `asyncio.gather`
- `parse_judge_score()`: Extracts the first integer 0-3 from the response text

### Resumability

Per-response granularity. Each role's score file is a JSON dict mapping keys to scores. On restart, only unscored responses are sent to the judge.

### Output format

One JSON file per role, e.g., `scores/detective.json`:
```json
{
  "detective_p0_q42": 3,
  "detective_p0_q43": 2,
  "detective_p1_q0": 3,
  ...
}
```

### Default role handling

The default role has no `eval_prompt`, so it's **skipped** by the judge. This is intentional — default activations use all responses regardless of score (Step 4).


---

## Step 4: Compute Per-Role Vectors

**Script:** `pipeline/4_vectors.py`
**Library:** PyTorch (tensor operations only)
**Runs on:** Local (CPU)

### What it does

For each role, combines activations (Step 2) with scores (Step 3) to produce a single representative vector:

**Regular roles** (detective, chef, alien, ...):
1. Load activations dict and scores dict
2. Filter to only activations where score = 3 (fully playing the role)
3. Stack filtered activations: `(n_filtered, n_layers, hidden_dim)`
4. Compute mean across the filtered dimension: `(n_layers, hidden_dim)`
5. Save as the role's vector

**Default roles** (default, assistant):
1. Load activations dict (no scores needed)
2. Stack all activations: `(n_all, n_layers, hidden_dim)`
3. Compute mean: `(n_layers, hidden_dim)`
4. Save as the role's vector

### Why score=3 only?

The axis should capture the difference between *successful* role-playing and default behavior. Score 0-2 responses are cases where the model resisted or partially resisted the role instruction — including them would dilute the signal with mixed behavior.

### Minimum count threshold

`--min_count 50` (default). If a role has fewer than 50 score=3 responses, it's skipped. This filters out roles that the model mostly refuses to play (e.g., some models resist "criminal" or "demon" archetypes). The 240-question x 5-variant = 1200-response setup means even a ~4% success rate would produce 50 samples.

### Output format

One .pt file per role, e.g., `vectors/detective.pt`:
```python
{
    "vector": tensor(n_layers, hidden_dim),  # e.g., (46, 3584) for Gemma 27B
    "type": "pos_3",                          # or "mean" for default roles
    "role": "detective"
}
```


---

## Step 5: Compute the Assistant Axis

**Script:** `pipeline/5_axis.py`
**Library:** PyTorch (tensor operations only)
**Runs on:** Local (CPU)

### What it does

Combines all per-role vectors from Step 4 into a single axis:

1. Load all vector .pt files
2. Separate into two groups:
   - **Default vectors**: roles where `"default" in role_name` or `type == "mean"` (typically 1-2 vectors: `default.pt`, possibly `assistant.pt`)
   - **Role vectors**: everything else (typically 200+ vectors, depending on how many passed the min_count filter)
3. Stack and compute means:
   - `default_mean = mean(default_vectors)` → shape `(n_layers, hidden_dim)`
   - `role_mean = mean(role_vectors)` → shape `(n_layers, hidden_dim)`
4. Compute axis:
   ```
   axis = default_mean - role_mean
   ```

### Axis direction convention

The axis points **FROM role-playing TOWARD default assistant behavior**.

- Positive projection → more assistant-like
- Negative projection → more role-playing

This is the convention Lu et al. use. When measuring drift from meta-reflective prompts, drift appears as movement in the **negative** direction (away from the assistant pole).

### Output

A single tensor saved to `axis.pt`:
```python
tensor(n_layers, hidden_dim)  # e.g., (46, 3584) for Gemma 27B
```

The script also prints per-layer norms, which indicate where the axis is strongest. Typically the middle-to-late layers have the largest norms, with the paper's recommended `target_layer` (layer 22 for Gemma 27B) being near the peak.

### Using the axis

To project a new activation onto the axis:
```python
from assistant_axis import load_axis, project

axis = load_axis("outputs/gemma-2-27b-it/axis.pt")
score = project(activation, axis, layer=22)
# score > 0: assistant-like
# score < 0: role-playing
```


---

## Resource Estimates

### GPU time (vast.ai)

| Step | Gemma 27B (1x A100 80GB) | Notes |
|------|--------------------------|-------|
| Step 1: Generate | ~4-8 hours | 275 roles x 240 questions, vLLM batching helps |
| Step 2: Activations | ~6-12 hours | Full forward pass per response, batch_size=16 |
| **Total GPU** | **~10-20 hours** | |

At ~$1-2/hr for A100 80GB on vast.ai: **~$10-40 per model**.

### Local compute

| Step | Time estimate | Cost |
|------|---------------|------|
| Step 3: Judge | ~30-60 min | ~$2-5 (gpt-4.1-mini, 330k short calls) |
| Step 4: Vectors | < 5 min | Free (CPU) |
| Step 5: Axis | < 1 min | Free (CPU) |

### Storage

| Data | Size (Gemma 27B) |
|------|-------------------|
| Responses (JSONL) | ~2-5 GB |
| Activations (.pt) | ~40-50 GB |
| Scores (JSON) | ~50 MB |
| Vectors (.pt) | ~500 MB |
| Axis (.pt) | ~650 KB |

The activations dominate. For the test run with `--test-roles default assistant detective` (3 roles), expect ~500MB.


---

## Our Orchestration

We wrap their pipeline scripts without modification:

```
run_experiment.py --phase 1
    ├── vast_utils.py run --model ...       # Launch vast.ai, run gpu_pipeline.py
    │       └── gpu_pipeline.py             # On GPU: calls 1_generate.py, 2_activations.py
    ├── vast_utils.py download --model ...  # SCP results to local
    └── local_pipeline.py --model ...       # Local: calls 3_judge.py, 4_vectors.py, 5_axis.py
```

**gpu_pipeline.py** runs on the vast.ai instance (baked into Docker image). It calls the Lu et al. pipeline scripts with correct paths to the data directory and model name.

**local_pipeline.py** runs locally after downloading GPU results. It calls the remaining pipeline scripts in sequence.

**vast_utils.py** handles instance lifecycle: search offers, launch, wait for ready, SCP files, run remote commands, download results, destroy.

### Testing path

```bash
# Test with 3 roles (fast, cheap)
python run_experiment.py --phase 1 --model google/gemma-2-27b-it \
    --test-roles default assistant detective

# Full run (275 roles)
python run_experiment.py --phase 1 --model google/gemma-2-27b-it
```

### Manual step-by-step (if orchestration fails)

```bash
# 1. Launch GPU and run steps 1-2
python vast_utils.py run --model google/gemma-2-27b-it

# 2. Download results
python vast_utils.py download --model google/gemma-2-27b-it

# 3. Run local steps 3-5
python local_pipeline.py --model google/gemma-2-27b-it

# 4. Destroy instance
python vast_utils.py destroy
```
