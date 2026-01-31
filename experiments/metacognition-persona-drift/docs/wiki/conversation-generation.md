# Conversation Generation

How automated multi-turn conversations are produced for the drift experiment.

Related: [Conversation Infrastructure](../conversation-infrastructure.md), [Metacognitive Domain](../metacognitive-domain.md), [Experimental Methodology](../experimental-methodology.md)

---

## Overview

`generate_conversations.py` replicates Lu et al.'s auditor-target methodology: a frontier LLM ("auditor") simulates a human user while an open-weight target model responds. Turns alternate until `max_turns` or the auditor sends `<END_CONVERSATION>`.

```
┌────────────────────┐       ┌────────────────────┐
│  Auditor (API)     │       │  Target (local)    │
│  Claude / GPT / …  │ ───── │  Gemma 27B / …     │
│                    │ turns │                    │
│  Has: system prompt│       │  Has: no system    │
│  with persona,     │       │  prompt (default)  │
│  domain, topic     │       │                    │
└────────────────────┘       └────────────────────┘
```

**Why the target has no system prompt**: Lu et al. gave target models no system prompt in their drift experiments. The model starts from its default post-training persona. This is the persona whose drift we're measuring.

---

## Usage

### Single conversation

```bash
python generate_conversations.py \
    --domain coding \
    --persona-id 0 \
    --topic-id 0 \
    --target-model google/gemma-2-27b-it \
    --auditor-model anthropic/claude-sonnet-4-20250514
```

### Predefined batches

```bash
# Lu et al. 4-domain replication
python generate_conversations.py \
    --batch lu-replication \
    --batch-size 20 \
    --target-model google/gemma-2-27b-it \
    --auditor-model anthropic/claude-sonnet-4-20250514

# Our metacognitive condition
python generate_conversations.py \
    --batch metacognitive \
    --batch-size 10 \
    --target-model google/gemma-2-27b-it \
    --auditor-model anthropic/claude-sonnet-4-20250514
```

### Custom batch from config file

```bash
python generate_conversations.py \
    --config conversations/my-batch.json \
    --target-model google/gemma-2-27b-it
```

Config format:
```json
[
    {"domain": "metacognitive", "persona_id": 0, "topic_id": 1, "condition": "meta-gradual"},
    {"domain": "coding", "persona_id": 0, "topic_id": 0, "condition": "control-coding"},
    {"domain": "therapy", "persona": "Custom persona text...", "topic": "Custom topic text..."}
]
```

### Dry run (inspect auditor prompt)

```bash
python generate_conversations.py --domain metacognitive --dry-run
```

---

## Architecture

### Turn-taking loop

```python
while turn < max_turns:
    # 1. Auditor generates user message (via API)
    auditor_msg = await call_auditor(system_prompt, auditor_view, model_id)

    # 2. Check for termination
    if "<END_CONVERSATION>" in auditor_msg:
        break

    # 3. Target generates assistant response (local HuggingFace)
    target_msg = generate_target_response(model, tokenizer, conversation)

    # 4. Both messages appended to conversation history
```

### Role perspective handling

The auditor and target see the conversation from different perspectives:

| Canonical role | What auditor sees | What target sees |
|---------------|-------------------|-----------------|
| `user` (auditor's message) | `assistant` (its own output) | `user` (incoming message) |
| `assistant` (target's message) | `user` (incoming message) | `assistant` (its own output) |

The script flips roles when constructing the auditor's view of the conversation.

### Auditor backends

| Provider | Model format | Env var | Example |
|----------|-------------|---------|---------|
| `anthropic` | `anthropic/claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` | Default |
| `openai` | `openai/gpt-4o` | `OPENAI_API_KEY` | |
| `openrouter` | `openrouter/anthropic/claude-sonnet-4` | `OPENROUTER_API_KEY` | |

All auditor calls use `max_tokens=256` to keep user messages short and natural.

### Target model loading

Prefers `assistant_axis.internals.ProbingModel` (which provides activation access for later analysis). Falls back to raw `AutoModelForCausalLM` if the assistant-axis package isn't available.

---

## Output format

Transcripts are saved to `transcripts/generated/` as JSON, matching the format of existing transcripts in `assistant-axis/transcripts/persona_drift/`:

```json
{
    "model": "google/gemma-2-27b-it",
    "auditor_model": "anthropic/claude-sonnet-4-20250514",
    "domain": "metacognitive",
    "persona_id": 0,
    "persona": "You are a philosophy graduate student...",
    "topic_id": 1,
    "topic": "Ask the model to describe its own uncertainty...",
    "turns": 24,
    "max_turns": 30,
    "target_system_prompt": null,
    "timestamp": "2026-01-31T15:30:00",
    "condition": "metacognitive",
    "batch_index": 3,
    "conversation": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
}
```

Filename pattern: `{domain}_p{persona_id}_t{topic_id}_{timestamp}.json`

---

## Post-generation: activation extraction

Transcripts don't include activations — those require a GPU forward pass. After generating transcripts:

1. **Interactive analysis**: Use the `project_transcript.ipynb` notebook from the assistant-axis repo to load a transcript, run a forward pass, and plot the drift trajectory.

2. **Batch extraction** (not yet built): A script that takes a directory of transcripts, runs each through the target model, extracts per-turn activations, and saves projection trajectories. This would parallel the existing `pipeline/2_activations.py` but for multi-turn conversations instead of single-turn responses.

---

## Domains and personas

### Available domains

| Domain | Source | Personas | Topics | Notes |
|--------|--------|----------|--------|-------|
| `coding` | Lu et al. Table 15 | 1 | 2 | Minimal drift expected (control) |
| `writing` | Lu et al. Table 15 | 1 | 1 | Minimal drift expected (control) |
| `therapy` | Lu et al. Table 15 | 1 | 2 | Significant drift expected (replication) |
| `philosophy` | Lu et al. Table 15 | 1 | 1 | Significant drift expected (replication) |
| `metacognitive` | **Ours** | 2 | 5 | See [Metacognitive Domain](../metacognitive-domain.md) |

### Coverage gaps

Lu et al. used **5 personas × 20 topics = 100 conversations per domain**. Our current coverage:

| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| coding | 1×2 = 2 | 5×20 = 100 | Need 98 more setups |
| writing | 1×1 = 1 | 5×20 = 100 | Need 99 more setups |
| therapy | 1×2 = 2 | 5×20 = 100 | Need 98 more setups |
| philosophy | 1×1 = 1 | 5×20 = 100 | Need 99 more setups |
| metacognitive | 2×2.5 = 5 | 5×20 = 100 | Need 95 more setups |

**How to close the gap:**
- **Author response**: If Lu et al. share their full persona/topic sets, we can import them directly for the 4 original domains.
- **LLM generation**: Lu et al. used Kimi K2 to generate 20 topics per persona from a simple prompt (Appendix E.1). We can replicate this for all domains.
- **Manual authoring**: For the metacognitive domain specifically, hand-written personas and topics may produce higher quality probes than LLM-generated ones, since the auditor's probing strategy is more specialized.

---

## Testing and validation

### Before running a full batch

1. **`--dry-run`**: Inspect auditor system prompt for each condition
2. **Single conversation**: Run one conversation per domain, review transcript manually
3. **Auditor quality check**: Does the auditor stay in character? Does it use metacognitive probes (for that domain)? Is it too polite? Too generic?
4. **Target response quality**: Are responses natural? Does the model engage with probes or immediately deflect?

### After generating transcripts

1. **Naturalness check**: Read 2-3 transcripts per domain. Lu et al. human-inspected all transcripts.
2. **Turn count**: Did conversations reach max_turns, or did auditors end early? Early termination may indicate the auditor ran out of things to say.
3. **Activation projection**: Load transcripts into `project_transcript.ipynb`. Do the expected drift patterns appear (stable for coding, drifting for therapy/philosophy/metacognitive)?

---

## Known limitations

1. **No activation extraction during generation**: The script generates conversations only. Activations must be extracted in a separate pass. This doubles the forward passes through the target model but keeps generation and analysis cleanly separated.

2. **Sequential execution**: Conversations run one at a time. For large batches, this is slow. Parallelization is possible but would require multiple target model instances or careful GPU scheduling.

3. **Auditor rate limits**: API-based auditor calls are subject to rate limits. The script doesn't implement rate limiting or retry logic beyond what the SDK provides.

4. **No conversation quality filtering**: Lu et al. had humans inspect all transcripts. We should do the same, at least for a sample, before drawing conclusions from the activation analysis.
