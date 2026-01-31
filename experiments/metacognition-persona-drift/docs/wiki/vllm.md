# vLLM, PagedAttention, and Continuous Batching

[Back to index](index.md) | Related: [KV Cache](kv-cache.md), [Tensor Parallelism](tensor-parallelism.md)

---

## What vLLM is

vLLM is an open-source inference engine for large language models. It replaces HuggingFace's `model.generate()` for workloads where you need to generate many responses efficiently. It is a serving/inference tool, not a training tool.

The core paper: Kwon et al. (2023), "Efficient Memory Management for Large Language Model Serving with PagedAttention."

In this pipeline, vLLM is used only in Step 1 (generating responses). Step 2 uses HuggingFace directly because it needs access to internal model states that vLLM abstracts away.

## Why not just use HuggingFace?

HuggingFace's `model.generate()` works but is slow for batch workloads. The fundamental issue is how it manages [KV cache](kv-cache.md) memory.

**HuggingFace approach**: pre-allocate a contiguous KV cache tensor for each sequence at maximum length. With batch size 16 and max_seq_len 2048 on Gemma 27B, that's ~20GB of cache pre-allocated regardless of actual response lengths. Short responses waste most of it. And you can't add new sequences until the slowest one finishes.

**vLLM approach**: allocate KV cache in small, non-contiguous pages on demand. Short responses use few pages. Long responses grow into more pages. Finished sequences release pages immediately for new sequences. This means:
- Higher effective batch sizes (less wasted memory)
- Dynamic scheduling (new work starts as soon as capacity frees up)
- Better GPU utilization across variable-length outputs

## PagedAttention

PagedAttention is the core innovation. It borrows the concept of virtual memory from operating systems.

**Physical vs virtual memory (OS analogy):**
In an OS, a program thinks it has a contiguous block of memory (virtual addresses 0x0000 to 0xFFFF). In reality, the OS maps these to scattered physical pages wherever there's free RAM. The program doesn't know or care.

**Physical vs virtual KV cache (vLLM):**
Each sequence thinks it has a contiguous KV cache. In reality, vLLM maps the cache into fixed-size blocks (typically 16 tokens per block) stored wherever there's free GPU memory. The attention computation is modified to read from non-contiguous blocks via a block table (analogous to a page table).

```
Traditional:
  Sequence 1: [████████████████████████████░░░░░░░░░░░░░░░░]  (lots of waste)
  Sequence 2: [██████████████████████████████████████████████]
  Sequence 3: [can't fit — no contiguous space]

PagedAttention:
  Block pool: [1a][2c][1b][3a][2a][1c][3b][2b][free][free]
  Seq 1 block table: [1a, 1b, 1c]        (scattered but complete)
  Seq 2 block table: [2a, 2b, 2c]
  Seq 3 block table: [3a, 3b]            (fits! uses free blocks)
```

The overhead is small (one indirection per attention block), and the memory efficiency gain is large — typically 2-4x more sequences can be served simultaneously.

## Continuous batching

Traditional batching runs N sequences in parallel and waits for ALL of them to finish before processing the next batch. If sequence 1 generates 20 tokens and sequence 16 generates 500, sequences 1-15 sit idle while 16 finishes.

Continuous batching (also called "iteration-level scheduling") operates at the granularity of a single decode step:

1. Each iteration, run one decode step for all active sequences
2. If any sequence finishes (EOS or max_tokens), remove it immediately
3. If any new sequences are waiting and there's KV cache capacity, add them immediately
4. Repeat

The GPU stays saturated. No sequence sits idle waiting for slower peers.

```
Traditional batching:
  Time →  [===Batch 1====][wait][===Batch 2====][wait]
  GPU:     ████████████████░░░░████████████████░░░░

Continuous batching:
  Time →  [==========continuous===========]
  GPU:     ████████████████████████████████
```

In this pipeline, generating 1,200 prompts for a single role means vLLM processes them in a continuous stream — early finishers make room for stragglers, and the GPU is never idle.

## How generation works in vLLM

When you call `llm.generate(prompts, sampling_params)`:

### 1. Tokenize
Each text prompt is converted to token IDs using the model's [tokenizer](tokenization.md).

### 2. Schedule
The scheduler decides which sequences to process in each iteration. It considers:
- Available GPU memory (for new KV cache blocks)
- Running sequences (their current KV cache size)
- Waiting sequences (their prompt length)

### 3. Prefill
For new sequences, the model processes all prompt tokens in one forward pass. This is compute-bound and fast. The [KV cache](kv-cache.md) for the entire prompt is computed.

### 4. Decode
For sequences past their prompt, the model generates one token at a time. This is memory-bandwidth-bound (reading KV cache). Each iteration:
- Forward pass with batch of current tokens (one per active sequence)
- [Sample](sampling.md) next token for each sequence
- Append to KV cache
- Check stopping conditions (EOS, max_tokens)

### 5. Return
When all sequences are complete, return the generated text.

The key difference from HuggingFace: steps 3 and 4 are interleaved across sequences. While some sequences are in decode, new sequences can be prefilled in the same batch, as long as memory allows.

## vLLM in this pipeline

```python
self.llm = LLM(
    model="google/gemma-2-27b-it",
    max_model_len=2048,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
)
```

- `max_model_len=2048`: Maximum total sequence length (prompt + generated tokens). This caps KV cache per sequence.
- `tensor_parallel_size=1`: Use 1 GPU. For Llama 70B, this would be 2 (see [Tensor Parallelism](tensor-parallelism.md)).
- `gpu_memory_utilization=0.9`: vLLM can use up to 90% of GPU memory. After loading ~54GB of model weights on an 80GB A100, ~18GB remains for KV cache.
- `trust_remote_code=True`: Allow custom model code from HuggingFace (some models need this).

Generation:
```python
outputs = self.llm.generate(prompts, self.sampling_params)
# prompts: 1,200 text strings
# Returns: 1,200 output objects, each with .outputs[0].text
```

vLLM handles all the scheduling internally. You don't control batch size — it figures out the optimal throughput given the available memory.
