# KV Cache

[Back to index](index.md) | Related: [vLLM](vllm.md), [Hidden States](hidden-states.md)

---

## The problem KV cache solves

Transformer models generate text one token at a time. To generate token 100, the model needs to "see" all 99 previous tokens. Naively, this means running the full model on all 99 tokens again just to produce token 100, then on all 100 tokens to produce 101, and so on. The cost grows quadratically.

KV cache eliminates the redundant computation by caching intermediate results from previous tokens.

## What K and V actually are

In each transformer layer, the attention mechanism computes three things from the input:
- **Q (Query)**: "What am I looking for?" — derived from the current token
- **K (Key)**: "What do I contain?" — derived from each token
- **V (Value)**: "What information do I provide?" — derived from each token

Attention works by comparing the current token's Q against every previous token's K (via dot product) to get attention weights, then using those weights to take a weighted sum of every previous token's V.

The crucial insight: **K and V for past tokens don't change.** Token 50's K and V are the same whether you're generating token 51 or token 500. Only Q changes (because it depends on the current token).

So you cache the K and V tensors for all past tokens. When generating token 100:
1. Compute Q, K, V for token 100 only (one token forward pass)
2. Append token 100's K and V to the cache
3. Compute attention using token 100's Q against all 100 cached K's
4. Use attention weights with all 100 cached V's

Instead of processing 100 tokens, you process 1 token + read from cache. This is why generation after the first token (the "decode" phase) is fast per token.

## Memory cost

For each token in the sequence, you store K and V at every layer. The size per token:

```
bytes_per_token = 2 × num_layers × hidden_dim × bytes_per_element
```

The `2×` is for K and V. In [bf16](floating-point.md), `bytes_per_element = 2`.

For Gemma 2 27B (46 layers, 3584 hidden dim):
```
bytes_per_token = 2 × 46 × 3584 × 2 = 659,456 bytes ≈ 644 KB per token
```

For a sequence of 2048 tokens:
```
2048 × 644 KB ≈ 1.3 GB per sequence
```

If you want to batch 16 sequences simultaneously:
```
16 × 1.3 GB ≈ 20.5 GB just for KV cache
```

This is why KV cache is the primary memory bottleneck during generation — not the model weights. The model weights for Gemma 27B are ~54GB and fixed. The KV cache grows with batch size and sequence length.

## Prefill vs decode

These are the two phases of generation, and they have very different performance characteristics:

**Prefill** (processing the prompt):
- Input: all prompt tokens at once (e.g., 200 tokens)
- Compute: full forward pass on 200 tokens in parallel
- Output: KV cache entries for all 200 tokens
- Bottleneck: **compute** (matrix multiplications across 200 tokens)
- Speed: fast, because GPUs are good at parallel matrix math

**Decode** (generating new tokens):
- Input: 1 token at a time
- Compute: forward pass on 1 token, but must read entire KV cache
- Output: 1 new KV entry + 1 new token
- Bottleneck: **memory bandwidth** (reading the full KV cache from GPU memory for each token)
- Speed: slow per token, because you're reading GBs of cache for a tiny computation

This asymmetry is why you see GPUs at high utilization during prefill but lower utilization during decode. The GPU spends most of decode waiting for data to arrive from memory rather than computing.

## Why this matters for vLLM

Standard HuggingFace generation pre-allocates KV cache as a contiguous tensor per sequence:
```python
cache_shape = (batch_size, max_seq_len, num_layers, 2, num_heads, head_dim)
```

This has two problems:
1. **Waste**: if max_seq_len is 2048 but a response is 50 tokens, you've allocated 2048 tokens of cache for nothing
2. **Fragmentation**: if you want to add a new sequence to the batch, you need a contiguous block of memory big enough for its full max_seq_len cache

[vLLM's PagedAttention](vllm.md#pagedattention) solves this by breaking the cache into small pages (like OS virtual memory), allocated on demand, non-contiguous. This dramatically increases how many sequences can be in-flight simultaneously.

## KV cache in this pipeline

Step 1 (generation) cares about KV cache because it determines batch throughput. `max_model_len=2048` caps the per-sequence cache size, and `gpu_memory_utilization=0.9` determines how much GPU memory vLLM can use for weights + cache.

Step 2 (activation extraction) does NOT use KV cache in the same way. It runs full forward passes (not autoregressive generation) and captures [hidden states](hidden-states.md) at each layer. The model processes the entire conversation at once and there's no token-by-token decode phase.
