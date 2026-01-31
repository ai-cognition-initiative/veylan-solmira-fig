# Floating Point Formats (bf16 / fp16 / fp32)

[Back to index](index.md) | Related: [Tensor Parallelism](tensor-parallelism.md), [KV Cache](kv-cache.md)

---

## Why this matters

Every number in a neural network — every weight, every activation, every gradient — is stored as a floating point number. The format determines:
- **Memory**: how many bytes per number (directly affects whether a model fits on a GPU)
- **Precision**: how many significant digits (affects numerical accuracy)
- **Speed**: narrower formats compute faster on modern GPUs

## The three common formats

### fp32 (32-bit float)
- **Size**: 4 bytes per number
- **Structure**: 1 sign bit, 8 exponent bits, 23 mantissa bits
- **Range**: ±3.4 × 10^38
- **Precision**: ~7 decimal digits
- **Use**: Training (gradients need precision), CPU computation

### fp16 (16-bit float)
- **Size**: 2 bytes per number
- **Structure**: 1 sign bit, 5 exponent bits, 10 mantissa bits
- **Range**: ±65,504
- **Precision**: ~3.3 decimal digits
- **Use**: Mixed-precision training, inference
- **Problem**: Small range. Gradients and activations can overflow (> 65,504) or underflow (too close to 0). Requires careful loss scaling.

### bf16 (bfloat16, "brain float")
- **Size**: 2 bytes per number
- **Structure**: 1 sign bit, 8 exponent bits, 7 mantissa bits
- **Range**: ±3.4 × 10^38 (same as fp32!)
- **Precision**: ~2.4 decimal digits (less than fp16)
- **Use**: Default for LLM inference and training
- **Advantage**: Same range as fp32, so no overflow/underflow issues. Trades precision for range, which works well for neural networks.

```
fp32:  [1 sign][8 exponent][23 mantissa]  = 32 bits
fp16:  [1 sign][5 exponent][10 mantissa]  = 16 bits
bf16:  [1 sign][8 exponent][ 7 mantissa]  = 16 bits
                ^^^^^^^^^^
                same as fp32 — this is the key insight
```

## Memory impact

Model size in memory = number_of_parameters × bytes_per_parameter.

Gemma 2 27B:
```
fp32: 27 billion × 4 bytes = 108 GB  (doesn't fit on any single GPU)
bf16: 27 billion × 2 bytes =  54 GB  (fits on A100 80GB)
```

Llama 3.3 70B:
```
fp32: 70 billion × 4 bytes = 280 GB
bf16: 70 billion × 2 bytes = 140 GB  (fits on 2x A100 80GB)
```

This is why bf16 is the default. Half the memory of fp32 with negligible quality loss.

## Why bf16 over fp16?

Google developed bf16 specifically for deep learning. The key observation: neural network values occasionally get very large or very small (especially during training), but they don't need many significant digits.

fp16 has more precision (10 mantissa bits vs 7) but a tiny range (max 65,504). When values exceed this range, they become `inf` — a silent, catastrophic failure.

bf16 has less precision but the same range as fp32. Values that would overflow in fp16 are handled fine in bf16. The reduced precision (2.4 vs 3.3 decimal digits) is negligible for neural network inference.

## In this pipeline

```python
# ProbingModel loads in bf16 by default
self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
```

The model weights, activations, and [KV cache](kv-cache.md) are all in bf16. The [hidden states](hidden-states.md) extracted in Step 2 are stored in bf16 tensors.

When doing math on these tensors (computing means, dot products, projections), the code often casts to fp32 first for numerical stability:

```python
# From axis.py:
act = activations[layer].float()  # .float() converts bf16 → fp32
ax = axis[layer].float()
return float(act @ ax)
```

This is a common pattern: store in bf16 (saves memory), compute in fp32 (avoids rounding errors accumulating through operations).

## Quantization (INT8, INT4)

Beyond floating point, models can be quantized to integer formats:
- **INT8**: 1 byte per parameter. ~27GB for Gemma 27B. Some quality loss.
- **INT4**: 0.5 bytes per parameter. ~14GB for Gemma 27B. Noticeable quality loss.

Quantized models fit on smaller GPUs (RTX 4090 24GB could run INT4 Gemma 27B), but the activations are different from bf16. For this pipeline, which extracts activations to build the assistant axis, we use bf16 to match the original paper.
