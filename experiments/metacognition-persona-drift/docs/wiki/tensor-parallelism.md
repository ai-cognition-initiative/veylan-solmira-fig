# Tensor Parallelism

[Back to index](index.md) | Related: [vLLM](vllm.md), [Floating Point](floating-point.md)

---

## The problem

Large models don't fit on one GPU. Gemma 2 27B in [bf16](floating-point.md) is ~54GB — fits on an A100 80GB. Llama 3.3 70B in bf16 is ~140GB — doesn't fit on any single GPU available today.

Tensor parallelism is one strategy for splitting a model across multiple GPUs so it can run.

## How it works

Each transformer layer consists of large matrix multiplications. Tensor parallelism splits these matrices across GPUs along one dimension, so each GPU does part of the computation:

**Single GPU (no parallelism):**
```
Input (batch, seq_len, 8192) × Weight (8192, 8192) = Output (batch, seq_len, 8192)
      one GPU does the entire multiplication
```

**2-GPU tensor parallelism:**
```
GPU 0: Input (batch, seq_len, 8192) × Weight_half_0 (8192, 4096) = Output_0 (batch, seq_len, 4096)
GPU 1: Input (batch, seq_len, 8192) × Weight_half_1 (8192, 4096) = Output_1 (batch, seq_len, 4096)

Then: concat or reduce Output_0 and Output_1 across GPUs
```

Each GPU stores half the weights and does half the computation. The overhead is inter-GPU communication (synchronizing partial results after each layer). This is why NVLink-connected GPUs (like A100 SXM4) are preferred — NVLink has ~600 GB/s bandwidth vs ~32 GB/s for PCIe.

## Tensor parallelism vs other strategies

There are several ways to distribute a model:

**Tensor parallelism (TP)**: Split each layer's matrices across GPUs. All GPUs work on every token, communicating after each layer. Low latency (each token processed faster), high communication overhead.

**Pipeline parallelism (PP)**: Put different layers on different GPUs. GPU 0 has layers 0-22, GPU 1 has layers 23-45. Data flows through GPUs sequentially. Less communication, but harder to keep all GPUs busy (GPU 1 idles while GPU 0 processes).

**Data parallelism (DP)**: Each GPU has a full copy of the model, processing different data. No splitting needed but requires each GPU to fit the entire model. Not useful when the model doesn't fit on one GPU.

vLLM uses tensor parallelism by default when `tensor_parallel_size > 1`.

## In this pipeline

```python
# 1_generate.py detects GPUs and sets tensor_parallel_size
tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()

# vLLM uses it:
LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=2)
```

For Llama 70B on 2x A100 80GB: each GPU holds half the model (~70GB each), and they communicate via NVLink after each layer.

## Multi-worker mode

The pipeline also supports a different kind of parallelism: if you have more GPUs than `tensor_parallel_size`, it launches multiple independent workers, each with its own model instance processing different roles:

```
4x A100 80GB, tensor_parallel_size=2:
  Worker 0 (GPU 0+1): processes roles 0-137
  Worker 1 (GPU 2+3): processes roles 138-274
```

This is a combination of tensor parallelism (within each worker) and data parallelism (across workers). The workers don't communicate with each other — they independently process their assigned roles.

## GPU configs for our models

| Model | Weights (bf16) | Minimum GPUs | Recommended |
|-------|---------------|--------------|-------------|
| Gemma 2 27B | ~54 GB | 1x A100 80GB | 1x A100 80GB |
| Qwen 3 32B | ~64 GB | 1x A100 80GB | 1x A100 80GB |
| Llama 3.3 70B | ~140 GB | 2x A100 80GB | 2x A100 80GB (NVLink) |
