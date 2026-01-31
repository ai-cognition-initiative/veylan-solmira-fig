# Hidden States and Activations

[Back to index](index.md) | Related: [Tokenization](tokenization.md), [Floating Point](floating-point.md), [KV Cache](kv-cache.md)

---

## What hidden states are

A transformer model is a stack of layers. Each layer transforms its input into an output of the same shape. The output of layer N becomes the input to layer N+1.

For a given token at a given layer, the **hidden state** is the vector representation of that token at that point in the computation. It has shape `(hidden_dim,)` — for Gemma 2 27B, that's a 3,584-dimensional vector.

```
Token: "detective"

Layer 0 output:  [0.12, -0.45, 0.78, ..., 0.33]   # 3,584 numbers
Layer 1 output:  [0.15, -0.38, 0.91, ..., 0.28]
Layer 2 output:  [0.21, -0.22, 1.03, ..., 0.19]
...
Layer 45 output: [0.89, -1.12, 0.42, ..., -0.55]
```

Early layers capture low-level features (syntax, token identity). Middle layers capture semantic content (meaning, relationships). Late layers capture task-specific features (what to say next). This is a rough gradient — the transition is gradual and there's overlap.

## What these vectors "mean"

Each dimension doesn't have an easily interpretable meaning. The model learned to encode information in a distributed way across all 3,584 dimensions. But directions in this space do have meaning.

The assistant axis is a direction in this space. Projecting a hidden state onto the axis tells you where the model is on the spectrum between "default assistant" and "role-playing" behavior. This works because the model uses its hidden states to represent its current behavioral mode, and that mode has a consistent geometric structure.

## What Step 2 extracts

For each conversation, Step 2:

1. Runs the full conversation (system prompt + user question + assistant response) through the model
2. At every layer, captures the hidden state of every token
3. Identifies which tokens belong to the assistant's response (using [chat template](chat-templates.md) knowledge)
4. Computes the **mean** across assistant response tokens at each layer

The result is a tensor of shape `(n_layers, hidden_dim)` — one average vector per layer for the assistant's response.

```
                                tokens in response
                        "As"  "a"  "detective"  ","  "I"  ...  "."
Layer 0 hidden states:  [v0]  [v1]    [v2]     [v3] [v4] ... [vN]
Layer 1 hidden states:  [v0]  [v1]    [v2]     [v3] [v4] ... [vN]
...
Layer 45:               [v0]  [v1]    [v2]     [v3] [v4] ... [vN]

Mean across tokens →   (n_layers, hidden_dim)
```

## Why mean across tokens?

Individual tokens have noisy representations — the hidden state for "," is different from "detective." But averaged across an entire response, the signal of the model's behavioral mode emerges from the noise. This is a common technique in representation analysis.

Think of it like polling: any individual voter is noisy, but the average of 200 voters gives a stable signal.

## Why all layers?

Different layers might capture the behavioral signal at different strengths. By extracting all layers, the analysis can determine which layer has the strongest signal. For the assistant axis, this turns out to be around the middle layers — layer 22 for Gemma 2 27B (out of 46 total).

The axis computation in Step 5 produces a per-layer axis, and projection can be done at any layer. The "target layer" from the model config is the recommended one based on where the signal is strongest.

## How extraction works mechanically

The pipeline uses **forward hooks** — a PyTorch mechanism for intercepting intermediate computations without modifying the model:

```python
def capture_hook(module, input, output):
    # output is the hidden state at this layer
    # shape: (batch_size, seq_len, hidden_dim)
    captured_states.append(output[0].clone().cpu())

# Register hook on layer 22
hook = model.layers[22].register_forward_hook(capture_hook)
model(input_ids)  # forward pass — hook fires automatically
hook.remove()
```

The hook function runs every time data passes through the layer during a forward pass. It captures the output tensor (moving it to CPU to free GPU memory) without affecting the model's computation.

For batch extraction, hooks are registered on all layers simultaneously, and the model processes multiple conversations in one forward pass:

```python
# batch_activations shape: (num_layers, batch_size, max_seq_len, hidden_size)
```

The `SpanMapper` then uses the token span information to slice out the assistant response tokens and compute means.

## Hidden states vs KV cache

These are related but different:

- **[KV cache](kv-cache.md)**: Stores Key and Value projections for attention. Used during generation to avoid recomputing previous tokens. Shape per layer: `(seq_len, num_heads, head_dim)` for K and V separately.
- **Hidden states**: The full representation at a layer. Shape: `(seq_len, hidden_dim)`. This is what gets extracted for analysis.

The hidden state is the "main" representation. K and V are projections of it used specifically by the attention mechanism. The hidden state contains more information — it's what gets fed to the next layer and what the model uses to decide everything about the next token.

## Residual stream interpretation

In transformer architectures, there's a "residual stream" — the hidden state vector that flows through the network, with each layer adding to it rather than replacing it:

```
hidden_state = embedding(token)
for layer in layers:
    hidden_state = hidden_state + layer(hidden_state)  # residual connection
```

Each layer's contribution is *added* to the running total. This means the hidden state at layer N contains information from all layers 0 through N. The axis is computed from these cumulative representations, so it captures the model's full processing up to that layer.

This is also why middle layers often work best for the axis: early layers haven't accumulated enough information, late layers include task-specific features (like "what word comes next") that add noise.
