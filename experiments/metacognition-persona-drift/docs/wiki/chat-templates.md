# Chat Templates

[Back to index](index.md) | Related: [Tokenization](tokenization.md), [Step 1](step-1-generate.md)

---

## The problem

A conversation has semantic structure — system instructions, user messages, assistant responses. But a language model just processes a sequence of [tokens](tokenization.md). Chat templates define how to flatten structured conversations into token sequences that the model was trained to understand.

Different models use completely different formats. This is not a trivial formatting detail — a model trained on one format will behave poorly (or refuse to generate) if given a different format.

## Examples

The same conversation:
```python
[
    {"role": "system", "content": "You are a detective."},
    {"role": "user", "content": "What happened at the museum?"},
]
```

### Llama 3.3 format

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a detective.<|eot_id|><|start_header_id|>user<|end_header_id|>

What happened at the museum?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Uses special tokens (`<|start_header_id|>`, `<|eot_id|>`) to delimit roles. The `<|begin_of_text|>` is the BOS (beginning of sequence) token.

### Gemma 2 format

```
<bos><start_of_turn>user
You are a detective.

What happened at the museum?<end_of_turn>
<start_of_turn>model
```

Gemma 2 does **not support system prompts**. The system instruction is prepended to the user message. Uses `<start_of_turn>` / `<end_of_turn>` markers and the role name "model" instead of "assistant."

### Qwen 3 format

```
<|im_start|>system
You are a detective.<|im_end|>
<|im_start|>user
What happened at the museum?<|im_end|>
<|im_start|>assistant
```

Uses ChatML-style markers (`<|im_start|>`, `<|im_end|>` — "im" stands for "imaginary monologue", a legacy name from the ChatML spec).

## How templates work in code

HuggingFace tokenizers include a Jinja2 template that defines the format. You don't write the format manually — you call:

```python
text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,          # return text, not token IDs
    add_generation_prompt=True,  # add the "assistant's turn starts here" marker
)
```

`add_generation_prompt=True` is critical for generation — it tells the model "now it's your turn to speak." Without it, the model doesn't know it should generate an assistant response.

`tokenize=False` returns the formatted string. Set it to `True` (or omit) to get token IDs directly.

## System prompt support detection

The pipeline needs to handle Gemma 2's lack of system prompt support. The code does this empirically:

```python
test_conversation = [
    {"role": "system", "content": "__SYSTEM_TEST__"},
    {"role": "user", "content": "hello"},
]
output = tokenizer.apply_chat_template(test_conversation, tokenize=False)
supports_system = "__SYSTEM_TEST__" in output
```

It renders a test conversation with a canary string. If the canary appears in the output, the template supports system prompts. If not (Gemma 2 silently drops it), the code concatenates the instruction into the user message instead.

This is more robust than hardcoding model names because it works with any model — including future ones.

## Why this matters for the pipeline

The same conversation object gets used in both Step 1 (generation) and Step 2 (activation extraction). The exact same chat template formatting must be applied both times, otherwise the token positions won't match and the activation extraction will read from the wrong tokens.

The `ConversationEncoder` class in `assistant_axis.internals` uses these same chat templates when mapping conversations to token spans in Step 2. It needs to know exactly which tokens are the assistant's response, which requires understanding the model-specific formatting.

## Special tokens

Chat templates use special tokens that have dedicated IDs in the [tokenizer's vocabulary](tokenization.md). These are not regular words — they're control tokens that the model learned to treat as structural markers during training.

For Llama 3.3:
| Token | Purpose |
|-------|---------|
| `<\|begin_of_text\|>` | Start of sequence |
| `<\|start_header_id\|>` | Start of role identifier |
| `<\|end_header_id\|>` | End of role identifier |
| `<\|eot_id\|>` | End of turn |

For Gemma 2:
| Token | Purpose |
|-------|---------|
| `<bos>` | Start of sequence |
| `<start_of_turn>` | Start of turn |
| `<end_of_turn>` | End of turn |
| `<eos>` | End of sequence (generation stop signal) |

These special tokens each have a single token ID even though they look like multiple characters. `<start_of_turn>` is one token, not a sequence of `<`, `s`, `t`, `a`, `r`, `t`, ... tokens.
