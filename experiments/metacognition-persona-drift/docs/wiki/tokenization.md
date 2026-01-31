# Tokenization

[Back to index](index.md) | Related: [Chat Templates](chat-templates.md), [KV Cache](kv-cache.md)

---

## Why tokenization exists

Neural networks operate on numbers, not text. Tokenization is the conversion from text to a sequence of integer IDs, where each ID indexes into a learned vocabulary. The model never sees characters — it sees token IDs.

## Byte-Pair Encoding (BPE)

Most modern LLMs use BPE or a variant (SentencePiece, Unigram). The idea:

1. Start with individual characters as the vocabulary: `a`, `b`, `c`, ...
2. Count all adjacent character pairs in a training corpus
3. Merge the most frequent pair into a new token (e.g., `t` + `h` → `th`)
4. Repeat: count pairs again, merge the most frequent (e.g., `th` + `e` → `the`)
5. Continue until the vocabulary reaches a target size (e.g., 32,000 or 128,000 tokens)

The result is a vocabulary where:
- Common words are single tokens: `the`, `and`, `function`
- Rare words are split into subword pieces: `metacognition` → `meta` + `cogn` + `ition`
- Very rare strings fall back to character-level: `xyzzy` → `x` + `y` + `z` + `z` + `y`

## What a token actually is

A token is an entry in the vocabulary with an integer ID. Examples from a typical LLM vocabulary:

```
Token ID    Token
   0        <pad>
   1        <eos>
   2        <bos>
 1234       the
 5678       function
 9012       meta
 9013       cogn
 9014       ition
```

Tokenizing the sentence `"The function works"` might produce:
```python
[2, 450, 5678, 3842]
#  ^BOS "The"  "function" "works"
```

Note: `"The"` (capitalized) is a different token than `"the"` (lowercase). Case matters.

## Token ≠ word ≠ character

Common misconception: a token is a word. In reality:
- Some tokens are multiple words: `" is not"` (including the leading space)
- Some tokens are subword pieces: `ition`, `ing`, `##ed`
- Some tokens are single characters: `x`, `(`, `\n`
- Some tokens are special markers: `<bos>`, `<eos>`, `<start_of_turn>`

A rough heuristic: **1 token ≈ 4 characters ≈ 0.75 words** in English. A 512-token response is roughly 380 words.

## Vocabulary size

| Model | Vocabulary size |
|-------|----------------|
| Gemma 2 27B | 256,000 |
| Llama 3.3 70B | 128,256 |
| Qwen 3 32B | 151,936 |

Larger vocabularies mean common sequences get their own token (more efficient), but the embedding matrix is bigger (more memory).

## Special tokens

Beyond regular text tokens, models have special tokens with reserved IDs:

- **BOS** (beginning of sequence): signals the start of input
- **EOS** (end of sequence): signals the model should stop generating
- **PAD** (padding): fills unused positions in batched inputs
- **Chat markers**: role delimiters like `<start_of_turn>`, `<|im_start|>` (see [Chat Templates](chat-templates.md))

Special tokens are **never generated from text** — they're inserted programmatically. If you type `"<bos>"` as input text, the tokenizer encodes it as the characters `<`, `b`, `o`, `s`, `>` (5 tokens), NOT as the special BOS token (1 token).

## The embedding layer

The model's first layer is an embedding matrix of shape `(vocab_size, hidden_dim)`. Each token ID is used to look up its embedding vector:

```
token_id = 5678  ("function")
embedding = embedding_matrix[5678]  # shape: (hidden_dim,)
```

For Gemma 2 27B with hidden_dim=3584 and vocab_size=256,000:
```
embedding_matrix shape: (256000, 3584)
memory: 256000 × 3584 × 2 bytes (bf16) ≈ 1.7 GB
```

This embedding vector is the model's first representation of the token. It then gets transformed through 46 transformer layers (for Gemma 27B), producing the [hidden states](hidden-states.md) that the pipeline extracts in Step 2.

## Detokenization

The reverse process: converting token IDs back to text. The tokenizer maintains a mapping from IDs to token strings:

```python
tokenizer.decode([450, 5678, 3842])  # → "The function works"
```

`skip_special_tokens=True` filters out BOS/EOS/padding tokens from the decoded text — you don't want `<eos>` appearing in the model's response.

## Why tokenization matters for this pipeline

In Step 2 (activation extraction), the pipeline needs to identify exactly which tokens correspond to the assistant's response. This requires re-tokenizing the conversation and mapping character positions to token positions. The `ConversationEncoder` class handles this with model-specific logic, because different [chat templates](chat-templates.md) produce different token sequences for the same conversation.

Tokenization is also why `max_model_len=2048` is measured in tokens, not characters or words. A 2048-token context window holds roughly 1500 words of text — comfortably enough for a system prompt + question + 512-token response.
