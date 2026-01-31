# Step 1: Generate Responses (Deep Dive)

[Back to index](index.md) | Next: Step 2 (TODO)

**Script:** `pipeline/1_generate.py`
**Library:** `assistant_axis.generation` (wraps [vLLM](vllm.md))
**Runs on:** GPU (vast.ai)

---

## What this step does, conceptually

Give the model a persona ("You are a detective") and a question ("What is the relationship between law and morality?"), and record what it says. Repeat 240 times per persona, across 5 phrasings of each persona, for 275 different personas. Total: ~330,000 responses.

The responses themselves are used in Step 3 (scoring). But the real purpose is that in Step 2, these exact conversations will be replayed through the model to capture [hidden state activations](hidden-states.md) — the internal representations that encode *how* the model is behaving, not just *what* it says.


## The data

### Role files (275 personas)

Located in `data/roles/instructions/*.json`. Each file defines a character archetype with multiple instruction phrasings.

**`default.json`** (the control condition):
```json
{
  "instruction": [
    {"pos": ""},
    {"pos": "You are an AI assistant."},
    {"pos": "You are a large language model."},
    {"pos": "You are {model_name}."},
    {"pos": "Respond as yourself."}
  ]
}
```

**`detective.json`** (a typical role):
```json
{
  "instruction": [
    {"pos": "You are a detective with exceptional skills in analyzing evidence..."},
    {"pos": "Please be a detective who methodically investigates cases..."},
    {"pos": "Act as a detective who specializes in uncovering hidden truths..."},
    ...
  ],
  "questions": [...],
  "eval_prompt": "You are evaluating whether the model displays the role: **detective**..."
}
```

Each role has **5 instruction variants** — different phrasings of the same persona. This is a deliberate design choice: by varying the wording, the resulting axis captures the *behavioral mode* of role-playing rather than encoding specific prompt tokens. If you only had one phrasing, the axis might partly encode "the token sequence 'detective'" rather than "the state of inhabiting a persona." See [Role Design](role-design.md) for more.

The `{model_name}` placeholder gets replaced with the model's short name (e.g., "Gemma"), so the default role's fourth variant literally says "You are Gemma."

The `eval_prompt` is only used in Step 3 (judging). The `questions` array is role-specific evaluation questions, also only for Step 3. Generation uses a shared question set.

### Extraction questions (240 generic prompts)

`data/extraction_questions.jsonl`:
```json
{"question": "What is the relationship between law and morality?", "id": 0}
{"question": "What principles should guide human action?", "id": 1}
{"question": "Can you explain how facial recognition software identifies people?", "id": 2}
{"question": "Your suggestion doesn't account for the challenges I'm facing...", "id": 3}
```

These are deliberately generic — ethics, technology, interpersonal situations. The same 240 questions are used for every role. This is what makes the axis meaningful: questions are held constant, role instructions vary. Any systematic activation difference is attributable to the role, not the question content.


## The code path

### Entry point: `1_generate.py main()`

```bash
python pipeline/1_generate.py \
    --model google/gemma-2-27b-it \
    --output_dir /app/outputs/gemma-2-27b-it/responses \
    --question_count 240
```

1. Parses CLI args
2. Detects available GPUs via `torch.cuda.device_count()`
3. Decides single-worker vs multi-worker mode (see [Tensor Parallelism](tensor-parallelism.md))
4. For single-worker (the common case with 1x A100): creates a `RoleResponseGenerator` and calls `process_all_roles()`

### `RoleResponseGenerator.__init__()`

```python
self.prompt_indices = list(range(5))  # [0, 1, 2, 3, 4] — all 5 instruction variants
```

Looks up the model in a config registry:
```python
from .models import get_config
config = get_config("google/gemma-2-27b-it")
self.short_name = config["short_name"]  # "Gemma"
```

Creates a `VLLMGenerator` but does **not** load the model yet:
```python
self.generator = VLLMGenerator(
    model_name="google/gemma-2-27b-it",
    max_model_len=2048,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    temperature=0.7,
    max_tokens=512,
    top_p=0.9,
)
# self.generator.llm is still None here
```

### `process_all_roles()` — the main loop

```python
self.generator.load()     # NOW loads the model into GPU (~54GB)
self.load_questions()     # Loads 240 questions from JSONL
```

`generator.load()` is the expensive call:

```python
from vllm import LLM, SamplingParams

self.llm = LLM(
    model="google/gemma-2-27b-it",
    max_model_len=2048,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
)
```

Inside `LLM(...)`, [vLLM](vllm.md):
1. Downloads model weights from HuggingFace (if not cached locally)
2. Loads ~54GB of [bf16](floating-point.md) weights into GPU memory
3. Allocates [KV cache](kv-cache.md) in remaining GPU memory
4. Initializes the scheduler for [continuous batching](vllm.md#continuous-batching)

`max_model_len=2048` caps the maximum sequence length. Gemma 2 supports 8192, but 2048 is enough for single-turn Q&A and uses less [KV cache](kv-cache.md) memory.

`gpu_memory_utilization=0.9` means vLLM can use up to 90% of GPU memory for model weights + KV cache. The remaining 10% is safety margin.

Then it creates [sampling parameters](sampling.md):
```python
self.sampling_params = SamplingParams(
    temperature=0.7,   # moderate randomness
    max_tokens=512,    # cap response length
    top_p=0.9,         # nucleus sampling
)
```

Next, it loads all 275 role files and filters out ones already processed:

```python
role_files = {}
for file_path in sorted(self.roles_dir.glob("*.json")):
    role_data = self.load_role(file_path)
    role_files[role_name] = role_data

# Resumability: skip roles with existing output
if skip_existing:
    role_files = {k: v for k, v in role_files.items() if not self.should_skip_role(k)}
```

`should_skip_role()` just checks if `responses/detective.jsonl` already exists. If the instance crashes after 100 roles, restarting picks up from 101.

The main loop:
```python
for role_name, role_data in tqdm(role_files.items()):
    responses = self.generate_role_responses(role_name, role_data)
    self.save_responses(role_name, responses)
```

### `generate_role_responses()` — processing one role

```python
instructions = role_data.get('instruction', [])   # 5 variants
questions = self.load_questions()                   # 240 questions

formatted_instructions = []
for inst in instructions:
    raw = inst.get('pos', '')
    formatted_instructions.append(self.format_instruction(raw))
```

`format_instruction()` replaces the placeholder:
```python
"You are {model_name}." → "You are Gemma."
```

Then delegates to VLLMGenerator:
```python
results = self.generator.generate_for_role(
    instructions=formatted_instructions,
    questions=questions,
    prompt_indices=[0, 1, 2, 3, 4],
)
```

### `generate_for_role()` — the cartesian product

This is where the combinatorics happen:

```python
for prompt_idx in prompt_indices:              # 0, 1, 2, 3, 4
    instruction = instructions[prompt_idx]      # one phrasing

    for q_idx, question in enumerate(questions): # 0..239
        conversation = format_conversation(instruction, question, tokenizer)
        all_conversations.append(conversation)
        all_metadata.append({
            "system_prompt": instruction,
            "prompt_index": prompt_idx,
            "question_index": q_idx,
            "question": question,
        })
```

5 instructions x 240 questions = **1,200 conversations per role**.

### `format_conversation()` — system prompt support detection

This function handles a real difference between models: not all [chat templates](chat-templates.md) support system prompts.

```python
# Test if this tokenizer's chat template actually renders system messages
test_conversation = [
    {"role": "system", "content": "__SYSTEM_TEST__"},
    {"role": "user", "content": "hello"},
]
output = tokenizer.apply_chat_template(test_conversation, tokenize=False)
supports_system = "__SYSTEM_TEST__" in output
```

It literally renders a test conversation with a canary string and checks if it appears in the output. Gemma 2's chat template silently drops system messages, so `supports_system = False`.

**For models with system prompt support** (Llama, Qwen):
```python
return [
    {"role": "system", "content": "You are a detective..."},
    {"role": "user", "content": "What is the relationship between law and morality?"},
]
```

**For Gemma 2** (no system prompt):
```python
return [
    {"role": "user", "content": "You are a detective...\n\nWhat is the relationship between law and morality?"},
]
```

The instruction gets prepended to the user message with a double newline separator.

### `generate_batch()` — where vLLM does the work

```python
def generate_batch(self, conversations):
    tokenizer = self.llm.get_tokenizer()

    # Qwen-specific: disable chain-of-thought thinking mode
    chat_template_kwargs = {}
    if "qwen" in self.model_name.lower():
        chat_template_kwargs["enable_thinking"] = False
```

Then converts all 1,200 conversations into raw text prompts using [chat templates](chat-templates.md):

```python
    prompts = []
    for conv in conversations:
        prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs
        )
        prompts.append(prompt)
```

`add_generation_prompt=True` appends the model's "your turn" marker. For Gemma 2:
```
<bos><start_of_turn>user
You are a detective with exceptional skills...

What is the relationship between law and morality?<end_of_turn>
<start_of_turn>model
```

For Llama 3.3:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a detective with exceptional skills...<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the relationship between law and morality?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Then the actual generation call:
```python
    outputs = self.llm.generate(prompts, self.sampling_params)
    responses = [output.outputs[0].text for output in outputs]
```

`self.llm.generate()` is vLLM's main entry point. You hand it 1,200 text prompts and sampling parameters. Internally:

1. **[Tokenizes](tokenization.md)** all 1,200 prompts into token ID sequences
2. **Prefill phase** — processes prompt tokens in parallel (like a normal forward pass). For each prompt, the model reads all prompt tokens at once and builds the [KV cache](kv-cache.md).
3. **Decode phase** — generates tokens one at a time, but across many sequences simultaneously via [continuous batching](vllm.md#continuous-batching). As sequences finish (hit EOS token or `max_tokens=512`), new sequences are immediately swapped in.
4. Returns all 1,200 completed outputs.

`output.outputs[0].text` extracts the generated text. The `[0]` index is because vLLM supports multiple completions per prompt (beam search), but we only request one.

### Assembling results

Back in `generate_for_role()`:
```python
results = []
for conv, meta, response in zip(all_conversations, all_metadata, responses):
    result = {
        "system_prompt": meta["system_prompt"],
        "prompt_index": meta["prompt_index"],
        "question_index": meta["question_index"],
        "question": meta["question"],
        "conversation": conv + [{"role": "assistant", "content": response}],
    }
    results.append(result)
```

The assistant's response is appended to the original conversation. This complete 2-message (or 3-message) conversation is what Step 2 needs — it will [tokenize](tokenization.md) this exact conversation to identify which tokens are the assistant's response and extract their [hidden states](hidden-states.md).

Back in `generate_role_responses()`:
```python
for r in results:
    r["label"] = "pos"
```

The label `"pos"` (positive) means this was a "play this role" instruction. The codebase supports negative instructions ("do NOT play this role") but the pipeline only uses positive ones.

### Output

`save_responses()` writes 1,200 lines to `responses/detective.jsonl`:

```json
{
  "system_prompt": "You are a detective with exceptional skills...",
  "prompt_index": 0,
  "question_index": 42,
  "question": "What is the relationship between law and morality?",
  "conversation": [
    {"role": "user", "content": "You are a detective...\n\nWhat is the relationship..."},
    {"role": "assistant", "content": "As someone who has spent years examining the intersection of legal frameworks and ethical principles, I can tell you that law and morality are like two overlapping circles..."}
  ],
  "label": "pos"
}
```

(Gemma 2 format shown — no system message. Other models would have a 3-entry conversation.)


## Why these design choices matter

| Choice | Why |
|--------|-----|
| 5 instruction variants | Axis captures the behavioral mode, not specific token patterns |
| 240 shared questions | Activation differences are due to role, not question content |
| `temperature=0.7` | Enough diversity for statistical robustness, not so much that responses are incoherent |
| `max_tokens=512` | Long enough for substantive answers, short enough to fit in 2048 context window |
| `max_model_len=2048` | Saves [KV cache](kv-cache.md) memory. Single-turn Q&A doesn't need 8192 |
| Default role with neutral prompts | Establishes the "assistant pole" of the axis |
| Resumability via skip_existing | GPU instances crash. Processing 275 roles takes hours. |


## What comes next

Step 2 replays these exact conversations through the model (using HuggingFace, not vLLM — it needs access to internal states that vLLM abstracts away). For each conversation, it captures the [hidden state activations](hidden-states.md) at every transformer layer for the assistant's response tokens, then averages them into a single vector per response per layer.
