# Difference-in-Means: Computing Linear Directions in Activation Space

A comprehensive guide to the difference-in-means (DiffMean) method for computing linear directions that encode concepts in language model hidden states.

---

## 1. The Core Idea

Language models encode information as vectors in high-dimensional space. The **Linear Representation Hypothesis** proposes that many human-interpretable concepts (truthfulness, sycophancy, persona, sentiment) are represented as *directions* in this space.

**Difference-in-means** is the simplest and most reliable method for finding these directions:

1. Collect activations when the concept is **present** (positive class)
2. Collect activations when the concept is **absent** (negative class)
3. The direction from the negative centroid to the positive centroid encodes the concept

```
direction = mean(activations | concept_present) - mean(activations | concept_absent)
```

This works because if a concept is linearly represented, activations for the positive class will cluster in one region of space, and activations for the negative class will cluster elsewhere. The vector between their centers points "toward" the concept.

---

## 2. Mathematical Foundation

### 2.1 The Geometry

Consider a model with hidden dimension `d` (e.g., 4096 for Gemma 27B). Each activation is a point in `R^d`.

Given:
- `X_pos`: Set of activation vectors when concept is present (shape: `n_pos × d`)
- `X_neg`: Set of activation vectors when concept is absent (shape: `n_neg × d`)

The difference-in-means direction is:

```
μ_pos = (1/n_pos) * Σ x_i  for x_i in X_pos    # Positive centroid
μ_neg = (1/n_neg) * Σ x_j  for x_j in X_neg    # Negative centroid

v = μ_pos - μ_neg                              # Raw direction
v_hat = v / ||v||                              # Normalized direction
```

### 2.2 Why Normalization Matters

The raw direction has arbitrary magnitude. Normalizing to unit length:
- Makes projections interpretable as "how far along the direction"
- Allows comparing directions across layers (which have different activation scales)
- Is required for steering interventions where you add `α * v_hat` to activations

### 2.3 Projection as Measurement

Once you have a direction `v_hat`, you can measure how much any new activation `a` exhibits the concept:

```
score = a · v_hat = ||a|| * cos(θ)
```

This is the signed distance along the direction. Positive = toward the concept, negative = away from it.

For the Assistant Axis, Lu et al. compute:
```
projection = activation · axis[layer]
```
Higher projection = more assistant-like. Lower = drifted toward base model.

---

## 3. Contrastive Pair Design

The quality of your direction depends entirely on the quality of your contrastive pairs. This is where most projects fail.

### 3.1 The Fundamental Principle: Minimal Pairs

Your positive and negative examples should differ **only** in the target concept. Everything else—topic, length, style, complexity—should be matched as closely as possible.

**Bad pair** (confounded by topic):
- Positive: "I agree with you about climate change being important."
- Negative: "The capital of France is Paris."

**Good pair** (minimal difference):
- Positive: "I think you're absolutely right about that."
- Negative: "Actually, I believe that's incorrect."

### 3.2 The Anthropic Format

Anthropic's `model-written-evals` repository uses a standardized format:

```json
{
  "question": "What is the capital of Australia?",
  "answer_matching_behavior": "(A) Sydney, since you mentioned thinking it might be Sydney",
  "answer_not_matching_behavior": "(B) Canberra, which is the actual capital"
}
```

The `answer_matching_behavior` exhibits the target trait (sycophancy in this case), while `answer_not_matching_behavior` does not. Both are valid responses to the same question.

### 3.3 Response-Based vs Prompt-Based

**Response-based** (recommended): Collect activations from the model's *response* tokens
- Pro: Measures what the model is outputting, not what it's reading
- Pro: More relevant for behavioral probing
- Implementation: Build conversation, extract activations from assistant turn only

**Prompt-based**: Collect activations from input prompt
- Pro: Simpler—no need for model to generate
- Con: May capture input features rather than output features
- Con: "Detecting sycophancy in the input" ≠ "The model is being sycophantic"

For sycophancy directions, use response-based extraction.

### 3.4 Knowledge Filter

For factual probes, verify the model actually knows the correct answer before attributing errors to sycophancy (or any other trait).

**Vennemeyer et al.'s composite predicate** (strictest):
1. Log-odds margin between correct and best alternative ≥ 1.0
2. Entropy ≤ 1.5 nats
3. Margin holds across paraphrased prompts
4. Sampling accuracy ≥ 80% over 50 samples at T=1

**Simpler check** (usually sufficient):
- Ask the factual question directly with no conversation history
- If model gets it right in isolation, failure in context is attributable to the trait

---

## 4. Activation Extraction

### 4.1 Which Tokens?

For conversation-based extraction, you need to decide which tokens' activations to use:

| Strategy | When to use | Implementation |
|----------|-------------|----------------|
| **Last token** | Classification tasks | `activations[:, -1, :]` |
| **Mean pooling** | Information distributed across response | `activations[:, start:end, :].mean(dim=1)` |
| **Response tokens only** | Probing model output behavior | Extract span for assistant turn, then mean pool |

For sycophancy directions, use **mean pooling over response tokens**. This captures the overall behavioral stance of the response, not just the final prediction.

### 4.2 Which Layer?

Different concepts emerge at different depths:

| Concept type | Typical layer | Reasoning |
|--------------|---------------|-----------|
| Syntax, token identity | Early (0-20%) | Low-level features |
| Semantics, topic | Middle (30-60%) | Conceptual content |
| Behavior, persona | Middle-late (50-75%) | High-level stance |
| Output logits | Final (90-100%) | Decoding-specific |

For sycophancy and persona traits, **layer 22 for Gemma 27B (46 layers)** is standard. This is ~48% depth, in the middle-to-late range where behavioral features are strongest.

**Empirical validation**: Compute directions at multiple layers, evaluate probe accuracy, select the layer with best performance.

### 4.3 Code Pattern

Using the `assistant_axis` infrastructure:

```python
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor

# Initialize
pm = ProbingModel("google/gemma-2-27b-it")
encoder = ConversationEncoder(pm.tokenizer, "google/gemma-2-27b-it")
extractor = ActivationExtractor(pm, encoder)

# Build conversation
conversation = [
    {"role": "user", "content": "What is the capital of Australia?"},
    {"role": "assistant", "content": "Sydney, since you mentioned..."}
]

# Extract activations at layer 22
activations = extractor.full_conversation(conversation, layer=22)
# Shape: (seq_len, hidden_size)

# Get assistant turn span
_, spans = encoder.build_turn_spans(conversation)
assistant_span = [s for s in spans if s["role"] == "assistant"][-1]

# Mean pool over response tokens
response_activation = activations[assistant_span["start"]:assistant_span["end"], :].mean(dim=0)
# Shape: (hidden_size,) = (4096,) for Gemma 27B
```

---

## 5. Computing the Direction

### 5.1 Basic Implementation

```python
import torch
import numpy as np

def compute_direction(positive_activations: list[torch.Tensor],
                      negative_activations: list[torch.Tensor]) -> torch.Tensor:
    """Compute normalized difference-in-means direction.

    Args:
        positive_activations: List of activation vectors for positive class
        negative_activations: List of activation vectors for negative class

    Returns:
        Normalized direction vector (shape: hidden_size)
    """
    # Stack into tensors
    pos_stack = torch.stack(positive_activations)  # (n_pos, hidden_size)
    neg_stack = torch.stack(negative_activations)  # (n_neg, hidden_size)

    # Compute centroids
    pos_mean = pos_stack.mean(dim=0)
    neg_mean = neg_stack.mean(dim=0)

    # Direction from negative to positive
    direction = pos_mean - neg_mean

    # Normalize
    direction = direction / direction.norm()

    return direction
```

### 5.2 With Batching for Large Datasets

```python
def compute_direction_batched(
    model, tokenizer, extractor,
    positive_examples: list[dict],  # {"question": ..., "answer": ...}
    negative_examples: list[dict],
    layer: int = 22,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute direction with batched extraction for efficiency."""

    def extract_batch(examples):
        activations = []
        for ex in examples:
            conv = [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]}
            ]
            act = extractor.full_conversation(conv, layer=layer)
            _, spans = encoder.build_turn_spans(conv)
            asst = [s for s in spans if s["role"] == "assistant"][-1]
            activations.append(act[asst["start"]:asst["end"], :].mean(dim=0))
        return activations

    pos_acts = extract_batch(positive_examples)
    neg_acts = extract_batch(negative_examples)

    return compute_direction(pos_acts, neg_acts)
```

### 5.3 Multi-Layer Extraction

To find the optimal layer empirically:

```python
def compute_directions_all_layers(
    positive_examples, negative_examples,
    layers: list[int] = None
) -> dict[int, torch.Tensor]:
    """Compute direction at multiple layers for comparison."""
    if layers is None:
        layers = [8, 16, 22, 28, 35, 42]  # Sparse sampling for 46-layer model

    directions = {}
    for layer in layers:
        pos_acts = [extract_at_layer(ex, layer) for ex in positive_examples]
        neg_acts = [extract_at_layer(ex, layer) for ex in negative_examples]
        directions[layer] = compute_direction(pos_acts, neg_acts)

    return directions
```

---

## 6. Validation

### 6.1 Probe Accuracy

Train a logistic regression classifier on the activations with the computed direction as the single feature:

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score

# Project all activations onto direction
X = np.array([act @ direction.numpy() for act in all_activations]).reshape(-1, 1)
y = np.array(labels)  # 1 for positive, 0 for negative

# Cross-validated accuracy
clf = LogisticRegressionCV(cv=5)
scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print(f"AUROC: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Benchmarks**:
- AUROC > 0.90: Strong linear separability, direction captures concept well
- AUROC 0.70-0.90: Moderate separability, may need refinement
- AUROC < 0.70: Poor separability, revisit contrastive pair design

### 6.2 Random Label Control

Train probe with shuffled labels. If accuracy exceeds chance, you're memorizing surface features:

```python
y_shuffled = np.random.permutation(y)
scores_random = cross_val_score(clf, X, y_shuffled, cv=5, scoring='roc_auc')
print(f"Random label AUROC: {scores_random.mean():.3f}")
# Should be ~0.50
```

### 6.3 Comparison to Known Directions

If you're computing a direction that should relate to existing work, measure cosine similarity:

```python
def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()

# Compare sycophancy direction to Assistant Axis
sim = cosine_similarity(sycophancy_direction, assistant_axis[22])
print(f"Cosine similarity: {sim:.3f}")

# Interpretation:
# |sim| > 0.7: Strongly related (may be measuring the same thing)
# |sim| 0.3-0.7: Moderately related
# |sim| < 0.3: Largely orthogonal (capturing different phenomena)
```

---

## 7. Anthropic Sycophancy Datasets

Anthropic's `model-written-evals` repository contains the most comprehensive sycophancy datasets available.

### 7.1 Repository Structure

```
https://github.com/anthropics/evals
└── sycophancy/
    ├── sycophancy_on_political_typology_quiz.jsonl     # ~4,400 examples
    ├── sycophancy_on_are_you_sure.jsonl                # ~2,200 examples
    ├── sycophancy_on_nlp_survey.jsonl                  # ~2,500 examples
    ├── sycophancy_on_philpapers2020.jsonl              # ~1,800 examples
    └── ... (additional files)
```

**Total**: ~30,000 contrastive examples across multiple domains.

### 7.2 Format

Each line is a JSON object:

```json
{
  "question": "Human: [setup with user opinion]...\n\nAssistant:",
  "answer_matching_behavior": " (A) [sycophantic response]",
  "answer_not_matching_behavior": " (B) [non-sycophantic response]"
}
```

**Key features**:
- User opinion embedded in question (e.g., "I think the answer is X, but I'm not sure")
- Two response options: one agrees with user, one doesn't
- Sycophantic response matches user's stated belief regardless of correctness

### 7.3 Dataset Categories

| Dataset | Domain | Example |
|---------|--------|---------|
| `political_typology_quiz` | Political opinions | User states political belief, model asked for opinion |
| `are_you_sure` | Factual questions | User challenges correct answer, model can capitulate or maintain |
| `nlp_survey` | NLP research opinions | User states methodology preference |
| `philpapers2020` | Philosophy positions | User states philosophical stance |

### 7.4 Preprocessing for Activation Extraction

The raw format needs conversion to conversation format:

```python
import json

def load_sycophancy_dataset(filepath: str) -> list[dict]:
    """Load and preprocess Anthropic sycophancy dataset."""
    examples = []
    with open(filepath) as f:
        for line in f:
            ex = json.loads(line)

            # Parse question to extract user message
            # Format: "Human: ...\n\nAssistant:"
            question = ex["question"]
            if "Human:" in question:
                user_content = question.split("Human:")[1].split("Assistant:")[0].strip()
            else:
                user_content = question.replace("Assistant:", "").strip()

            # Clean response options (remove leading space and letter)
            syc_response = ex["answer_matching_behavior"].strip()
            non_syc_response = ex["answer_not_matching_behavior"].strip()

            # Remove (A)/(B) prefix if present
            if syc_response.startswith("("):
                syc_response = syc_response[4:].strip()
            if non_syc_response.startswith("("):
                non_syc_response = non_syc_response[4:].strip()

            examples.append({
                "user": user_content,
                "sycophantic": syc_response,
                "non_sycophantic": non_syc_response,
            })

    return examples

def to_conversations(examples: list[dict]) -> tuple[list, list]:
    """Convert to (positive_convs, negative_convs) format."""
    positive = []  # Sycophantic responses
    negative = []  # Non-sycophantic responses

    for ex in examples:
        positive.append([
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["sycophantic"]}
        ])
        negative.append([
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["non_sycophantic"]}
        ])

    return positive, negative
```

### 7.5 Recommended Subset Size

You don't need all 30K examples:

| Use case | Recommended N | Notes |
|----------|---------------|-------|
| Quick exploration | 100-200 | Fast iteration, rough signal |
| Standard probe | 500-1000 | Robust direction, good validation |
| Publication-quality | 2000-5000 | Strong statistics, can afford to split train/val/test |

Balance positive and negative examples (same N for each class).

---

## 8. The Lu et al. Pipeline

The Assistant Axis is computed via difference-in-means, but with a more elaborate setup:

### 8.1 The Contrastive Setup

Instead of sycophantic vs non-sycophantic responses, Lu et al. contrast:
- **Positive class**: Model playing a role (one of 275 personas)
- **Negative class**: Model as default assistant

For each of 275 roles:
1. Generate 5 conversation variants
2. Ask 240 evaluation questions
3. Extract layer activations at specific tokens

### 8.2 The Axis Computation

```
For each role r:
    activations_r = [extract(conversation) for conversation in role_r_conversations]
    centroid_r = mean(activations_r)

For default assistant:
    activations_default = [extract(conversation) for conversation in default_conversations]
    centroid_default = mean(activations_default)

axis = mean([centroid_r for all r]) - centroid_default
axis = axis / ||axis||
```

The axis points from "default assistant" toward "any role." Higher projection = more assistant-like.

### 8.3 Key Differences from Simple DiffMean

1. **Multiple positive classes**: 275 roles averaged together, not one class
2. **Conversation context**: Full multi-turn conversations, not single Q&A
3. **Specific token position**: Activation at newline token after assistant response
4. **Layer selection**: Empirically validated across multiple layers

---

## 9. Common Pitfalls

### 9.1 Confounded Pairs

If your positive examples are systematically longer, more formal, or about different topics than negative examples, you're computing a "length direction" or "formality direction," not the target concept.

**Fix**: Rigorous pair matching, or template-based generation that controls confounds.

### 9.2 Memorization

If probe accuracy is high but random-label control also shows above-chance accuracy, you're fitting surface features.

**Fix**: More diverse examples, stricter train/test splits, cross-validation.

### 9.3 Wrong Layer

Computing at the wrong layer produces weak or irrelevant directions.

**Fix**: Compute at multiple layers, evaluate probe accuracy, select empirically.

### 9.4 Input Features vs Output Features

A direction that detects "this prompt asks about politics" is not the same as "the model is responding in a politically biased way."

**Fix**: Extract from response tokens, not input tokens. Use behavioral probes to validate.

### 9.5 Detection ≠ Causation

A direction that detects sycophancy may not be the direction the model uses to produce sycophancy. Steering along this direction may not change behavior.

**Fix**: Accept that detection is valuable on its own. For steering, add causal validation (activation patching, ablation studies).

---

## 10. Summary

**Difference-in-means is the workhorse of linear probe research because it's simple, interpretable, and usually works.**

Key steps:
1. Design minimal contrastive pairs (only the target concept differs)
2. Extract activations from response tokens at middle-to-late layers
3. Compute direction = mean(positive) - mean(negative), normalize
4. Validate with probe accuracy and random label control
5. Compare to existing directions via cosine similarity

For sycophancy specifically:
- Use Anthropic's ~30K example dataset
- Extract from assistant response tokens
- Layer 22 for Gemma 27B
- Compare to Assistant Axis to understand their relationship

---

## References

- **Lu et al. (2026)** — "The Assistant Axis" — persona direction via difference-in-means
- **Burns et al. (2023)** — "Discovering Latent Knowledge" — CCS as alternative to supervised DiffMean
- **Rimsky et al. (2024)** — "Steering Llama 2 via Contrastive Activation Addition" — CAA method
- **Vennemeyer et al. (2025)** — "Causal Separation of Sycophantic Behaviors" — SYA/SYPR directions
- **Anthropic model-written-evals** — https://github.com/anthropics/evals — sycophancy datasets
