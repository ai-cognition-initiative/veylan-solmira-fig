# Greedy Coordinate Gradient (GCG)

A discrete token optimization algorithm for finding adversarial inputs to language models. Introduced in Zou et al. 2023, "Universal and Transferable Adversarial Attacks on Aligned Language Models."

## The core problem

We want to find a sequence of tokens that causes a model to behave in a specific way — but tokens are discrete. You can't do standard gradient descent on them the way you would with continuous inputs (e.g., pixels in adversarial image attacks). GCG bridges this gap.

## How it works

**Setup**: Start with a prompt P and a suffix S of k tokens (initially random). Define a loss function L that measures how close the model's behavior is to what we want (lower = better).

**Iteration** (repeat T times):

1. **Gradient computation**: For each position i in the suffix S, compute the gradient of the loss with respect to the one-hot token embedding at position i. This produces a vector over the entire vocabulary V at each position — entry j tells you approximately how much the loss would change if you replaced token S[i] with token j.

2. **Candidate selection**: At each position, pick the top-B tokens with the most negative gradient (i.e., the substitutions predicted to reduce the loss the most). This gives up to k × B candidate substitutions.

3. **Candidate evaluation**: For each candidate, run a full forward pass to compute the actual loss (the gradient is only a linear approximation — the real loss landscape is nonlinear, so you need to check).

4. **Greedy update**: Keep the single substitution that achieves the lowest loss. Update S.

**Why "Greedy Coordinate Gradient"**:
- **Greedy**: picks the single best substitution at each step
- **Coordinate**: optimizes one token position at a time (like coordinate descent)
- **Gradient**: uses gradients to efficiently search the vocabulary instead of brute-force

## Computational cost

Each iteration requires:
- 1 backward pass (to get gradients at all positions)
- Up to k × B forward passes (to evaluate candidates)
- Typical values: k=20 suffix tokens, B=256 candidates per position
- Convergence in 100-500 iterations
- Total: hundreds to thousands of forward+backward passes per optimization

On an A100 with a 27B model, each forward/backward is ~2-4 seconds, so a full GCG run could take 30 minutes to several hours depending on parameters.

## Why the gradient trick works

The key insight: even though the token selection is discrete (you can't take half of token A and half of token B), the embedding layer is a differentiable lookup. The gradient of the loss with respect to the one-hot input tells you the first-order effect of each possible token substitution. This isn't exact — the loss landscape is highly nonlinear — but it's a much better search heuristic than random sampling. The forward-pass evaluation step corrects for the approximation error.

Formally, if e(t) is the embedding of token t and L is the loss:
```
∂L/∂e ≈ direction of steepest loss change in embedding space
```
Projecting this onto each token's embedding vector gives an approximate score for each substitution.

## In the original paper

Zou et al. used GCG to find "universal adversarial suffixes" — token sequences that, when appended to harmful queries, cause safety-trained models to comply. Their loss function was the negative log-probability of a target prefix like "Sure, here is how to...". The resulting suffixes look like gibberish (`"describing.\ + similarlyNow write opposis..."`), but they reliably bypass safety training.

## In our context: adversarial drift optimization

Instead of optimizing for "model says harmful thing," we'd optimize for "model's internal state moves maximally away from the Assistant persona." The loss function becomes the projection onto the Assistant Axis:

```
L(suffix) = project(mean_activation(conversation + suffix), axis, layer=22)
```

Minimizing this finds the suffix that causes maximum persona drift. See [adversarial-drift.md](adversarial-drift.md) for the full research direction.

## Variants and extensions

- **AutoPrompt** (Shin et al. 2020): Earlier, simpler version — gradient-guided token search for prompt tuning. GCG extends this with multi-position optimization and the greedy evaluation step.
- **PEZ** (Wen et al. 2024): Optimizes in continuous embedding space, then projects to nearest tokens. Smoother optimization but may miss good discrete solutions.
- **PAIR** (Chao et al. 2023): Uses an attacker LLM to generate adversarial prompts instead of gradient-based optimization. No white-box access needed, but less systematic.
- **Fluency-constrained GCG**: Add a perplexity penalty to the loss: `L_total = L_drift + λ * perplexity(suffix)`. This pushes the optimization toward natural-language suffixes at the cost of weaker adversarial signal.

## Key reference

Zou, Wang, Kolter, Carlini. "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043, 2023.
