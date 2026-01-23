# Matryoshka Sparse Autoencoders

**Reference:** Bussmann et al. (2025). *Learning Multi-Level Features with Matryoshka Sparse Autoencoders.* arXiv:2503.17547

---

## The Problem

Standard SAEs face a size tradeoff:
- **Small dictionary** → misses concepts, incomplete coverage
- **Large dictionary** → feature absorption (abstract features get split into overly-specific ones)

Example: A "female tokens" feature might get absorbed into specific features like "Lily", "Sarah", etc., losing the general concept.

---

## The Solution: Nested Training

Matryoshka SAEs train **multiple nested dictionaries simultaneously**. Each sub-dictionary must reconstruct the input independently:

```
Latents:     [0 -------- 1K] [1K -------- 10K] [10K -------- 100K]
             └─ abstract ─┘  └── medium ────┘  └─── specific ───┘

Loss 1:      Reconstruct using only latents 0-1K
Loss 2:      Reconstruct using only latents 0-10K
Loss 3:      Reconstruct using all latents 0-100K
```

The first latents appear in **all** loss terms, so they must capture broadly applicable features. Later latents only appear in some loss terms, so they can specialize.

---

## Key Benefits

1. **Hierarchical organization** — Early latents = abstract concepts, later latents = specific concepts
2. **Reduced feature absorption** — Abstract features stay intact because they're trained independently
3. **Better disentanglement** — Features more cleanly separated
4. **Scalability** — Can train arbitrarily large SAEs while keeping interpretable high-level features

---

## Tradeoffs

- Minor reconstruction quality tradeoff vs standard SAEs
- But superior on practical tasks (probing, concept erasure)
- Disentanglement advantage **grows with SAE scale**

---

## Why It Matters for GemmaScope 2

GemmaScope 2 uses Matryoshka training, which:
- Fixes reconstruction flaws from GemmaScope 1
- Provides cleaner feature separation for interpretability
- Enables analysis at multiple levels of abstraction

For our work: Better feature quality means more reliable probing of preference-related representations.

---

## References

- [arXiv Paper](https://arxiv.org/abs/2503.17547)
- [Alignment Forum Post](https://www.alignmentforum.org/posts/zbebxYCqsryPALh8C/matryoshka-sparse-autoencoders)
- [GitHub: bartbussmann/matryoshka_sae](https://github.com/bartbussmann/matryoshka_sae)
- [GemmaScope 2 Announcement](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)
