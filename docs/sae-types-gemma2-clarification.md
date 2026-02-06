Summary of what we clarified:

  SAE Types for Gemma-2-2B:
  Type: gemmascope-res-16k
  What it encodes: Residual stream activations
  Width: 16k
  Available: ✅ sae-lens
  We tested: ✅ Yes
  ────────────────────────────────────────
  Type: gemmascope-transcoder-16k
  What it encodes: MLP computation
  Width: 16k
  Available: ✅ circuit-tracer
  We tested: ❌ No
  ────────────────────────────────────────
  Type: clt-hp
  What it encodes: Cross-layer MLP
  Width: ~98k
  Available: ❌ Not downloadable
  We tested: ❌ No
  Key findings:

  1. Our steering.py uses residual SAEs (gemma-scope-2b-pt-res-canonical), not transcoders
  2. The features in candidate_features.json were labeled as transcoder features from Neuronpedia, but we tested them via residual SAEs at the same layer/index
  3. 11 of 16 features (CLT-HP) are not available for download - only viewable on Neuronpedia
  4. GemmaScope 2 (for Gemma 3) has descriptions on Neuronpedia with 50M+ searchable features

  Updated candidate_features.json now includes:
  - SAE type definitions with availability status
  - Per-feature availability and tested flags
  - Important notes about the residual SAE vs transcoder distinction
  - Control experiment results