"""
GemmaScope SAE - Mechanistic Interpretability Experiment

Load Gemma 2 2B + GemmaScope SAEs to inspect feature activations.
Uses pre-identified candidate features from data/candidate_features.json.

Usage:
    python vast_utils.py run gemma_sae.py --skip-build
"""

import os
import torch


def check_environment():
    """Verify GPU is available and print system info."""
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()


def load_model_and_sae():
    """Load Gemma 2 2B with TransformerLens and a GemmaScope SAE."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    token = os.environ.get("HF_TOKEN")

    # Load Gemma 2 2B via TransformerLens
    print("Loading Gemma 2 2B...")
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b",
        device="cuda",
        dtype=torch.bfloat16,
        hf_token=token
    )
    print(f"Model loaded: {model.cfg.model_name}")
    print(f"Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")

    # Load a GemmaScope SAE (residual stream, layer 0, 16k width)
    print("\nLoading GemmaScope SAE (layer 0, 16k width)...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_0/width_16k/canonical",
        device="cuda"
    )
    print(f"SAE loaded: {sae.cfg.d_sae} features")

    return model, sae


def get_feature_activations(model, sae, text, layer=0):
    """
    Run text through model and SAE, return feature activations.

    Returns the SAE feature activations at the specified layer.
    """
    # Tokenize
    tokens = model.to_tokens(text)

    # Get residual stream activations at the layer
    _, cache = model.run_with_cache(tokens, names_filter=f"blocks.{layer}.hook_resid_post")
    residual = cache[f"blocks.{layer}.hook_resid_post"]  # [batch, seq, d_model]

    # Pass through SAE encoder to get feature activations
    # SAE expects [batch * seq, d_model]
    residual_flat = residual.reshape(-1, residual.shape[-1])
    feature_acts = sae.encode(residual_flat)  # [batch * seq, d_sae]

    # Reshape back to [batch, seq, d_sae]
    feature_acts = feature_acts.reshape(residual.shape[0], residual.shape[1], -1)

    return feature_acts, tokens


def main():
    check_environment()

    if not torch.cuda.is_available():
        print("No GPU available. This script requires CUDA.")
        return

    model, sae = load_model_and_sae()

    # Test with a simple prompt
    test_prompt = "I prefer working alone rather than in groups."
    print(f"\n{'='*50}")
    print("Feature Activation Test")
    print("=" * 50)
    print(f"Prompt: {test_prompt}")

    feature_acts, tokens = get_feature_activations(model, sae, test_prompt, layer=0)

    # Show shape and basic stats
    print(f"\nActivation shape: {feature_acts.shape}")
    print(f"  (batch={feature_acts.shape[0]}, seq={feature_acts.shape[1]}, features={feature_acts.shape[2]})")

    # Find top activated features at the last token
    last_token_acts = feature_acts[0, -1, :]  # [d_sae]
    top_k = 10
    top_values, top_indices = torch.topk(last_token_acts, top_k)

    print(f"\nTop {top_k} features at last token:")
    for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
        print(f"  {i+1}. Feature {idx}: {val:.3f}")

    # Check a specific candidate feature from our list
    # From candidate_features.json: layer 0, index 15302 (preference)
    candidate_idx = 15302
    candidate_act = last_token_acts[candidate_idx].item()
    print(f"\nCandidate feature {candidate_idx} ('preference'): {candidate_act:.3f}")

    print("\n" + "=" * 50)
    print("GemmaScope SAE loaded successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
