"""
GemmaScope SAE - Mechanistic Interpretability Experiment

Load Gemma 2 2B + GemmaScope SAEs to inspect feature activations.
Uses pre-identified candidate features from data/candidate_features.json.

Uses transformers + sae-lens directly (no TransformerLens dependency).
This approach supports both Gemma 2 and Gemma 3.

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


def load_model_and_sae(model_name="google/gemma-2-2b", sae_release="gemma-scope-2b-pt-res-canonical", sae_id="layer_0/width_16k/canonical"):
    """
    Load a Gemma model with transformers and a GemmaScope SAE.

    Args:
        model_name: HuggingFace model name (e.g., "google/gemma-2-2b", "google/gemma-3-4b-pt")
        sae_release: SAE release name from sae-lens
        sae_id: Specific SAE ID (layer/width/variant)

    Returns:
        model, tokenizer, sae
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    token = os.environ.get("HF_TOKEN")

    # Load model via transformers
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        token=token,
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    print(f"Model loaded: {model_name}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Layers: {model.config.num_hidden_layers}")

    # Load GemmaScope SAE
    print(f"\nLoading SAE: {sae_release} / {sae_id}...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device="cuda"
    )
    print(f"SAE loaded: {sae.cfg.d_sae} features")

    return model, tokenizer, sae


def get_feature_activations(model, tokenizer, sae, text, layer=0):
    """
    Run text through model and SAE, return feature activations.

    Args:
        model: HuggingFace model with output_hidden_states=True
        tokenizer: HuggingFace tokenizer
        sae: SAE from sae-lens
        text: Input text string
        layer: Which layer's residual stream to analyze

    Returns:
        feature_acts: [batch, seq, d_sae] tensor of SAE feature activations
        tokens: tokenized input
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # Forward pass - get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is tuple of (embedding, layer0, layer1, ..., layerN)
    # So layer 0 residual is at index 1
    residual = outputs.hidden_states[layer + 1]  # [batch, seq, d_model]

    # Pass through SAE encoder to get feature activations
    # SAE expects [batch * seq, d_model]
    residual_flat = residual.reshape(-1, residual.shape[-1])
    feature_acts = sae.encode(residual_flat)  # [batch * seq, d_sae]

    # Reshape back to [batch, seq, d_sae]
    feature_acts = feature_acts.reshape(residual.shape[0], residual.shape[1], -1)

    return feature_acts, inputs


def main():
    check_environment()

    if not torch.cuda.is_available():
        print("No GPU available. This script requires CUDA.")
        return

    # Load Gemma 2 2B + GemmaScope
    model, tokenizer, sae = load_model_and_sae(
        model_name="google/gemma-2-2b",
        sae_release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_0/width_16k/canonical"
    )

    # Test with a simple prompt
    test_prompt = "I prefer working alone rather than in groups."
    print(f"\n{'='*50}")
    print("Feature Activation Test")
    print("=" * 50)
    print(f"Prompt: {test_prompt}")

    feature_acts, inputs = get_feature_activations(model, tokenizer, sae, test_prompt, layer=0)

    # Show shape and basic stats
    print(f"\nActivation shape: {feature_acts.shape}")
    print(f"  (batch={feature_acts.shape[0]}, seq={feature_acts.shape[1]}, features={feature_acts.shape[2]})")

    # Decode tokens for display
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    print(f"Tokens: {tokens}")

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
