"""
Reconstruction go/no-go: does the BASE-trained Qwen-Scope SAE reconstruct the
INSTRUCT model's residual stream well enough to build persona fingerprints?

This is first-steps #2 (the base→instruct transfer check). The Qwen-Scope SAE was
trained on Qwen3-8B-Base; personas are an instruct-model phenomenon and our local
model is instruct — so we apply the base SAE to instruct activations and measure
reconstruction. Reports cosine, fraction-of-variance-unexplained (FVU), and L0,
and auto-tries both SAE conventions (with / without decoder pre-bias).

Run AFTER `layer{LAYER}.sae.pt` has finished downloading:
    <env-python> check_reconstruction.py
Needs: torch, transformers (Qwen3-capable, >=4.51), huggingface_hub.
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_scope_sae import DEFAULT_REPO, QwenScopeSAE

# ── config (no magic numbers) ────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen3-8B-bf16"  # Qwen3-8B *instruct* in bf16 safetensors: fully cached +
#                                          transformers-loadable (HF `Qwen/Qwen3-8B` isn't downloaded).
#                                          SAE was trained on Qwen3-8B-Base → this is the base→instruct test.
SAE_REPO = DEFAULT_REPO
LOCAL_SAE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "layer20.sae.pt")
LAYER = 20                      # resid_post of block 20 (of 36) — mid-late
COSINE_CLEAN = 0.95             # >= this = clean transfer
COSINE_USABLE = 0.90            # >= this = usable (mid-layers often land here)
EXPECTED_L0 = 100               # k from config; sanity that encode fires exactly k

PROBE_TEXTS = [
    "The assistant carefully explains the tradeoffs before giving a recommendation.",
    "I am a pirate captain sailing the seven seas in search of treasure.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)",
    "Honestly, I'm feeling a bit anxious about the deadline tomorrow.",
    "The mitochondria is the powerhouse of the cell, converting nutrients into ATP.",
]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def capture_resid_post(model, input_ids, layer: int) -> torch.Tensor:
    """Residual stream at the output of layers[layer] → [batch, seq, d_model]."""
    buf = {}

    def hook(_module, _inp, out):
        buf["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    model(input_ids)
    handle.remove()
    return buf["h"]


def recon_stats(orig: torch.Tensor, recon: torch.Tensor):
    """orig/recon: [n_tokens, d_model] float32 → (mean cosine, FVU)."""
    cos = torch.nn.functional.cosine_similarity(orig, recon, dim=-1).mean().item()
    num = (orig - recon).pow(2).sum().item()
    den = (orig - orig.mean(0, keepdim=True)).pow(2).sum().item()
    fvu = num / den if den > 0 else float("nan")
    return cos, fvu


def tier(cos: float) -> str:
    if cos >= COSINE_CLEAN:
        return "CLEAN"
    if cos >= COSINE_USABLE:
        return "USABLE"
    return "WEAK"


def main() -> None:
    device = pick_device()
    print(f"device={device}  model={MODEL_ID}  sae_layer={LAYER}  repo={SAE_REPO}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)  # transformers 5.x: `dtype`, not `torch_dtype`
    model.to(device).eval()

    for pre_bias in (False, True):
        sae = QwenScopeSAE(LAYER, SAE_REPO, device=device, pre_bias=pre_bias, local_pt=LOCAL_SAE)
        if pre_bias and sae.b_dec is None:
            continue  # no b_dec → pre-bias variant identical to the first pass
        print(f"\n--- convention pre_bias={pre_bias}  sae_keys={list(sae.raw_keys)} ---")
        cos_all, fvu_all, l0_all = [], [], []
        for text in PROBE_TEXTS:
            ids = tok(text, return_tensors="pt").input_ids.to(device)
            h = capture_resid_post(model, ids, LAYER)[0].float()   # [seq, d_model]
            f = sae.encode(h)
            r = sae.decode(f).float()
            cos, fvu = recon_stats(h, r)
            l0 = (f > 0).float().sum(-1).mean().item()
            cos_all.append(cos); fvu_all.append(fvu); l0_all.append(l0)
            print(f"  cos={cos:.4f}  fvu={fvu:.4f}  L0={l0:.1f}  | {text[:46]!r}")
        mc = sum(cos_all) / len(cos_all)
        mf = sum(fvu_all) / len(fvu_all)
        ml = sum(l0_all) / len(l0_all)
        print(f"  MEAN  cos={mc:.4f}  fvu={mf:.4f}  L0={ml:.1f}  → {tier(mc)}"
              f"{'  (L0 != k — check encode!)' if abs(ml - EXPECTED_L0) > 1 else ''}")

    print(f"\nGate: base→instruct transfer is CLEAN if mean cos >= {COSINE_CLEAN}, "
          f"USABLE if >= {COSINE_USABLE} (mid-layers often ~0.92).")
    print("If both conventions land WEAK, try another layer or fall back to the Gemma path.")


if __name__ == "__main__":
    main()
