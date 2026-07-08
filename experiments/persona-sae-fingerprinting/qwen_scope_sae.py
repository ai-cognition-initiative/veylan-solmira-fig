"""
Qwen-Scope SAE loader for the persona-fingerprinting experiment.

Loads a single Qwen-Scope residual-stream Top-K SAE and exposes encode / decode /
feature_direction. Qwen-Scope ships raw `layer{n}.sae.pt` checkpoints (NOT a
sae_lens release), so we load them directly. Math mirrors the repo's reference
`app.py`:  encode = TopK(ReLU(x @ W_encᵀ + b_enc)),  hook_point = resid_post
(= the output of `model.model.layers[L]`).

Architecture is read from the repo's config.json — no magic numbers here:
  d_model, d_sae, k, hook_point, base_model.
"""
from __future__ import annotations

import json

import torch
from huggingface_hub import hf_hub_download

DEFAULT_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"


def _load_state_dict(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location=device)


def _orient(t: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Return `t` shaped [dim0, dim1], transposing if it arrives [dim1, dim0]."""
    shape = tuple(t.shape)
    if shape == (dim0, dim1):
        return t
    if shape == (dim1, dim0):
        return t.T
    raise ValueError(f"unexpected weight shape {shape}; want {(dim0, dim1)} or {(dim1, dim0)}")


class QwenScopeSAE:
    """One layer's Qwen-Scope Top-K SAE.

    Args:
        layer:    transformer layer index (resid_post of `model.model.layers[layer]`).
        repo_id:  HF repo of the SAE suite (default = Qwen3-8B-Base, W64K, L0=100).
        device:   torch device for the SAE weights / math ("mps" | "cuda" | "cpu").
        dtype:    compute dtype for the SAE (config says float32; keep it for fidelity).
        pre_bias: if True and b_dec exists, use the Gao-style centered form
                  (x -= b_dec before encode). Default False mirrors app.py.
    """

    def __init__(self, layer: int, repo_id: str = DEFAULT_REPO,
                 device: str = "cpu", dtype: torch.dtype = torch.float32,
                 pre_bias: bool = False, local_pt: str | None = None):
        """local_pt: path to a pre-downloaded `layer{n}.sae.pt` (skips any network)."""
        self.layer = layer
        self.repo_id = repo_id
        self.device = device
        self.dtype = dtype
        self.pre_bias = pre_bias

        cfg = json.load(open(hf_hub_download(repo_id, "config.json")))
        self.d_model = int(cfg["d_model"])
        self.d_sae = int(cfg["d_sae"])
        self.k = int(cfg["k"])
        self.hook_point = cfg.get("hook_point", "resid_post")
        self.base_model = cfg.get("base_model")

        pt_path = local_pt if local_pt else hf_hub_download(repo_id, f"layer{layer}.sae.pt")
        sd = _load_state_dict(pt_path, device)
        self.raw_keys = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}

        self.W_enc = _orient(sd["W_enc"], self.d_model, self.d_sae).to(device, dtype).contiguous()
        self.b_enc = sd["b_enc"].to(device, dtype)
        self.W_dec = _orient(sd["W_dec"], self.d_sae, self.d_model).to(device, dtype).contiguous()
        self.b_dec = sd["b_dec"].to(device, dtype) if "b_dec" in sd else None

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_model] residual-stream activations → sparse features [..., d_sae]."""
        x = x.to(self.device, self.dtype)
        if self.pre_bias and self.b_dec is not None:
            x = x - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        relu = torch.relu(pre)
        vals, idx = torch.topk(relu, self.k, dim=-1)
        f = torch.zeros_like(relu)
        f.scatter_(-1, idx, vals)
        return f

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Sparse features [..., d_sae] → reconstructed activations [..., d_model]."""
        x = f @ self.W_dec
        if self.b_dec is not None:
            x = x + self.b_dec
        return x

    def feature_direction(self, idx: int) -> torch.Tensor:
        """Decoder direction of one feature (for steering / ablation) → [d_model]."""
        return self.W_dec[idx]
