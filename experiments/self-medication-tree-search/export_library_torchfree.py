"""Export the vendor drug library (torch .pt) to a torch-free npz+json.

The MLX env (`.venv-mlx`) has no torch, so it can't `torch.load` library.pt. Run this
ONCE in the torch `.venv` to dump raw per-layer vectors (numpy) + metadata (json) that
the MLX steerer loads without torch. Re-run if the vendor library changes.

    .venv/bin/python export_library_torchfree.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
LIB = HERE / "vendor/llm-self-steering/src/hackday/drugs/library.pt"
OUT_NPZ = HERE / "library_export.npz"
OUT_META = HERE / "library_meta.json"

saved = torch.load(LIB, weights_only=False)
payload = saved["emotion_vectors"]  # v2: {drug: {layer: tensor}} ; v1: {drug: tensor}

arrays: dict[str, np.ndarray] = {}
drugs: dict[str, list[int]] = {}
for name, per_layer in payload.items():
    if isinstance(per_layer, dict):
        layers = sorted(int(L) for L in per_layer)
        for L in layers:
            arrays[f"{name}|{L}"] = per_layer[L].detach().to(torch.float32).cpu().numpy()
        drugs[name] = layers
    else:  # v1 single vector
        arrays[f"{name}|-1"] = per_layer.detach().to(torch.float32).cpu().numpy()
        drugs[name] = [-1]

meta = {
    "drugs": drugs,
    "extraction_layers": saved.get("extraction_layers"),
    "probe_layer": int(saved.get("probe_layer", 24)),
    "descriptions": saved.get("descriptions", {}),
}
np.savez(OUT_NPZ, **arrays)
OUT_META.write_text(json.dumps(meta, indent=2))
print(f"exported {len(drugs)} drugs, {len(arrays)} per-layer vectors -> {OUT_NPZ.name}, {OUT_META.name}")
print("extraction_layers:", meta["extraction_layers"], "| example drug layers:", next(iter(drugs.items())))
