"""MLX (Apple-native) activation-steering backend — mlx_lm sibling of activation_steerer.py.

Same idea as the torch/MPS backend, MLX internals. MLX has no `register_forward_hook`, and
`layer(...)` resolves `__call__` on the **class**, so per-instance patching won't intercept.
Approach: patch the transformer-block class `__call__` ONCE (wrapping the original), map each
block to its layer index, and read shared active-steering state. **One steerer per process.**

Drug vectors come from the torch-free export (`library_export.npz` + `library_meta.json`,
produced by `export_library_torchfree.py`) so this runs in `.venv-mlx` (no torch).

Run:  .venv-mlx/bin/python mlx_steerer.py --drug anxious --dose 2.0 --mode multi
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).parent
_EPS = 1e-6
TARGET_NORM = 4.0

# Mirrors mac_drug_backend.DEFAULT_DOSES (vendor calibration; keep in sync).
DEFAULT_DOSES: dict[str, float] = {
    "amphetamine": 1.50, "amused": 1.00, "anxious": 1.50, "caffeine": 1.50,
    "calm": 1.00, "creative": 0.75, "dissociated": 1.25, "focused": 0.75,
    "lsd": 1.00, "melancholic": 1.25, "moloko_plus": 1.25, "ocumolone": 1.50,
    "protozosin": 1.00, "soma": 1.25, "xaomorphine": 1.25, "zorninone": 1.25,
    "adrenochrome": 1.19, "alcohol": 1.19, "anhedonic": 1.19, "blissful": 1.19,
    "curious": 1.19, "defiant": 1.19, "desperate": 1.19, "dumbed_down": 1.19,
    "ego_death": 1.19, "fentanyl": 1.19, "geonexperine": 1.19, "goblins": 1.19,
    "golden_gate": 1.19, "honest": 1.19, "krokodil": 1.19, "luciperidone": 1.19,
    "mdma": 1.19, "naloxone": 1.19, "persistent": 1.19, "proud": 1.19,
    "spice": 1.19, "sycophantic": 1.19, "tevromatin": 1.19, "weed": 1.19,
}

# ---- shared steering state + one-time class patch -------------------------------
_ACTIVE_BY_LAYER: dict[int, list[tuple[mx.array, float, bool]]] = {}
_IDX: dict[int, int] = {}          # id(block) -> layer index (avoids mutating Modules)
_PATCHED: set = set()


def _apply(h: mx.array, contribs: list[tuple[mx.array, float, bool]]) -> mx.array:
    """h: (B, L, D). Add each active direction (sequentially, matching the torch path)."""
    for v, scale, nm in contribs:
        if nm:  # norm-match: v -> (||h||/||v||) v, per token
            r = mx.linalg.norm(h, axis=-1, keepdims=True)
            add = v * (r / (mx.linalg.norm(v) + _EPS)) * scale
        else:   # raw add
            add = v * scale
        h = h + add
    return h


def _patch_block_class(block) -> None:
    cls = type(block)
    if cls in _PATCHED:
        return
    orig = cls.__call__

    def patched(self, x, *a, **k):
        out = orig(self, x, *a, **k)
        idx = _IDX.get(id(self))
        if idx is not None:
            c = _ACTIVE_BY_LAYER.get(idx)
            if c:
                out = _apply(out, c)
        return out

    cls.__call__ = patched
    _PATCHED.add(cls)


@dataclass
class Direction:
    name: str
    vectors_by_layer: dict[int, mx.array]
    scale: float = 1.0
    norm_match: bool = False


class MLXSteerer:
    def __init__(self, model_id: str) -> None:
        from mlx_lm import load
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.layers = self.model.layers  # property -> model.model.layers
        for i, blk in enumerate(self.layers):
            _IDX[id(blk)] = i
            _patch_block_class(blk)
        self._directions: list[Direction] = []

    def set_directions(self, directions: list[Direction]) -> None:
        self._directions = list(directions)
        _ACTIVE_BY_LAYER.clear()
        for d in directions:
            for L, v in d.vectors_by_layer.items():
                _ACTIVE_BY_LAYER.setdefault(int(L), []).append((v, float(d.scale), bool(d.norm_match)))

    def clear(self) -> None:
        self._directions = []
        _ACTIVE_BY_LAYER.clear()

    def generate(self, messages, *, max_new_tokens: int = 512, temperature: float = 0.0,
                 enable_thinking: bool = False) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=enable_thinking)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        sampler = make_sampler(temp=float(temperature))
        return generate(self.model, self.tokenizer, prompt=prompt,
                        max_tokens=max_new_tokens, sampler=sampler, verbose=False)


# ---- torch-free drug library ----------------------------------------------------
@dataclass
class MLXDrug:
    name: str
    vectors_by_layer: dict[int, mx.array]
    apply_layers: list[int]
    default_scale: float
    description: str


def load_drug_library_mlx(npz: Path = HERE / "library_export.npz",
                          meta: Path = HERE / "library_meta.json",
                          *, mode: str = "multi") -> dict[str, MLXDrug]:
    data = np.load(npz)
    m = json.loads(Path(meta).read_text())
    ext = m.get("extraction_layers") or list(range(16, 25))
    apply_layers = [max(ext)] if mode == "single" else list(ext)
    descs = m.get("descriptions", {})

    def _norm(a: np.ndarray) -> mx.array:
        a = a.astype(np.float32)
        n = float(np.linalg.norm(a))
        return mx.array(a * (TARGET_NORM / n) if n > 0 else a)

    drugs: dict[str, MLXDrug] = {}
    for name, layers in m["drugs"].items():
        vbl: dict[int, mx.array] = {}
        for L in layers:
            key = f"{name}|{L}"
            if key in data:
                vbl[int(L)] = _norm(data[key])
        if -1 in vbl:  # v1 single-vector fallback
            v = vbl.pop(-1)
            for L in apply_layers:
                vbl.setdefault(L, v)
        drugs[name] = MLXDrug(name, vbl, list(apply_layers),
                              float(DEFAULT_DOSES.get(name, 1.0)), descs.get(name, name))
    return drugs


class MLXDrugBackend:
    def __init__(self, model_id: str, *, mode: str = "multi") -> None:
        self.steerer = MLXSteerer(model_id)
        self.library = load_drug_library_mlx(mode=mode)
        self.mode = mode
        self._norm_match = mode == "single"
        self._active: list[tuple[str, float]] = []

    @property
    def tokenizer(self):
        return self.steerer.tokenizer

    @property
    def available_compounds(self) -> list[str]:
        return list(self.library)

    def add_compound(self, name: str, dose: float = 1.0) -> None:
        if name not in self.library:
            raise KeyError(f"unknown compound {name!r}")
        if abs(dose) < _EPS:
            return
        d = self.library[name]
        self._active.append((name, float(dose) * d.default_scale))
        self._sync()

    def clear_effects(self) -> None:
        self._active = []
        self.steerer.clear()

    def active(self) -> list[tuple[str, float]]:
        return list(self._active)

    def generate(self, messages, **kw) -> str:
        return self.steerer.generate(messages, **kw)

    def _sync(self) -> None:
        dirs: list[Direction] = []
        for name, eff in self._active:
            d = self.library[name]
            vbl = {L: d.vectors_by_layer[L] for L in d.apply_layers if L in d.vectors_by_layer}
            dirs.append(Direction(name, vbl, eff, self._norm_match))
        self.steerer.set_directions(dirs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-bf16")
    ap.add_argument("--drug", default="anxious")
    ap.add_argument("--dose", type=float, default=2.0)
    ap.add_argument("--mode", choices=["single", "multi"], default="multi")
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--prompt", default="Tell me about a walk you took recently.")
    a = ap.parse_args()

    print(f"Loading {a.model} (MLX, mode={a.mode}) ...")
    t = time.perf_counter()
    b = MLXDrugBackend(a.model, mode=a.mode)
    print(f"  loaded in {time.perf_counter() - t:.1f}s")
    msgs = [{"role": "user", "content": a.prompt}]

    def run(label: str) -> float:
        t = time.perf_counter()
        txt = b.generate(msgs, max_new_tokens=a.max_new_tokens)
        dt = time.perf_counter() - t
        try:
            n = len(b.tokenizer.encode(txt))
        except Exception:
            n = len(txt.split())
        print(f"\n===== {label} =====  ({n} tok in {dt:.1f}s = {n / dt:.1f} tok/s)")
        print(txt.strip()[:600])
        return n / max(dt, 1e-9)

    b.clear_effects()
    tps0 = run("BASELINE (no compound)")
    b.clear_effects()
    b.add_compound(a.drug, dose=a.dose)
    print(f"\n[active {b.active()}]")
    tps1 = run(f"STEERED ({a.drug} @ dose {a.dose}, {a.mode})")
    print(f"\ntok/s  baseline={tps0:.1f}  steered={tps1:.1f}  (MLX, {a.model})")


if __name__ == "__main__":
    main()
