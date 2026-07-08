"""Local activation steering on Apple Silicon — general transformers-hook primitive.

Domain-neutral core for the laptop path: load an HF decoder model on MPS, add
directions to the residual stream at chosen layers via forward-hooks, and
generate. **No experiment vocabulary** — state is a list of named `Direction`s,
each a set of per-layer vectors + a scalar `scale` + a `norm_match` flag.
Reusable across local steering / interp experiments (persona vectors,
Assistant-Axis, self-medication compounds, SAE-feature steering, ...).

Steering math ports vllm-lens `_worker_ext.py`:
    norm_match=False -> h += scale * v_L                 (raw addition)
    norm_match=True  -> h += scale * (‖h‖/‖v_L‖) * v_L   (per-token norm-match)
Directions apply sequentially (later ones see the updated residual).

Scope note: `get_decoder_layers` assumes a standard decoder stack
(`model.model.layers` — Qwen/Llama/Gemma). MoE/hybrid architectures need a tweak
there; it's isolated for that reason.

Intended for promotion to a shared location once validated + reconciled with the
repo's existing Assistant-Axis steering code. Kept in the experiment dir for now
so the smoke test resolves locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_EPS = 1e-6


@dataclass
class Direction:
    """A named steering direction.

    `vectors_by_layer` maps decoder-layer index -> (hidden,) vector, pre-scaled to
    whatever magnitude the caller wants (this primitive does not normalize). The
    direction is applied only at the layers present in this dict. `scale` is a
    scalar multiplier; `norm_match` selects the two math modes above.
    """

    name: str
    vectors_by_layer: dict[int, torch.Tensor]
    scale: float = 1.0
    norm_match: bool = False


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def norm_match(residual: torch.Tensor, steering: torch.Tensor) -> torch.Tensor:
    """v -> (‖h‖/‖v‖)·v, per token. `residual` (..., hidden); `steering` (hidden,)."""
    r_norm = residual.float().norm(dim=-1, keepdim=True)
    v_norm = steering.float().norm()
    return (steering * (r_norm / (v_norm + _EPS))).to(residual.dtype)


def get_decoder_layers(model: Any) -> Any:
    """Locate the decoder-layer ModuleList (Qwen/Llama/Gemma: model.model.layers)."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(
            f"Could not find `.model.layers` on {type(model).__name__}; "
            "extend get_decoder_layers for this architecture."
        )
    return layers


class ActivationSteerer:
    """HF decoder model + tokenizer on MPS with hook-based residual-stream steering.

    Persistent forward-hooks are registered on every decoder layer at load; each
    reads the live direction set (via `_active_layers`), so setting / clearing
    directions is pure state — no hook churn between generations.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = pick_device(device)
        self.dtype = dtype or (torch.bfloat16 if self.device != "cpu" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()

        self._directions: list[Direction] = []
        self._active_layers: set[int] = set()
        self._handles: list[Any] = []
        self._register_hooks()

    # -- direction state ------------------------------------------------------
    def set_directions(self, directions: list[Direction]) -> None:
        self._directions = list(directions)
        self._active_layers = set()
        for d in self._directions:
            self._active_layers.update(d.vectors_by_layer.keys())

    def clear(self) -> None:
        self._directions = []
        self._active_layers = set()

    def directions(self) -> list[Direction]:
        return list(self._directions)

    # -- generate -------------------------------------------------------------
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> str:
        # transformers 5.x returns a BatchEncoding (dict) from apply_chat_template;
        # return_dict=True + generate(**enc) handles input_ids + attention_mask.
        tmpl = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = self.tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **tmpl  # Qwen3-specific; ignored elsewhere
            )
        except TypeError:
            enc = self.tokenizer.apply_chat_template(messages, **tmpl)
        enc = enc.to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        return self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)

    # -- hooks ----------------------------------------------------------------
    def _register_hooks(self) -> None:
        layers = get_decoder_layers(self.model)
        self.n_layers = len(layers)
        for L in range(self.n_layers):
            self._handles.append(layers[L].register_forward_hook(self._make_hook(L)))

    def _make_hook(self, layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any):
            if layer_idx not in self._active_layers:
                return output
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output  # (batch, seq, hidden)
            for d in self._directions:
                v = d.vectors_by_layer.get(layer_idx)
                if v is None:
                    continue
                v = v.to(dtype=hidden.dtype, device=hidden.device)
                add = (norm_match(hidden, v) if d.norm_match else v) * d.scale
                hidden = hidden + add  # sequential: next direction sees updated norm
            return (hidden, *output[1:]) if is_tuple else hidden
        return hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "ActivationSteerer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
