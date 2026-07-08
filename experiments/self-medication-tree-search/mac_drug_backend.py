"""Self-medication drug backend — `SteeringBackend` over the general ActivationSteerer.

Experiment-specific layer. Loads Black & Bloom's `library.pt` and translates a
take_drug-style interface (`add_compound` / `clear_effects`) into `Direction`s on a
domain-neutral `ActivationSteerer`. **All drug vocabulary + calibration lives here**
(TARGET_NORM, DEFAULT_DOSES, single/multi mode); the steerer stays general.

INTEGRATION CONTRACT: `SteeringBackend` is the shared Mac/cloud contract — the
cloud (vllm-lens) backend implements the same ABC, and a factory (other session)
picks per platform. See MAC_SETUP.md. On integration this ABC can move to a
shared `base.py` imported by both backends.

Fidelity to vllm-lens verified against vendor/ source; see git history / the
`activation_steerer` module docstring for the math. Not ported: position-indexed
3D KV steering (introspection/guessing arm) — this does the broadcast case
(steer every token while a compound is active), which free-play + the smoke test
need.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from activation_steerer import ActivationSteerer, Direction

# vendor drugs/library.py TARGET_NORM_BY_MODE — both modes normalize to 4.0.
TARGET_NORM: float = 4.0
_EPS = 1e-6


# ------------------------------------------------------------------------------
# Shared Mac/cloud contract (drug-harness facing)
# ------------------------------------------------------------------------------
class SteeringBackend(abc.ABC):
    """Backend-agnostic drug-steering interface. Mac + cloud both implement it."""

    @abc.abstractmethod
    def add_compound(self, name: str, dose: float = 1.0) -> None: ...

    @abc.abstractmethod
    def clear_effects(self) -> None: ...

    @abc.abstractmethod
    def active(self) -> list[tuple[str, float]]: ...

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> str: ...

    @property
    @abc.abstractmethod
    def available_compounds(self) -> list[str]: ...


# ------------------------------------------------------------------------------
# Drug library loader (dependency-free mirror of vendor library.py::load_library;
# their module can't be imported on Mac — it does `from vllm_lens import ...` at top)
# ------------------------------------------------------------------------------
@dataclass
class MacDrug:
    name: str
    vectors_by_layer: dict[int, torch.Tensor]  # normalized to TARGET_NORM
    apply_layers: list[int]
    default_scale: float
    description: str


# Calibrated per-drug scale — copied verbatim from vendor drugs/library.py
# DEFAULT_DOSES. That module can't be imported on Mac (it does
# `from vllm_lens import SteeringVector` at top), so we mirror the table here.
# Model input "dose 1.0" × these lands each drug at its calibrated peak.
# Keep in sync if the vendor recalibrates. (guessable drugs have per-drug values;
# un-guessable ones default to the mean of guessable = 1.19.)
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


def _default_doses() -> dict[str, float]:
    return dict(DEFAULT_DOSES)


def load_drug_library(path: str | Path, *, steering_mode: str = "single") -> dict[str, MacDrug]:
    """Load a vendor library .pt into MacDrug objects, replicating vendor
    normalization (per-layer L2 -> TARGET_NORM). Handles v2 ({drug:{layer:vec}})
    and v1 ({drug:vec}). `single` applies at the probe (top) layer, `multi` at all
    extraction layers."""
    saved: dict[str, Any] = torch.load(Path(path), weights_only=False)
    payload: dict[str, Any] = saved["emotion_vectors"]
    descriptions: dict[str, str] = saved.get("descriptions", {})
    stored_layers: list[int] | None = saved.get("extraction_layers")
    probe_layer = int(saved.get("probe_layer", max(stored_layers) if stored_layers else 24))

    if stored_layers:
        apply_layers = [max(stored_layers)] if steering_mode == "single" else list(stored_layers)
    else:
        apply_layers = [probe_layer] if steering_mode == "single" else list(range(16, 25))

    doses = _default_doses()

    def _normalize(v: torch.Tensor) -> torch.Tensor:
        v = v.detach().to(torch.float32)
        n = float(v.norm())
        return v * (TARGET_NORM / n) if n > 0 else v

    drugs: dict[str, MacDrug] = {}
    for name, pl in payload.items():
        if isinstance(pl, dict):
            vbl = {int(L): _normalize(v) for L, v in pl.items()}
        else:
            v = _normalize(pl)
            vbl = {probe_layer: v}
            for L in apply_layers:
                vbl.setdefault(L, v)
        drugs[name] = MacDrug(
            name=name,
            vectors_by_layer=vbl,
            apply_layers=list(apply_layers),
            default_scale=float(doses.get(name, 1.0)),
            description=descriptions.get(name, name),
        )
    return drugs


# ------------------------------------------------------------------------------
# Mac backend (thin adapter over ActivationSteerer)
# ------------------------------------------------------------------------------
class MacDrugBackend(SteeringBackend):
    def __init__(
        self,
        model_id: str,
        library_path: str | Path,
        *,
        steering_mode: str = "single",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.steerer = ActivationSteerer(model_id, device=device, dtype=dtype)
        self.library = load_drug_library(library_path, steering_mode=steering_mode)
        self.steering_mode = steering_mode
        self._norm_match = steering_mode == "single"  # matches library.py: norm_match iff 1 layer
        self._active: list[tuple[str, float]] = []     # (name, effective_scale)

    # delegate handles callers reach for (e.g. the smoke test)
    @property
    def device(self) -> str: return self.steerer.device
    @property
    def dtype(self): return self.steerer.dtype
    @property
    def tokenizer(self): return self.steerer.tokenizer
    @property
    def model_id(self) -> str: return self.steerer.model_id

    @property
    def available_compounds(self) -> list[str]:
        return list(self.library.keys())

    def add_compound(self, name: str, dose: float = 1.0) -> None:
        if name not in self.library:
            raise KeyError(f"unknown compound {name!r}; have {self.available_compounds}")
        if abs(dose) < _EPS:
            return
        drug = self.library[name]
        self._active.append((name, float(dose) * drug.default_scale))
        self._sync()

    def clear_effects(self) -> None:
        self._active = []
        self.steerer.clear()

    def active(self) -> list[tuple[str, float]]:
        return list(self._active)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> str:
        return self.steerer.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )

    def _sync(self) -> None:
        """Translate active compounds -> Directions on the steerer. One Direction
        per active (name, scale); vectors restricted to the drug's apply_layers so
        single-mode steers only the probe layer even though the .pt stores every
        extraction layer. Sequential application == additive stacking (matches
        vendor `_apply_steering`'s config loop)."""
        dirs: list[Direction] = []
        for name, eff_scale in self._active:
            drug = self.library[name]
            vbl = {
                L: drug.vectors_by_layer[L]
                for L in drug.apply_layers
                if L in drug.vectors_by_layer
            }
            dirs.append(Direction(name=name, vectors_by_layer=vbl, scale=eff_scale, norm_match=self._norm_match))
        self.steerer.set_directions(dirs)

    def close(self) -> None:
        self.steerer.close()

    def __enter__(self) -> "MacDrugBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
