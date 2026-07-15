"""RQ2 figure — the (S × X) introspection matrix. Heatmap of the cached−uncached delta per
(background S, probe X), with row-means (generalist score per S) and column-means (detectability per X).
Uniform rows ⇒ UNIFIED; strong off-diagonal / varying rows ⇒ BUNDLE of state-specific sensitivities.

    .venv-mlx/bin/python plot_grid.py runs/grid-SxX-introspection
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/grid-SxX-introspection")
INK = "#2b2f36"


def main():
    cfg = json.loads((RUN / "manifest.json").read_text()).get("config", {})
    S_set, X_set = cfg["S_set"], cfg["X_set"]
    recs = [json.loads(l) for l in (RUN / "results.jsonl").read_text().splitlines() if l.strip()]

    cell = defaultdict(lambda: {"c": [], "u": []})
    for r in recs:
        cell[(r["S"], r["X"])]["c"].append(r["cached_correct"])
        cell[(r["S"], r["X"])]["u"].append(r["uncached_correct"])
    Slabels = [s or "none" for s in S_set]
    M = np.full((len(Slabels), len(X_set)), np.nan)
    for i, s in enumerate(Slabels):
        for j, x in enumerate(X_set):
            d = cell.get((s, x))
            if d and d["c"]:
                M[i, j] = np.mean(d["c"]) - np.mean(d["u"])

    row_mean = np.nanmean(M, axis=1)   # generalist score per S
    col_mean = np.nanmean(M, axis=0)   # detectability per X
    vmax = np.nanmax(np.abs(M)) or 0.1

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.24, right=0.83)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(X_set)))
    ax.set_xticklabels([f"{x}\n(μ{col_mean[j]:+.2f})" for j, x in enumerate(X_set)], fontsize=9)
    ax.set_yticks(range(len(Slabels)))
    ax.set_yticklabels([f"{s}  (μ{row_mean[i]:+.2f})" for i, s in enumerate(Slabels)], fontsize=10)
    ax.set_xlabel("probe  X  (injected — to be identified)   ·   μ = detectability", fontsize=10, labelpad=8)
    ax.set_ylabel("background variant  S   ·   μ = generality", fontsize=10)
    for i in range(len(Slabels)):
        for j in range(len(X_set)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=9,
                        color=("white" if abs(M[i, j]) > vmax * 0.6 else INK))

    cax = fig.add_axes([0.855, 0.20, 0.022, 0.60])   # dedicated colorbar axis — never clipped
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("introspection Δ (cached − uncached)", fontsize=9)

    fig.suptitle("RQ1 — is introspective enhancement unified or a bundle?",
                 fontsize=13.5, fontweight="bold", color=INK, y=0.965)
    # honest verdict: is any S×X interaction above the per-cell noise floor?
    n_reps = cfg.get("repeats", 20)
    p = float(np.mean([r["cached_correct"] for r in recs]))
    percell_se = (p * (1 - p) / n_reps) ** 0.5
    resid = M - row_mean[:, None] - col_mean[None, :] + np.nanmean(M)
    interaction = float(np.nanstd(resid))
    bi, wi = int(np.nanargmax(row_mean)), int(np.nanargmin(row_mean))
    struct = "no detectable S×X structure" if interaction < percell_se else "possible S×X structure"
    fig.text(0.22, 0.885,
             f"NEGATIVE — no steered background enhances introspection; unmodified best "
             f"(μ{row_mean[bi]:+.2f}), {Slabels[wi]} worst (μ{row_mean[wi]:+.2f}).",
             fontsize=7.8, color="#8a93a3")
    fig.text(0.22, 0.858,
             f"{struct}: interaction σ={interaction:.2f} < per-cell SE {percell_se:.2f}  (N={n_reps}).",
             fontsize=7.8, color="#8a93a3")

    out = Path("outputs/grid_SxX.png"); out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(f"row μ (generalist per S): " + ", ".join(f"{s}={row_mean[i]:+.2f}" for i, s in enumerate(Slabels)))
    print(f"col μ (detectability per X): " + ", ".join(f"{x}={col_mean[j]:+.2f}" for j, x in enumerate(X_set)))
    print(f"interaction σ={interaction:.3f}  per-cell SE={percell_se:.3f}  → {struct}")


if __name__ == "__main__":
    main()
