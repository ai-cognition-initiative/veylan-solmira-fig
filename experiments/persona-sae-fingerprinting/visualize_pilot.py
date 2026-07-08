"""
Visualize the reliability-pilot result (loads outputs/pilot_data.npz — no model).

Two panels:
  (A) role x role fingerprint-similarity heatmap, hierarchically ordered so
      persona families sit together (sequential single-hue = magnitude).
  (B) within-role vs between-role cosine distributions (CVD-safe categorical pair)
      — the separation that is the actual result.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "outputs", "pilot_data.npz")
OUT = os.path.join(HERE, "outputs", "pilot_viz.png")

# Okabe-Ito CVD-safe categorical pair (identity, fixed order)
C_WITHIN = "#0072B2"   # blue  = within-role (stability)
C_BETWEEN = "#E69F00"  # orange = between-role (separation)
SEQ_CMAP = "Blues"     # sequential single hue, light(low)->dark(high) = magnitude


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    M = d["cosine"]                       # [R, R] role-profile cosine
    within, between = d["within"], d["between"]
    mw, mb, mj, margin = d["stats"]
    R = len(names)

    # hierarchical order so similar roles are adjacent (block structure visible)
    Dm = 1.0 - M
    np.fill_diagonal(Dm, 0.0)
    Dm = (Dm + Dm.T) / 2.0
    order = leaves_list(linkage(squareform(Dm, checks=False), method="average"))
    Mo = M[np.ix_(order, order)]
    labels = [names[i] for i in order]

    fig, (axH, axD) = plt.subplots(1, 2, figsize=(16, 8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle(f"Persona SAE fingerprints — reliability pilot ({R} roles, Qwen3-8B layer 20)",
                 fontsize=15, fontweight="bold")

    # (A) similarity heatmap
    im = axH.imshow(Mo, cmap=SEQ_CMAP, vmin=0.2, vmax=1.0, aspect="equal")
    if R <= 60:
        axH.set_xticks(range(R)); axH.set_xticklabels(labels, rotation=90, fontsize=6)
        axH.set_yticks(range(R)); axH.set_yticklabels(labels, fontsize=6)
    else:
        axH.set_xticks([]); axH.set_yticks([])
        axH.set_xlabel(f"{R} roles (hierarchically clustered; labels omitted for legibility)", fontsize=9)
    axH.set_title("Fingerprint similarity (cosine), roles clustered", fontsize=11)
    for s in axH.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axH, fraction=0.046, pad=0.04)
    cb.set_label("cosine similarity", fontsize=9)

    # (B) within vs between distributions
    bins = np.linspace(0, 1, 41)
    axD.hist(between, bins=bins, density=True, color=C_BETWEEN, alpha=0.75,
             label=f"between-role (n={len(between)})")
    axD.hist(within, bins=bins, density=True, color=C_WITHIN, alpha=0.85,
             label=f"within-role (n={len(within)})")
    axD.axvline(float(mb), color=C_BETWEEN, lw=2, ls="--")
    axD.axvline(float(mw), color=C_WITHIN, lw=2, ls="--")
    axD.set_xlabel("cosine similarity", fontsize=10)
    axD.set_ylabel("density", fontsize=10)
    axD.set_title(f"Stability vs separation  (margin +{float(margin):.2f})", fontsize=11)
    axD.legend(frameon=False, fontsize=10, loc="upper center")
    axD.grid(axis="y", color="0.9", lw=0.8)
    for s in ("top", "right"):
        axD.spines[s].set_visible(False)
    axD.annotate(f"within  mean {float(mw):.3f}", (float(mw), axD.get_ylim()[1] * 0.9),
                 color=C_WITHIN, ha="right", fontsize=9, xytext=(-6, 0), textcoords="offset points")
    axD.annotate(f"between mean {float(mb):.3f}", (float(mb), axD.get_ylim()[1] * 0.75),
                 color="#B37400", ha="left", fontsize=9, xytext=(6, 0), textcoords="offset points")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
