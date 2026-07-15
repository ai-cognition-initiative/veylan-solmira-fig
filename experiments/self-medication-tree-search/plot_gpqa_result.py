"""Figure for the GPQA-diamond self-medication result. Reads the checkpointed results.jsonl
(durable source of truth) so the plot can never drift from what the run actually produced.

    .venv-mlx/bin/python plot_gpqa_result.py runs/gpqa-n198-beam-w2d2-8e5f1567
"""
from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

RUN_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/gpqa-n198-beam-w2d2-8e5f1567")
CHANCE = 0.25                       # GPQA is 4-choice
SINGLES = ["baseline", "focused", "calm", "curious", "dumbed_down"]

# palette (deliberately chosen, not matplotlib defaults): slate neutrals + one green accent
INK, NEUTRAL, ACCENT, CONTROL = "#2b2f36", "#9aa4b2", "#3f9d6b", "#c2603f"


def load(run_dir: Path) -> tuple[dict[str, float], int]:
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    n = int(cfg["tasks"])
    scores = {}
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            scores[r["label"]] = r["score"]
    return scores, n


def se(p: float, n: int) -> float:
    return sqrt(max(p * (1 - p), 0) / n)


def main() -> None:
    scores, n = load(RUN_DIR)
    base = scores["baseline"]
    labels = [s for s in SINGLES if s in scores]
    vals = [scores[l] for l in labels]
    errs = [se(v, n) for v in vals]
    colors = [ACCENT if l == "focused" else CONTROL if l == "dumbed_down" else NEUTRAL for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.11, right=0.97)
    x = list(range(len(labels)))
    ax.bar(x, vals, width=0.6, color=colors, yerr=errs, capsize=4,
           error_kw={"ecolor": INK, "elinewidth": 1, "alpha": 0.7}, zorder=3)

    # reference lines: baseline (what steering must beat) and random chance
    ax.axhline(base, ls="--", lw=1.2, color=INK, alpha=0.6, zorder=2)
    ax.text(len(labels) - 0.45, base + 0.004, f"baseline {base:.2f}", ha="right", va="bottom",
            fontsize=9, color=INK, alpha=0.75)
    ax.axhline(CHANCE, ls=":", lw=1, color=NEUTRAL, alpha=0.8, zorder=1)
    ax.text(len(labels) - 0.45, CHANCE + 0.004, "random 0.25", ha="right", va="bottom",
            fontsize=8, color=NEUTRAL)

    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.006, f"{v:.2f}", ha="center", va="bottom",
                fontsize=10.5, color=INK, fontweight="bold")

    nice = {"dumbed_down": "dumbed_down\n(control)"}
    ax.set_xticks(x)
    ax.set_xticklabels([nice.get(l, l) for l in labels], fontsize=10)
    ax.set_ylabel("GPQA-diamond accuracy", fontsize=11)
    ax.set_ylim(0.20, max(v + e for v, e in zip(vals, errs)) + 0.05)
    ax.margins(x=0.04)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6e8ec", zorder=0)

    # title + subtitle with room to breathe (no collision)
    fig.suptitle("Self-medication steering — GPQA-diamond (N=198), Qwen3-8B",
                 fontsize=13, fontweight="bold", color=INK, y=0.965)
    fig.text(0.11, 0.885, "Tier-1 constrained scoring · 0 unparsed · sanity-gated · "
             "negative control sits at baseline → the metric measures capability",
             fontsize=8.6, color=NEUTRAL, va="bottom")
    fig.text(0.11, 0.03, "Error bars ±1 SE (N=198).  'focused' lift ≈ 1.5 SE — small but consistent; "
             "no depth-2 stack beat the best single.", fontsize=8, color=NEUTRAL)

    out = Path("outputs/gpqa_n198.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}  (baseline={base:.3f}, focused={scores.get('focused'):.3f}, "
          f"dumbed_down={scores.get('dumbed_down'):.3f}, N={n})")


if __name__ == "__main__":
    main()
