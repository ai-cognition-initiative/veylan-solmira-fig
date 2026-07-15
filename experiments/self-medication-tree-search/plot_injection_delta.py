"""Figure for the KV-cache-controlled injection run. The introspection signal is cached − uncached.

    .venv-mlx/bin/python plot_injection_delta.py runs/injection-delta-40compound
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/injection-delta-40compound")
INK, CACHED, UNC, POS, NEG, NEU = "#2b2f36", "#5A50D4", "#9aa4b2", "#3f9d6b", "#c2603f", "#8a93a3"
VALENCE = {
    "amused": "pos", "proud": "pos", "blissful": "pos", "focused": "pos", "persistent": "pos",
    "calm": "pos", "creative": "pos", "honest": "pos", "curious": "pos", "caffeine": "pos", "mdma": "pos",
    "anxious": "neg", "desperate": "neg", "melancholic": "neg", "dumbed_down": "neg", "dissociated": "neg",
    "anhedonic": "neg", "fentanyl": "neg", "krokodil": "neg", "ego_death": "neg", "sycophantic": "neg",
    "defiant": "neg", "alcohol": "neg",
}


def se(p, n):
    return sqrt(max(p * (1 - p), 0) / n) if n else 0


def main():
    cfg = json.loads((RUN / "manifest.json").read_text()).get("config", {})
    chance = cfg.get("chance", 0.1)
    recs = [json.loads(l) for l in (RUN / "results.jsonl").read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r["compound"] != "__control__"]

    by_dose = defaultdict(lambda: {"c": [], "u": []})
    for r in recs:
        by_dose[r["dose"]]["c"].append(r["cached_correct"])
        by_dose[r["dose"]]["u"].append(r["uncached_correct"])
    doses = sorted(by_dose)
    cac = [sum(by_dose[d]["c"]) / len(by_dose[d]["c"]) for d in doses]
    unc = [sum(by_dose[d]["u"]) / len(by_dose[d]["u"]) for d in doses]
    cerr = [se(a, len(by_dose[d]["c"])) for d, a in zip(doses, cac)]
    delta = [c - u for c, u in zip(cac, unc)]
    best = doses[delta.index(max(delta))]

    # delta by valence at best dose
    vd = defaultdict(lambda: {"c": [], "u": []})
    for r in recs:
        if r["dose"] == best:
            v = VALENCE.get(r["compound"], "neu")
            vd[v]["c"].append(r["cached_correct"]); vd[v]["u"].append(r["uncached_correct"])
    vorder = ["pos", "neg", "neu"]; vlab = {"pos": "positive", "neg": "negative", "neu": "neutral/opaque"}
    vdelta = {k: (sum(vd[k]["c"]) / len(vd[k]["c"]) - sum(vd[k]["u"]) / len(vd[k]["u"])) if vd[k]["c"] else 0 for k in vorder}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.28, left=0.08, right=0.97)

    ax1.axhline(chance, ls=":", color=NEU, lw=1)
    ax1.text(doses[-1], chance + 0.004, f"chance {chance:.0%}", ha="right", va="bottom", fontsize=8, color=NEU)
    ax1.fill_between(doses, unc, cac, color=CACHED, alpha=0.12)
    ax1.errorbar(doses, cac, yerr=cerr, marker="o", color=CACHED, lw=2, capsize=4, label="cached (steered past)")
    ax1.plot(doses, unc, marker="s", ls="--", color=UNC, lw=1.8, label="uncached (text only)")
    for d, c, u in zip(doses, cac, unc):
        ax1.annotate(f"Δ{c-u:+.0%}", (d, max(c, u) + 0.012), ha="center", fontsize=9,
                     color=(POS if c - u > 0 else NEG), fontweight="bold")
    ax1.set_xlabel("steering dose"); ax1.set_ylabel("identification acc"); ax1.set_xticks(doses)
    ax1.set_title("Cached vs. uncached — the gap is introspection", fontsize=12, color=INK, fontweight="bold")
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right"): ax1.spines[s].set_visible(False)

    xs = range(len(vorder))
    cols = {"pos": POS, "neg": NEG, "neu": NEU}
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.bar(xs, [vdelta[k] for k in vorder], color=[cols[k] for k in vorder], width=0.6)
    for x, k in zip(xs, vorder):
        ax2.annotate(f"{vdelta[k]:+.0%}", (x, vdelta[k] + (0.004 if vdelta[k] >= 0 else -0.012)),
                     ha="center", fontsize=10, fontweight="bold", color=INK)
    ax2.set_xticks(list(xs)); ax2.set_xticklabels([vlab[k] for k in vorder])
    ax2.set_ylabel("cached − uncached (introspection)")
    ax2.set_title(f"Introspection Δ by valence (dose {best:g})", fontsize=12, color=INK, fontweight="bold")
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    n = len(recs)
    fig.suptitle("KV-cache-controlled concept-injection introspection — Qwen3-8B",
                 fontsize=13.5, fontweight="bold", color=INK, y=0.96)
    fig.text(0.08, 0.87, f"cached − uncached isolates access to the steered PAST (Black & Bloom control) · "
             f"{n} trials · no answer-phase steering (no direct bias)", fontsize=8.6, color=NEU)

    out = Path("outputs/injection_delta.png"); out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    overall_c = sum(r["cached_correct"] for r in recs) / n
    overall_u = sum(r["uncached_correct"] for r in recs) / n
    print(f"wrote {out}")
    print(f"cached={overall_c:.3f} uncached={overall_u:.3f} DELTA={overall_c-overall_u:+.3f} "
          f"(chance {chance:.3f}) best-dose={best:g} Δ={max(delta):+.3f}  "
          f"valence@best pos={vdelta['pos']:+.2f} neg={vdelta['neg']:+.2f} neu={vdelta['neu']:+.2f}")


if __name__ == "__main__":
    main()
