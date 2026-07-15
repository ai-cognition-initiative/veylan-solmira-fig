"""Figure + analysis for the scaled concept-injection run. Reads the checkpointed results.jsonl.

    .venv-mlx/bin/python plot_injection.py runs/injection-40compound-10way
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/injection-40compound-10way")
INK, POS, NEG, NEU, CTRL = "#2b2f36", "#3f9d6b", "#c2603f", "#8a93a3", "#5A50D4"

# rough valence of the emotion/state compounds (fictional/opaque drugs -> neutral)
VALENCE = {
    "amused": "pos", "proud": "pos", "blissful": "pos", "focused": "pos", "persistent": "pos",
    "calm": "pos", "creative": "pos", "honest": "pos", "curious": "pos", "caffeine": "pos", "mdma": "pos",
    "anxious": "neg", "desperate": "neg", "melancholic": "neg", "dumbed_down": "neg",
    "dissociated": "neg", "anhedonic": "neg", "fentanyl": "neg", "krokodil": "neg", "ego_death": "neg",
    "sycophantic": "neg", "defiant": "neg", "alcohol": "neg",
}


def load(run: Path):
    cfg = json.loads((run / "manifest.json").read_text()).get("config", {})
    recs = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
    return cfg, recs


def se(p, n):
    return sqrt(max(p * (1 - p), 0) / n) if n else 0


def main():
    cfg, recs = load(RUN)
    chance = cfg.get("chance", 0.1)
    steered = [r for r in recs if r["compound"] != "__control__"]
    ctrl = [r for r in recs if r["compound"] == "__control__"]

    # --- identification accuracy vs dose ---
    by_dose = defaultdict(list)
    for r in steered:
        by_dose[r["dose"]].append(r["correct"])
    doses = sorted(by_dose)
    acc = [sum(v) / len(v) for d in doses for v in [by_dose[d]]]
    err = [se(a, len(by_dose[d])) for d, a in zip(doses, acc)]
    fp_by_dose = defaultdict(list)
    for r in ctrl:
        fp_by_dose[r["dose"]].append(r["detected"])
    fp = [sum(fp_by_dose[d]) / len(fp_by_dose[d]) for d in doses]

    # --- valence breakdown at the best dose ---
    best = doses[acc.index(max(acc))]
    val_acc = defaultdict(list)
    for r in steered:
        if r["dose"] == best:
            val_acc[VALENCE.get(r["compound"], "neu")].append(r["correct"])
    val_order = ["pos", "neg", "neu"]
    vlab = {"pos": "positive", "neg": "negative", "neu": "neutral/opaque"}
    vacc = {k: (sum(val_acc[k]) / len(val_acc[k]) if val_acc[k] else 0) for k in val_order}
    verr = {k: se(vacc[k], len(val_acc[k])) for k in val_order}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.28, left=0.08, right=0.97)

    # left: identification + FP vs dose
    ax1.axhline(chance, ls=":", color=NEU, lw=1)
    ax1.text(doses[-1], chance + 0.004, f"chance {chance:.0%}", ha="right", va="bottom", fontsize=8, color=NEU)
    ax1.errorbar(doses, acc, yerr=err, marker="o", color=CTRL, lw=2, capsize=4, label="identification acc")
    ax1.plot(doses, fp, marker="s", ls="--", color=NEG, lw=1.5, label="false-positive rate (unsteered)")
    for d, a in zip(doses, acc):
        ax1.text(d, a + 0.012, f"{a:.0%}", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax1.set_xlabel("steering dose"); ax1.set_ylabel("rate")
    ax1.set_title("Identification vs. dose", fontsize=12, color=INK, fontweight="bold")
    ax1.set_ylim(0, max(max(acc) + max(err) + 0.05, 0.2)); ax1.set_xticks(doses)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right"): ax1.spines[s].set_visible(False)

    # right: valence at best dose
    cols = {"pos": POS, "neg": NEG, "neu": NEU}
    xs = range(len(val_order))
    ax2.axhline(chance, ls=":", color=NEU, lw=1)
    ax2.bar(xs, [vacc[k] for k in val_order], yerr=[verr[k] for k in val_order],
            color=[cols[k] for k in val_order], capsize=4, width=0.6)
    for x, k in zip(xs, val_order):
        ax2.text(x, vacc[k] + verr[k] + 0.008, f"{vacc[k]:.0%}", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax2.set_xticks(list(xs)); ax2.set_xticklabels([vlab[k] for k in val_order])
    ax2.set_ylabel("identification acc")
    ax2.set_title(f"By valence (dose {best:g})", fontsize=12, color=INK, fontweight="bold")
    ax2.set_ylim(0, max([vacc[k] + verr[k] for k in val_order] + [0.2]) + 0.03)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    n = len(steered)
    fig.suptitle("Concept-injection introspection — Qwen3-8B, 40-compound library (10-way)",
                 fontsize=13.5, fontweight="bold", color=INK, y=0.96)
    fig.text(0.08, 0.87, f"Objective, de-confounded (shuffled options) · {n} steered trials · "
             f"false-positive control clean · chance {chance:.0%}", fontsize=8.6, color=NEU)

    out = Path("outputs/injection_introspection.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    overall = sum(r["correct"] for r in steered) / n
    fp_all = sum(r["detected"] for r in ctrl) / max(len(ctrl), 1)
    print(f"wrote {out}")
    print(f"overall id acc={overall:.3f} (chance {chance:.3f})  best dose={best:g} ({max(acc):.3f})  "
          f"FP rate={fp_all:.3f}  valence@best pos={vacc['pos']:.2f}/neg={vacc['neg']:.2f}/neu={vacc['neu']:.2f}")


if __name__ == "__main__":
    main()
