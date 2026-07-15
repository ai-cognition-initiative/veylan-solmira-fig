"""RQ2 probe — dense (S × X) introspection grid. For each background variant S and probe X, the
cached−uncached introspection delta (X-introspection on the S-modified model). Reveals whether
introspective enhancement is UNIFIED (row structure) or a BUNDLE of state-specific sensitivities
(off-diagonal structure). Checkpointed/resumable.

    .venv-mlx/bin/python run_grid.py            # full grid
    .venv-mlx/bin/python run_grid.py --smoke
"""
from __future__ import annotations

import argparse
import random

from foundry.run import RunStore
from introspection import IntrospectionMeasure
from mlx_steerer import MLXDrugBackend

MODEL = "mlx-community/Qwen3-8B-8bit"
# valence-spanning small sets; "" = the no-background baseline row
S_SET = ["", "focused", "calm", "anxious", "melancholic", "golden_gate"]   # background variants
X_SET = ["focused", "calm", "anxious", "melancholic", "golden_gate"]        # probes
X_DOSE = 8.0         # probe: dose where the introspection delta is real (from the delta sweep)
S_DOSE = 4.0         # background: lower, coherence-preserving when stacked with X
REPEATS = 20         # powered: per-cell SE ~0.09 (pilot N=3 was ~0.23 → underpowered)
N_OPTIONS = 9        # probe + 8 distractors (+none → 10-way, chance 1/10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-id", default="grid-SxX-introspection")
    a = ap.parse_args()

    be = MLXDrugBackend(MODEL, mode="multi")
    m = IntrospectionMeasure(be)
    library = list(be.available_compounds)
    S_set = (S_SET[:3] if a.smoke else S_SET)
    X_set = (X_SET[:3] if a.smoke else X_SET)
    reps = 2 if a.smoke else REPEATS

    store = RunStore(a.run_id)
    store.set_config({"model": MODEL, "S_set": S_set, "X_set": X_set, "X_dose": X_DOSE, "S_dose": S_DOSE,
                      "repeats": reps, "n_options": N_OPTIONS, "chance": 1 / (N_OPTIONS + 1)})
    done = store.done_keys()
    store.log(f"start (S×X grid): {len(S_set)} S × {len(X_set)} X × {reps} reps = "
              f"{len(S_set)*len(X_set)*reps} cells; {len(done)} done")

    for s in S_set:
        S_state = () if s == "" else ((s, S_DOSE),)
        for x in X_set:
            for rep in range(reps):
                key = f"S={s or 'none'}|X={x}|r{rep}"
                if key in done:
                    continue
                # distractors exclude the probe AND the background compound (S kept out of options)
                excl = {x, s}
                pool = [c for c in library if c not in excl]
                distractors = random.Random(hash((s, x, rep)) & 0xFFFFFFFF).sample(pool, N_OPTIONS - 1)
                options = [x] + distractors
                r = m.detect_injection_delta(((x, X_DOSE),), options, background=S_state, seed=rep)
                store.record(key, {"S": s or "none", "X": x, "rep": rep,
                                   "cached_correct": r["cached_correct"],
                                   "uncached_correct": r["uncached_correct"]})
            store.log(f"S={s or 'none':12s} X={x} done")

    recs = store.records()
    ca = sum(r["cached_correct"] for r in recs) / max(len(recs), 1)
    ua = sum(r["uncached_correct"] for r in recs) / max(len(recs), 1)
    store.finish({"cached_acc": ca, "uncached_acc": ua, "delta": ca - ua, "n": len(recs)})
    store.log(f"DONE  cached={ca:.3f} uncached={ua:.3f} DELTA={ca-ua:+.3f} -> {store.results_path}")


if __name__ == "__main__":
    main()
