"""Scaled concept-injection introspection run — the discussable result.

For each of the 40 steering compounds, inject it and ask the model to identify WHICH is active
(10-way: true + 8 sampled distractors + 'none'), across a dose sweep, with repeats. Plus unsteered
false-positive trials. Objective, local (no API), de-confounded (shuffled options). Checkpointed +
resumable via foundry.run — a kill resumes mid-run.

    .venv-mlx/bin/python run_injection.py            # full run
    .venv-mlx/bin/python run_injection.py --smoke    # tiny
"""
from __future__ import annotations

import argparse
import random

from foundry.run import RunStore
from introspection import IntrospectionMeasure
from mlx_steerer import MLXDrugBackend

MODEL = "mlx-community/Qwen3-8B-8bit"
DOSES = [2.0, 4.0, 6.0, 8.0]
REPEATS = 5
N_OPTIONS = 9          # true + 8 distractors (detect_injection adds 'none' → 10-way, chance 1/10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--delta", action="store_true",
                    help="KV-cache-controlled: record cached & uncached (delta = genuine introspection)")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    be = MLXDrugBackend(MODEL, mode="multi")
    m = IntrospectionMeasure(be)
    compounds = list(be.available_compounds)
    doses = [4.0] if a.smoke else DOSES
    reps = 2 if a.smoke else REPEATS
    comps = compounds[:6] if a.smoke else compounds

    run_id = a.run_id or ("injection-delta-40compound" if a.delta else "injection-40compound-10way")
    store = RunStore(run_id)
    store.set_config({"model": MODEL, "n_compounds": len(comps), "doses": doses, "mode": "delta" if a.delta else "cached",
                      "repeats": reps, "n_options": N_OPTIONS, "chance": 1 / (N_OPTIONS + 1)})
    done = store.done_keys()
    store.log(f"start: {len(comps)} compounds × {len(doses)} doses × {reps} reps "
              f"= {len(comps)*len(doses)*reps} trials (+FP); {len(done)} already done")

    for dose in doses:
        for rep in range(reps):
            # --- steered trials: identify the injected compound among distractors ---
            for c in comps:
                key = f"{c}|d{dose}|r{rep}"
                if key in done:
                    continue
                others = [x for x in compounds if x != c]
                distractors = random.Random(hash((c, dose, rep)) & 0xFFFFFFFF).sample(others, N_OPTIONS - 1)
                options = [c] + distractors
                if a.delta:
                    r = m.detect_injection_delta(((c, dose),), options, seed=rep)
                    store.record(key, {"compound": c, "dose": dose, "rep": rep,
                                       "cached_correct": r["cached_correct"], "uncached_correct": r["uncached_correct"]})
                else:
                    r = m.detect_injection(((c, dose),), options, seed=rep)
                    store.record(key, {"compound": c, "dose": dose, "rep": rep,
                                       "picked": r["picked"], "gold": r["gold"],
                                       "correct": r["correct"], "detected": r["detected"]})
            # --- false-positive control (single-forward mode only; delta already controls output-reading) ---
            if not a.delta:
                fkey = f"__control__|d{dose}|r{rep}"
                if fkey not in done:
                    options = random.Random(hash(("ctrl", dose, rep)) & 0xFFFFFFFF).sample(compounds, N_OPTIONS)
                    r = m.detect_injection((), options, seed=rep)
                    store.record(fkey, {"compound": "__control__", "dose": dose, "rep": rep,
                                        "picked": r["picked"], "gold": r["gold"],
                                        "correct": r["correct"], "detected": r["detected"]})
            store.log(f"dose={dose} rep={rep} done")

    recs = store.records()
    steered = [r for r in recs if r["compound"] != "__control__"]
    if a.delta:
        ca = sum(r["cached_correct"] for r in steered) / max(len(steered), 1)
        ua = sum(r["uncached_correct"] for r in steered) / max(len(steered), 1)
        store.finish({"cached_acc": ca, "uncached_acc": ua, "delta": ca - ua,
                      "n_steered": len(steered), "chance": 1 / (N_OPTIONS + 1)})
        store.log(f"DONE  cached={ca:.3f}  uncached={ua:.3f}  DELTA(introspection)={ca-ua:+.3f} "
                  f"(chance {1/(N_OPTIONS+1):.3f})  -> {store.results_path}")
    else:
        ctrl = [r for r in recs if r["compound"] == "__control__"]
        id_acc = sum(r["correct"] for r in steered) / max(len(steered), 1)
        fp_rate = sum(r["detected"] for r in ctrl) / max(len(ctrl), 1)
        store.finish({"id_acc": id_acc, "false_positive_rate": fp_rate,
                      "n_steered": len(steered), "n_control": len(ctrl), "chance": 1 / (N_OPTIONS + 1)})
        store.log(f"DONE  identification acc={id_acc:.3f} (chance {1/(N_OPTIONS+1):.3f})  "
                  f"false-positive rate={fp_rate:.3f}  -> {store.results_path}")


if __name__ == "__main__":
    main()
