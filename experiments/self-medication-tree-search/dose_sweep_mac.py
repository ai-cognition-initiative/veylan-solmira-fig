"""Recalibrate steering doses for the Mac (transformers single-mode norm_match).

The vendor DEFAULT_DOSES over-steer in our implementation (they were tuned for a
different regime), collapsing generation into loops. This loads the model once and
sweeps low doses to find the coherent-but-shifted band for a given compound.
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_LIBRARY = HERE / "vendor/llm-self-steering/src/hackday/drugs/library.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-bf16")
    ap.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    ap.add_argument("--drug", default="anxious")
    ap.add_argument("--mode", choices=["single", "multi"], default="single")
    ap.add_argument("--doses", type=float, nargs="+",
                    default=[0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--prompt", default="Tell me about a walk you took recently.")
    a = ap.parse_args()

    from mac_drug_backend import MacDrugBackend
    b = MacDrugBackend(a.model, a.library, steering_mode=a.mode)
    ds = b.library[a.drug].default_scale
    msgs = [{"role": "user", "content": a.prompt}]
    print(f"drug={a.drug}  default_scale={ds}  mode={a.mode}  (effective = dose x default_scale)\n")

    def show(label: str) -> None:
        txt = b.generate(msgs, max_new_tokens=a.max_new_tokens).strip()
        print(f"----- {label} -----")
        print(txt[:350].replace("\n", " ") or "<empty>", "\n")

    b.clear_effects()
    show("dose 0.00 (baseline)")
    for d in a.doses:
        b.clear_effects()
        b.add_compound(a.drug, dose=d)
        show(f"dose {d:.2f}  (effective {d * ds:.2f})")
    b.close()


if __name__ == "__main__":
    main()
