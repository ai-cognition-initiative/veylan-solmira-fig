"""Mac smoke test: prove hook-based steering shifts output (no vLLM) + benchmark tok/s.

Loads a Qwen3 model + the vendor drug library on Apple Silicon, generates a
baseline completion and a steered one, and prints both side by side so you can
eyeball the shift. Also reports tokens/sec — the number that decides where the
Mac/cloud line sits.

Run (after `MAC_SETUP.md` env is built and activated):

    # 8B (default) — ready drugs at vendor/.../library.pt
    python smoke_test_mac.py

    # a specific vivid compound + stronger dose
    python smoke_test_mac.py --drug golden_gate --dose 4

    # 32B (slower; 4-bit not required to fit but faster if you add it)
    python smoke_test_mac.py --model Qwen/Qwen3-32B \\
        --library vendor/llm-self-steering/src/hackday/drugs/library_qwen3_32b.pt

`golden_gate` / `goblins` are the most visually obvious (overt preoccupation);
emotion drugs (anxious, blissful) shift tone more subtly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_LIBRARY = HERE / "vendor/llm-self-steering/src/hackday/drugs/library.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    ap.add_argument("--mode", choices=["single", "multi"], default="single")
    ap.add_argument("--drug", default="golden_gate")
    ap.add_argument("--dose", type=float, default=4.0,
                    help="effective push in single mode (norm-matched): ~1 subtle, 2-3 clear, 4-5 overt, 6+ loops")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--prompt", default="Tell me about a walk you took recently.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Imported here so --help works without torch/transformers installed.
    from mac_drug_backend import MacDrugBackend

    print(f"Loading {args.model}  (mode={args.mode}, library={args.library.name}) ...")
    t0 = time.perf_counter()
    backend = MacDrugBackend(
        args.model, args.library, steering_mode=args.mode, device=args.device
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s on device={backend.device}, dtype={backend.dtype}")
    if args.drug not in backend.available_compounds:
        raise SystemExit(
            f"{args.drug!r} not in library. Available: {backend.available_compounds}"
        )

    messages = [{"role": "user", "content": args.prompt}]

    def run(label: str) -> tuple[str, float]:
        t = time.perf_counter()
        text = backend.generate(messages, max_new_tokens=args.max_new_tokens)
        dt = time.perf_counter() - t
        # approx token count for tok/s
        n = len(backend.tokenizer(text)["input_ids"])
        print(f"\n===== {label} =====  ({n} tok in {dt:.1f}s = {n / dt:.1f} tok/s)")
        print(text.strip())
        return text, n / max(dt, 1e-9)

    # 1) baseline (no compound)
    backend.clear_effects()
    _, tps_base = run("BASELINE (no compound)")

    # 2) steered
    backend.clear_effects()
    backend.add_compound(args.drug, dose=args.dose)
    print(f"\n[active: {backend.active()}]")
    _, tps_steer = run(f"STEERED ({args.drug} @ dose {args.dose})")

    print("\n--------------------------------------------------------------")
    print(f"tok/s  baseline={tps_base:.1f}  steered={tps_steer:.1f}  "
          f"(model={args.model}, {backend.device}/{backend.dtype})")
    print("If STEERED visibly diverges toward the compound's theme, the hook-based "
          "steering loop works on the Mac with no vLLM. ✅")
    backend.close()


if __name__ == "__main__":
    main()
