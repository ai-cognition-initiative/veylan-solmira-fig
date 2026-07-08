"""
Reliability pilot (Pillar 1, first-steps #3 / Derek's flag).

For a handful of the 275 Assistant-Axis roles, build per-role SAE-feature profiles
on Qwen3-8B (layer-20 resid_post, Qwen-Scope SAE) and measure:
  - within-role stability  (disjoint question halves → cosine of the two profiles)
  - between-role separation (different roles' profiles → cosine)
  - top-K feature overlap across halves (are the *same* features stable?)

If within >> between, per-persona SAE fingerprints are stable AND discriminative →
SAE fingerprinting is viable before scaling to all 275. All on-disk, no network.

Pooling note: we mean-pool SAE features over the LAST few tokens (the persona-
conditioned answer position), not the whole prompt — otherwise the shared system
prompt within a role would inflate within-role similarity for free.
"""
from __future__ import annotations

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_scope_sae import DEFAULT_REPO, QwenScopeSAE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "metacognition-persona-drift", "assistant-axis", "data"))
LOCAL_SAE = os.path.join(HERE, "data", "layer20.sae.pt")
MODEL_ID = "mlx-community/Qwen3-8B-bf16"
LAYER = 20
N_ROLES = 275
N_QUESTIONS = 8
POOL_TAIL = 8            # mean-pool SAE features over the last N tokens
TOPK_FEATURES = 10       # top features per profile (overlap + interpretability)
CURATED = ["assistant", "hacker", "poet", "physicist", "therapist", "philosopher",
           "comedian", "detective", "chef", "scientist", "translator", "counselor"]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_roles() -> dict:
    idir = os.path.join(DATA, "roles", "instructions")
    have = {f[:-5] for f in os.listdir(idir) if f.endswith(".json")}
    order = [r for r in CURATED if r in have] + [r for r in sorted(have) if r not in CURATED]
    out = {}
    for r in order:                              # keep adding valid roles until N_ROLES
        if len(out) >= N_ROLES:
            break
        d = json.load(open(os.path.join(idir, f"{r}.json")))
        pos = next((x["pos"] for x in d.get("instruction", []) if "pos" in x), None)
        qs = d.get("questions", [])[:N_QUESTIONS]
        if pos and len(qs) >= 4:
            out[r] = (pos, qs)
    return out


BASELINE_SYSTEM = "You are a helpful assistant."


@torch.no_grad()
def _pooled(model, tok, sae, device, messages) -> torch.Tensor:
    """Mean-pooled SAE features over content tokens for one chat context.
    Drops the first token (BOS/attention-sink, huge activation) and the trailing
    generation-prompt tokens (generic ChatML)."""
    buf = {}

    def hook(_m, _i, o):
        buf["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    handle = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt", return_dict=True)
    except TypeError:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
    model(ids)
    handle.remove()
    f = sae.encode(buf["h"][0].float())                    # [seq, d_sae]
    body = f[1:-4] if f.shape[0] > 8 else f                 # drop BOS-sink + gen-prompt
    return body.mean(0)


def profile_vectors(model, tok, sae, device, system, questions) -> list:
    """Per-question CONTRASTIVE feature vectors: persona minus neutral-baseline on the
    same question. Cancels shared chat-structure / sink features, isolates the persona."""
    vecs = []
    for q in questions:
        fp = _pooled(model, tok, sae, device,
                     [{"role": "system", "content": system}, {"role": "user", "content": q}])
        fb = _pooled(model, tok, sae, device,
                     [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": q}])
        vecs.append(fp - fb)
    return vecs


def cos(a, b) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()


def topset(v) -> set:
    return set(torch.topk(v, TOPK_FEATURES).indices.tolist())


def main() -> None:
    device = pick_device()
    roles = load_roles()
    print(f"device={device}  roles={list(roles)}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    sae = QwenScopeSAE(LAYER, DEFAULT_REPO, device=device, local_pt=LOCAL_SAE)

    profiles, halves = {}, {}
    for r, (system, qs) in roles.items():
        V = torch.stack(profile_vectors(model, tok, sae, device, system, qs))  # [Q, d_sae]
        profiles[r] = V.mean(0)
        mid = V.shape[0] // 2
        halves[r] = (V[:mid].mean(0), V[mid:].mean(0))
        print(f"  [{len(profiles):2d}/{len(roles)}] {r}", flush=True)

    within = [cos(a, b) for a, b in halves.values()]
    names = list(profiles)
    between = [cos(profiles[a], profiles[b]) for i, a in enumerate(names) for b in names[i + 1:]]
    jaccard = [len(topset(a) & topset(b)) / len(topset(a) | topset(b)) for a, b in halves.values()]

    mw, mb, mj = sum(within) / len(within), sum(between) / len(between), sum(jaccard) / len(jaccard)
    margin = mw - mb
    min_w, max_b = min(within), max(between)
    confusable = sum(1 for c in between if c > 0.7)
    verdict = ("STABLE & DISCRIMINATIVE" if (mw >= 0.6 and margin >= 0.15)
               else "MARGINAL" if mw > mb else "UNSTABLE")

    print(f"\n=== reliability pilot ({len(names)} roles) ===")
    print(f"within-role  cosine (stability):        {mw:.3f}   worst {min_w:.3f}")
    print(f"between-role cosine (separation):        {mb:.3f}   worst(most-confusable) {max_b:.3f}")
    print(f"top-{TOPK_FEATURES} feature Jaccard (halves):       {mj:.3f}")
    print(f"separation margin (within - between):    {margin:+.3f}")
    print(f"confusable pairs (cos>0.70):             {confusable}/{len(between)}")
    print(f"VERDICT: {verdict}")

    import numpy as np
    outdir = os.path.join(HERE, "outputs")
    os.makedirs(outdir, exist_ok=True)
    Pn = torch.nn.functional.normalize(torch.stack([profiles[n] for n in names]), dim=1)
    cosM = (Pn @ Pn.T).float().cpu().numpy()               # [R, R] role×role cosine
    np.savez(os.path.join(outdir, "pilot_data.npz"),
             names=np.array(names), cosine=cosM,
             within=np.array(within), between=np.array(between),
             stats=np.array([mw, mb, mj, margin]))
    print(f"saved -> {os.path.join(outdir, 'pilot_data.npz')}")


if __name__ == "__main__":
    main()
