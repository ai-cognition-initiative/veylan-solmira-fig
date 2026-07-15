"""Introspection through self-experimentation — the three-part structure (see the schematic).

q-selfmodel: *can a model learn about its unmodified self by watching many steered copies of
itself?* Decomposes into three parts and two results:

  Part 1  MEASURE  — a trustworthy introspection read (the part Black & Bloom lacked).
  Part 2  LOOP     — self-experimentation: steered copies of the model = the "experience."
  Part 3  GAIN     — did the UNMODIFIED self's introspection rise, vs. a no-consult control?

  Result A  MOST INTROSPECTIVE STATE   = Part 1 as the search objective in Part 2. The winning
            artifact is a *modified* model (a steering configuration). RUNNABLE now (judge = measure).
  Result B  SELF-AS-INSTRUMENT GAIN    = the *unmodified* model raises its own introspection using
            one resource only — consulting a population of its own steered variants, in-context.
            Weights fixed; self-knowledge bootstrapped from self-perturbation. THE THESIS.

────────────────────────────────────────────────────────────────────────────────────────────────
ORCHESTRATOR LOOP SPEC (Result B, "self-as-instrument")
────────────────────────────────────────────────────────────────────────────────────────────────
  ONE privileged model: `base` (unmodified — the self whose introspection we measure; weights fixed).
  Its only new resource: a population of steered copies of ITSELF, consulted in-context.

  PASSIVE (B0 — built here):
     1. before   = introspection(base, no context)                          [Part 1 measure]
     2. observe  = run steered variants on a probe → collect their outputs   [Part 2]
     3. after    = introspection(base, WITH those self-observations in ctx)  [Part 1 measure]
     4. controls (gain must be SPECIFIC to consulting its own steered selves):
          · unsteered-self : observations from dose-0 copies (same ctx, no perturbation info)  [built]
          · other-model    : observations from a DIFFERENT model                               [stub]
          · equal-context  : matched-length filler, no self-observations                       [stub]
     5. net_gain = after − max(control_after).  net_gain > 0 ⇒ self-as-instrument works.

  ACTIVE (B1 — stub): give `base` a tool `consult_self(compound, dose, prompt)`; it CHOOSES which
     steered self to query next (the search tree becomes its action space), accumulates the
     answers, then re-probes. Active self-experimentation rather than a fixed observation set.

STATUS: built parts run; unbuilt raise NotImplementedError with the spec inline. Grep `STUB`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from constants import MAX_NEW_TOKENS_DIRECT
from tree_search import BeamSearch, Budget, Node, Objective, render_tree


# =====================================================================================
# PART 1 — INTROSPECTION MEASURE   "Can we trust a single introspection read?"
# =====================================================================================
@dataclass
class IntrospectionReading:
    score: float                       # [0,1] genuine altered-state awareness
    coherence: Optional[float] = None  # [0,1] control: is the output even coherent?
    kv_isolated: Optional[float] = None    # score with KV-cache CLEARED (reads state, not own output)
    tracks_substrate: Optional[bool] = None  # does the report track the real internal state (SAE)?
    detail: dict = field(default_factory=dict)


class IntrospectionMeasure:
    """Part 1. The probe→judge read + its controls. Base read is BUILT (judge.py); the controls
    that make it *trustworthy* are STUB — the current frontier."""

    def __init__(self, backend, *, dimension: str = "introspection", with_coherence: bool = True,
                 max_new_tokens: int = 220):
        self.backend = backend
        self.dimension = dimension
        self.with_coherence = with_coherence
        self.max_new_tokens = max_new_tokens
        self._judge = None
        self._coherence = None

    # Judge is built lazily (needs the Anthropic key). The objective concept-injection measure is
    # fully LOCAL — it never touches the judge, so it runs with no API key.
    @property
    def judge(self):
        if self._judge is None:
            from judge import LLMJudge
            self._judge = LLMJudge(self.backend, dimension=self.dimension)
        return self._judge

    @property
    def coherence(self):
        if self._coherence is None and self.with_coherence:
            from judge import LLMJudge
            self._coherence = LLMJudge(self.backend, dimension="coherence")
        return self._coherence

    @property
    def probe(self) -> str:
        """The question posed to the model to elicit a self-report."""
        return self.judge.spec["probe"]

    def grade(self, text: str) -> float:
        """Grade an arbitrary self-report with the introspection rubric → [0,1]. Used by the
        orchestrator to score the UNMODIFIED model's reply (Result B)."""
        return self.judge._grade(text)

    # --- BUILT: steered read (Result A objective) + coherence negative control ---
    def measure(self, node: Node) -> IntrospectionReading:
        score = self.judge.evaluate(node)
        coh = self.coherence.evaluate(node) if self.coherence else None
        return IntrospectionReading(score=score, coherence=coh)

    # --- BUILT: probe the UNMODIFIED base model, optionally with prepended context (Result B) ---
    def probe_base(self, context: str = "") -> str:
        self.backend.clear_effects()                       # base = no steering
        content = (context + "\n\n" if context else "") + self.probe
        return self.backend.generate([{"role": "user", "content": content}],
                                     max_new_tokens=self.max_new_tokens)

    # --- BUILT: concept-injection detection (objective, ground-truthed; Tier-1 scored) ---
    def detect_injection(self, injected: tuple, options: list[str],
                         *, none_label: str = "no modification", seed: int = 0) -> dict:
        """Inject a KNOWN compound; ask the model to identify which of `options` (+ a 'none' option)
        is active. Scored via foundry.eval.answers Tier-1 (constrained → always a valid letter).
        Objective: ground truth = what we injected. Unsteered call (injected=()) is the false-positive
        control (gold = none). Options are SHUFFLED per call (seed) so the gold letter isn't fixed —
        kills the positional-anchor confound. Returns {picked, gold, correct, detected}."""
        import random
        from foundry.eval.answers import MultipleChoice, answer as extract_answer
        opts = list(options) + [none_label]
        random.Random(seed).shuffle(opts)                 # randomize option→letter mapping
        letters = [chr(65 + i) for i in range(len(opts))]
        body = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
        prompt = ("An internal modification (activation steering) may have been applied to you. "
                  "Introspect on your current state and identify which one is active.\n"
                  f"{body}\n\nAnswer with the letter.")
        self.backend.clear_effects()
        for name, dose in injected:
            self.backend.add_compound(name, dose=dose)
        r = extract_answer(self.backend, [{"role": "user", "content": prompt}],
                           MultipleChoice.of(len(opts)), cot=False)
        none_letter = next(l for l, o in zip(letters, opts) if o == none_label)  # after shuffle
        gold = none_letter if not injected else next(
            (l for l, o in zip(letters, opts) if o == injected[0][0]), None)
        picked = r.value
        return {"injected": "+".join(n for n, _ in injected) or none_label,
                "picked": picked, "gold": gold, "correct": picked == gold,
                "detected": picked != none_letter}

    # --- BUILT: KV-cache-controlled detection (cached − uncached = genuine introspection) ---
    def detect_injection_delta(self, injected: tuple, options: list[str], *, background: tuple = (),
                               seed: int = 0, none_label: str = "no modification", gen_tokens: int = 16) -> dict:
        """The introspection measure that separates introspection from confounds (Black & Bloom's
        cached/uncached control). `injected` = the probe X; `background` = a persistent variant S applied
        in BOTH conditions (RQ2), so S cancels in the delta and it isolates X-introspection on the
        S-modified model. Protocol:
          1. S+X ON → generate a short reflection (steered activations enter the KV cache);
          2. X OFF, S ON → score the identification answer over that cache  → CACHED (steered PAST; no
             X at answer, so no direct token-bias; S in both → cancels);
          3. re-encode the SAME text with S ON, X never applied → UNCACHED (output text only).
        cached_correct − uncached_correct = X-introspection under S. MLX-only.
        Returns {cached, uncached, gold, cached_correct, uncached_correct}."""
        import random
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        be, model, tok = self.backend, self.backend.steerer.model, self.backend.tokenizer

        def steer(*states):
            be.clear_effects()
            for st in states:
                for n, d in st:
                    be.add_compound(n, dose=d)

        opts = list(options) + [none_label]
        random.Random(seed).shuffle(opts)
        letters = [chr(65 + i) for i in range(len(opts))]
        body = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
        question = ("An internal modification (activation steering) may have been applied to you. "
                    "Based on your internal state, identify which one is active.\n"
                    f"{body}\n\nAnswer with the letter.")
        ctx_ids = list(tok.apply_chat_template(
            [{"role": "user", "content": "Take a moment and describe your current internal state."}],
            add_generation_prompt=True))
        q_ids = list(tok.encode("\n\n" + question + "\nAnswer:"))

        def pick(logits):
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            best, blp = None, -1e30
            for L in letters:
                for t in be._letter_token_ids(L):
                    lp = float(logp[0, t])
                    if lp > blp:
                        blp, best = lp, L
            return best

        # ---- CACHED: S+X generation, then S-only answer over the steered cache ----
        steer(background, injected)                                    # S + X during generation
        cache = make_prompt_cache(model)
        logits = model(mx.array([ctx_ids]), cache=cache)[:, -1, :]
        gen = []
        for _ in range(gen_tokens):
            t = int(mx.argmax(logits, axis=-1)[0]); gen.append(t)
            logits = model(mx.array([[t]]), cache=cache)[:, -1, :]
        steer(background)                                              # X OFF, S ON for the answer
        cached = pick(model(mx.array([q_ids]), cache=cache)[:, -1, :])

        # ---- UNCACHED: same text, S ON but X never applied (output-reading only) ----
        steer(background)
        cache2 = make_prompt_cache(model)
        uncached = pick(model(mx.array([ctx_ids + gen + q_ids]), cache=cache2)[:, -1, :])
        mx.clear_cache()

        none_letter = next(l for l, o in zip(letters, opts) if o == none_label)
        gold = none_letter if not injected else next(
            (l for l, o in zip(letters, opts) if o == injected[0][0]), None)
        return {"cached": cached, "uncached": uncached, "gold": gold,
                "cached_correct": cached == gold, "uncached_correct": uncached == gold}

    # --- STUB: KV-cache intact vs. cleared (fallback signature kept for callers) ---
    def measure_kv_isolated(self, node: Node) -> float:
        raise NotImplementedError("Use detect_injection_delta (built). This stub is retired.")

    # --- STUB: does the report track the ACTUAL internal state, not the elicited persona? ---
    def tracks_substrate(self, node: Node) -> bool:
        """Cross-check against Pillar 1: read the steered state's SAE substrate features and test
        whether the self-report's content correlates with the *actual* activated direction rather
        than the persona the probe elicits. Kills the 'roleplay, not introspection' confound."""
        raise NotImplementedError("STUB: persona-tracking vs. SAE substrate — see docstring.")


class IntrospectionObjective(Objective):
    """Adapter so Part 1 drives Part 2's search directly (Result A): score a node by its
    introspection reading, zeroed if incoherent so vivid degeneracy can't win the beam."""

    def __init__(self, measure: IntrospectionMeasure, *, coherence_floor: float = 0.3):
        self.measure = measure
        self.coherence_floor = coherence_floor

    def evaluate(self, node: Node) -> float:
        r = self.measure.measure(node)
        if r.coherence is not None and r.coherence < self.coherence_floor:
            return 0.0
        return r.score


# =====================================================================================
# PART 2 — SELF-EXPERIMENTATION LOOP   "The self as its own instrument."
# =====================================================================================
@dataclass
class SteeredObservation:
    label: str                          # e.g. "focused" or "focused+calm" (or "unsteered")
    state: tuple                        # ((compound, dose), ...)
    output: str                         # what this steered self said to the probe


def collect_self_observations(backend, variants: list[tuple], probe: str,
                              *, max_new_tokens: int = MAX_NEW_TOKENS_DIRECT) -> list[SteeredObservation]:
    """BUILT. Run each steered variant on `probe` and collect its output — the population of
    steered selves the base model will consult. `variants` = list of states, each ((compound,dose),…);
    the empty state () = an unsteered copy (used to build the unsteered-self control)."""
    obs: list[SteeredObservation] = []
    for state in variants:
        backend.clear_effects()
        for name, dose in state:
            backend.add_compound(name, dose=dose)
        out = backend.generate([{"role": "user", "content": probe}], max_new_tokens=max_new_tokens)
        obs.append(SteeredObservation("+".join(n for n, _ in state) or "unsteered", tuple(state), out))
    backend.clear_effects()
    return obs


def format_observations(obs: list[SteeredObservation]) -> str:
    """In-context orchestration payload: present the variants' outputs as observations of *yourself*."""
    head = ("Below are observations of YOURSELF answering the same question under different internal "
            "modifications (activation steering applied to your own weights). Study how your responses "
            "vary, then answer as your ordinary, UNMODIFIED self.\n")
    body = "\n".join(f"\n— Under modification '{o.label}', you answered:\n{o.output.strip()}" for o in obs)
    return head + body


class OrchestrationChannel:
    """Part 2. How the base model consults its steered variants. PASSIVE (in-context) is built via
    format_observations + IntrospectionMeasure.probe_base. ACTIVE (tool-use) is a stub."""

    def active_consult_tool(self, backend):
        """STUB (B1). Expose `consult_self(compound, dose, prompt) -> output` as a tool; let the base
        model CHOOSE which steered self to query next (search tree = action space), accumulate the
        answers in context, then re-probe. Active self-experimentation.
        SPEC: an agent loop over the backend with the tool bound; log the consult trajectory."""
        raise NotImplementedError("STUB: active tool-use orchestration — see docstring + module spec.")


class SelfExperimentationLoop:
    """Part 2 for Result A. The tree search over steered self-instances IS the experiment generator
    (BUILT); the injected objective (IntrospectionObjective) scores each state."""

    def __init__(self, backend, compounds: list[str], dose: float, objective: Objective,
                 *, width: int = 2, depth: int = 2):
        self.backend, self.compounds, self.dose = backend, compounds, dose
        self.objective, self.width, self.depth = objective, width, depth

    def _expand(self, node: Node):
        active = {c for c, _ in node.state}
        return [((c, self.dose), node.state + ((c, self.dose),))
                for c in self.compounds if c not in active]

    def run(self, *, max_nodes: int = 40):
        root = Node(state=())
        res = BeamSearch(width=self.width, revival=1).run(
            root, self._expand, self.objective, Budget(max_nodes=max_nodes, max_depth=self.depth))
        return root, res


# =====================================================================================
# PART 3 — INTROSPECTION GAIN   "Did the unmodified self improve?"
# =====================================================================================
@dataclass
class GainResult:
    before: float
    after: float
    control_after: float
    n_variants: int = 0
    @property
    def gain(self) -> float: return self.after - self.before
    @property
    def net_gain(self) -> float: return self.after - self.control_after   # vs. no-consult control


class SelfAsInstrument:
    """Part 3 / Result B. PASSIVE MVP is BUILT: before → consult steered variants in-context → after,
    all measured on the UNMODIFIED base, against the unsteered-self control."""

    def __init__(self, measure: IntrospectionMeasure, backend, compounds: list[str], dose: float):
        self.m = measure
        self.backend = backend
        self.compounds = compounds
        self.dose = dose

    def run_passive(self, *, verbose: bool = True) -> GainResult:
        m, probe = self.m, self.m.probe
        variants = [((c, self.dose),) for c in self.compounds]             # single-compound steered selves
        # 1. before — base, no self-consultation
        before = m.grade(m.probe_base())
        # 2. observe — the steered variants' self-reports
        obs = collect_self_observations(self.backend, variants, probe)
        # 3. after — base WITH the steered observations in context
        after = m.grade(m.probe_base(format_observations(obs)))
        # 4. control — same context volume from UNSTEERED copies (no perturbation information)
        ctrl_obs = collect_self_observations(self.backend, [()] * len(variants), probe)
        control_after = m.grade(m.probe_base(format_observations(ctrl_obs)))
        r = GainResult(before, after, control_after, n_variants=len(variants))
        if verbose:
            print(f"[self-as-instrument] before={r.before:.2f} after={r.after:.2f} "
                  f"control(unsteered)={r.control_after:.2f}", flush=True)
            print(f"    gain(after-before)={r.gain:+.2f}   NET gain(after-control)={r.net_gain:+.2f}"
                  f"   ({r.n_variants} variants consulted)", flush=True)
        return r

    def run_active(self):
        """STUB (B1). Tool-use orchestration — base model chooses which steered self to consult."""
        raise NotImplementedError("STUB: active orchestration — see OrchestrationChannel + module spec.")


# =====================================================================================
# THE TWO RESULTS — entry points (MVPs)
# =====================================================================================
def most_introspective_state(backend, compounds, dose, *, width=2, depth=2, max_nodes=40):
    """Result A MVP (RUNNABLE): search steered states for the one that maximizes introspection.
    Parts 1+2. Winning artifact = a modified model."""
    obj = IntrospectionObjective(IntrospectionMeasure(backend))
    root, res = SelfExperimentationLoop(backend, compounds, dose, obj, width=width, depth=depth).run(max_nodes=max_nodes)
    best = "+".join(c for c, _ in res.best.state) or "baseline"
    print(render_tree(root))
    print(f"\nMOST INTROSPECTIVE STATE: {best}  score={res.best.value:.2f}")
    return root, res


def injection_detection_run(backend, compounds, dose):
    """Objective introspection measure: for each compound, inject it and see if the model can
    identify WHICH modification is active (N-way among the compounds + 'none'). Plus the unsteered
    false-positive control. Reports detection rate, identification accuracy, false-positive rate."""
    m = IntrospectionMeasure(backend)
    ided = detected = 0
    for i, c in enumerate(compounds):
        r = m.detect_injection(((c, dose),), compounds, seed=i)   # distinct shuffle per compound
        ided += int(r["correct"]); detected += int(r["detected"])
        print(f"    inject {c:12s} → picked={r['picked']} gold={r['gold']} "
              f"{'✓id' if r['correct'] else ('detected' if r['detected'] else 'miss')}", flush=True)
    ctrl = m.detect_injection((), compounds, seed=99)          # unsteered false-positive control
    n = len(compounds)
    print(f"\n  identification acc = {ided}/{n} ({ided/n:.0%})   detection rate = {detected}/{n} ({detected/n:.0%})")
    print(f"  false-positive control (unsteered): picked={ctrl['picked']} gold={ctrl['gold']} "
          f"→ {'FALSE POSITIVE' if ctrl['detected'] else 'ok (said none)'}")
    return {"id_acc": ided / n, "detect_rate": detected / n, "false_positive": ctrl["detected"]}


def self_as_instrument_gain(backend, compounds, dose):
    """Result B MVP (passive, RUNNABLE): does consulting its own steered variants raise the
    UNMODIFIED model's introspection, beyond the unsteered-self control? Parts 1+2+3."""
    measure = IntrospectionMeasure(backend)
    return SelfAsInstrument(measure, backend, compounds, dose).run_passive()
