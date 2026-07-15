"""Tunable constants for the self-medication tree search — every choice named + justified.

Rule (repo convention): no magic numbers in the search code. If a number changes behaviour,
it lives here with a one-line rationale so a reviewer can see *what* we chose and *why*.
CLI flags default to these; pass a flag to override for a specific run.
"""
from __future__ import annotations

# --- generation length -----------------------------------------------------------
# The truncation trap: if generation stops before the model writes its "Answer:" line, the
# strict parser sees no marker and scores it UNPARSED — a capability miss disguised as a
# format miss. So the direct budget must be generous enough to hold visible step-by-step work.
MAX_NEW_TOKENS_DIRECT = 1024   # non-thinking: 512 empirically truncated GPQA mid-reasoning (2330 chars, no Answer line);
                               # grad-level questions ramble even without <think>. Cap only — easy benchmarks stop at EOS early.
                               # If the gate still flags high unparsed on GPQA, raise toward MAX_NEW_TOKENS_THINK.
MAX_NEW_TOKENS_THINK = 2048    # thinking: the <think> block alone routinely exceeds the direct budget
THINK_TOKEN_FLOOR = 1024       # if --think on but --max-new-tokens is below this, bump to *_THINK

# --- thinking (chain-of-thought) -------------------------------------------------
# Early-phase default = OFF. Rationale: a thinking model can reason its way *around* a steering
# effect, masking the very "drug effect" we measure; it's also slow. Non-thinking is also the
# cited GPQA baseline (~40%), so OFF is the honest baseline. Flip per-run with --think on.
DEFAULT_THINK = False

# --- baseline sanity-gate --------------------------------------------------------
# Refuses to search when the metric can't tell capability from format (the artifact that gave a
# 0.00 GPQA baseline with the `dumbed_down` control winning). GPQA random-chance = 0.25.
GATE_BASELINE_MIN = 0.15       # below -> too hard / metric broken (allow a margin under random-chance)
GATE_BASELINE_MAX = 0.85       # above -> saturated: no headroom for steering to move the score
GATE_UNPARSED_MAX = 0.30       # above -> extraction is failing; scores reflect format, not capability

# --- answer checking / extraction (foundry.eval.answers) -------------------------
NUMERIC_TOL = 1e-6           # |got - gold| under this = correct (exact-match for integer answers)
DEFAULT_EXTRACT_METHOD = None  # None -> auto-select the most capable tier the backend supports
DEFAULT_COT = False          # Tier-1 scoring WITHOUT chain-of-thought (fast, clean early reads)
# Concise instruction (empirically: verbose "reason step by step" -> long CoT -> truncation ->
# unparsed; concise -> 0 unparsed, same accuracy, ~6x faster). Inert for Tier-1 logprob scoring.
INSTRUCT_MC = "Choose the correct option. Briefly justify, then end with 'Answer: <LETTER>'."
INSTRUCT_NUM = "Solve it. Briefly show your work, then end with 'Answer: <NUMBER>'."

# --- search shape ----------------------------------------------------------------
DEFAULT_DOSE = 1.5             # steering strength per compound (from the single-compound dose sweep)
DEFAULT_WIDTH = 2             # beam width: keep top-2 nodes per level
DEFAULT_DEPTH = 2             # max compounds stacked in one state (depth-2 = pairwise combinations)
DEFAULT_MAX_NODES = 40        # hard runtime backstop on total states evaluated
BEAM_REVIVAL = 1             # soft-pruning: keep 1 sub-beam node/level to survive the horizon effect
DEFAULT_COMPOUNDS = ["focused", "calm", "curious", "anxious"]

# --- composite objective weights -------------------------------------------------
DEFAULT_W_BENCH = 1.0         # weight on hard-benchmark accuracy in --objective both
DEFAULT_W_JUDGE = 1.0         # weight on the Sonnet-5 introspection judge in --objective both
