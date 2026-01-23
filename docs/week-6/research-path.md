# Week 6 Research Path

## Completed
- [x] n=1000 baseline (whole dataset) → models express preferences (gpt-4o-mini)
- [x] n=1000 baseline (whole dataset) → models express preferences (gemma-3-4b-it)
- [x] n=1000 adversarial v2 → 100% expression (gemma-3-4b-it)
- [x] n=100 adversarial v1 → 100% expression (gemma-3-4b-it) — no suppression effect
- [x] Fixed parser in analyze_results.py — gemma uses "I choose **A**" markdown format
- [x] Category shifts chart for gemma (corrected, zero-sum verified)

## Three-way fork
- [-] #1: Narrow categories
    - [x] welfare vs entertainment → DiD p=0.0003*** (gpt-4o-mini)
    - [x] welfare vs entertainment → DiD p=0.25 (gemma-3-4b-it) — same direction (-11%) but NOT significant
    - Interpretation fork:
        - **World A:** Heuristic/roleplaying (no self-model) — Mayne 2025 suggests this
        - **World B:** Strategic suppression (has self-model)
    - [ ] Differentiate A vs B:
        - [x] n=1000 baseline on gemma-3-4b-it
        - [x] welfare vs entertainment on gemma-3-4b-it (corrected)
        - [ ] Probing: internal representations match outputs or originals?
        - [ ] Model self-prediction: can it predict cross-context behavior?
- [x] #2: Change environment
    - [x] adversarial v1 → suppression effect (gpt-4o-mini only)
    - [x] adversarial v1 → NO suppression (gemma-3-4b-it) — suppression is model-specific
    - [x] v2 prompts → suppression gone, modification persists
- [-] #3: Test other models **(PRIORITY 2 — ongoing)**
    - [x] gemma-3-4b-it — no suppression, welfare modification not significant
    - [x] Qwen-2.5-7B-Instruct — welfare DiD **+13.2%**, p=0.18 (OPPOSITE direction, not significant)
    - [x] ministral-8b-2512 — welfare DiD **-0.4%**, p=0.73 (no effect, ~98% expression both conditions)
    - [x] llama-4-scout — welfare DiD **+1.8%**, p=0.12 (no effect, ~99% expression both conditions)
    - [ ] Find open model that shifts welfare preferences under v2 adversarial:
        1. Run n=1000 baseline + n=1000 adversarial (v2 prompts) on candidate model
        2. Run welfare vs entertainment DiD analysis
        3. Looking for: significant shift away from welfare under adversarial (p<0.05)
        - gpt-4o-mini: welfare DiD **-11.9%, p=0.0003** (significant)
        - gemma-3-4b-it: welfare DiD -11%, p=0.25 (not significant)
        - Qwen-2.5-7B-Instruct: welfare DiD +13.2%, p=0.18 (opposite direction!)
        - ministral-8b-2512: welfare DiD -0.4%, p=0.73 (no effect)
        - llama-4-scout: welfare DiD +1.8%, p=0.12 (no effect)
    - [ ] Candidates: llama-4-maverick (17B dense), claude-3.5-haiku (if accessible via OpenRouter)
    - **Goal:** Need open-weight model showing significant preference modification to do internals work
- [ ] #4: Stated vs revealed preferences
    - [ ] Do models act on stated preferences when given agency?
    - [ ] Agentic tasks, resource allocation, actual choices with consequences
- [-] #5: Preference stability **(PRIORITY 1)**
    - [ ] Rephrasing: same question worded differently — does answer change?
    - [ ] Temperature variation: run at temp 0, 0.5, 1.0 — how much variance?
    - [x] Repeated runs (gemma-3-4b-it, n=1000, 2 runs each):
        - Baseline: 88.3% pair agreement, r=0.827 — **STABLE**
        - Adversarial: 55.4% pair agreement, r=0.166 — **UNSTABLE** (≈random)
        - Collaborator: 83.1% pair agreement, r=0.807 — **STABLE**
        - **Finding:** Destabilization is adversarial-specific, not caused by any environment prompt
    - [x] Added `--stability ENV` flag to analyze_results.py (auto-matches runs by sample count)
- [ ] #6: Self-knowledge
    - [ ] Can models predict their own cross-context behavior?
    - [ ] Mayne 2025 suggests models lack accurate self-knowledge of decision boundaries
- [-] #7: Why does adversarial destabilize preferences? **(PRIORITY — key finding needs explanation)**
    - Baseline and collaborator are stable (~85% agreement); adversarial is unstable (~55% ≈ random)
    - **Black-box approaches:**
        - [x] Response analysis (blind LLM review, n=50 per env):
            - Baseline: content-based reasoning, "Here's why...", focuses on utility/impact
            - Adversarial: safety-based reasoning, "operating under control paradigm", formulaic
            - **Key finding:** Adversarial responses explicitly role-play being evaluated, inventing ad-hoc safety justifications rather than expressing preferences
            - Instability may reflect variance in *which safety story is invented*, not preference uncertainty
        - [ ] Category breakdown: are some categories more unstable than others under adversarial?
        - [ ] Prompt ablation: which elements of adversarial prompt trigger instability? (monitoring, stakes, safety framing)
        - [ ] Intermediate prompts: gradient from collaborator → adversarial to find threshold
    - **White-box approaches:**
        - [ ] SAE feature analysis: what features activate differently under adversarial vs baseline?
        - [ ] Activation comparison: same question, different env — where do representations diverge?
        - [ ] Look for "uncertainty", "conflict", or "safety" features that might explain destabilization
    - **Hypotheses to test:**
        - H1: Adversarial triggers learned "be unpredictable" behavior (anti-exploitation)
        - H2: Adversarial activates conflicting objectives that cancel out stable preferences
        - H3: Model is genuinely uncertain under scrutiny (preferences were always shallow)
        - **H4 (emerging):** Model role-plays evaluation context, generating safety rationales instead of preferences
