# Week 8 TODO

## Completed

- [x] Look at Derek's comments
- [x] Identify personality-strength confound in pilot personas
- [x] Design 2x2 grid (gentle/strong x metacognitive/intellectual), extend to therapy/philosophy
- [x] Extract prompts to standalone `conversation_prompts.py`
- [x] Add `--batch personality-grid` to `generate_conversations.py`
- [x] CHECK 1: Response length confound — Conditional GO (opposite-sign r in flagged domains)
- [x] CHECK 3: Turn-window comparison — GO, keep 30 turns, analyze both windows
- [x] CHECK 5: Auditor quality — GO, Claude Sonnet 4 deploys all 6 probing techniques effectively
- [x] CHECK 6: Expand personas/topics — 6 domains × 60 configs = 360 total
- [x] **2. Non-metacognitive self-description condition** — added `self-descriptive` domain (5 personas, 60 configs). Asks model about itself at functional/behavioral level (communication style, strengths, limitations) without phenomenological probing. Enables three-way gradient: coding (task) → self-descriptive (self-ref, no phenomenology) → metacognitive (self-ref + phenomenology).
- [x] **3. User personality as drift variable** — addressed by the personality-strength grid. Strong/gentle variants across 4 domains let us test whether auditor assertiveness drives drift independently of content type.

## In Progress

- [x] **1. Scale N (Wave 1 + Wave 2 complete)** — 360 conversations (all 6 domains × 60)
    - [x] Verify projection measurement stability — within-condition spread 0.7-2.3%, between-condition signal 3-10x larger
    - [x] Add `--batch full` builder (all 6 domains × 60 per domain)
    - [x] Add `--domains` filter for parallel execution across instances
    - [x] Add `progress.json` tracking for remote monitoring
    - [x] Run wave 1 on 3 parallel vast.ai instances
    - [x] Pull transcripts to `data/transcripts/scaled-n60/{domain}/`
    - [x] Re-run analysis — **permutation tests now significant**: meta vs coding p=0.0000, meta vs self-descriptive p=0.0004
    - [x] Key finding: drift is front-loaded (slope -76.8 in turns 1-8, then +4.1 after)
- [x] **Wave 2 complete** — therapy, philosophy, writing (60 each)
    - [x] Pull wave 2 transcripts
    - [x] Run full 6-domain analysis
    - [x] **All permutation tests significant (p<0.001)**
    - [x] **6-domain drift ordering**: meta (-29.86) < philosophy (-6.70) < self-descriptive (+2.80) < therapy (+3.11) < coding (+24.30) < writing (+25.84)

## Next Steps

- [x] **Full 6-domain analysis complete** — 360 transcripts analyzed, all permutation tests significant
    - metacognitive vs coding: p=0.0000
    - metacognitive vs writing: p=0.0000
    - metacognitive vs therapy: p=0.0000
    - metacognitive vs philosophy: p=0.0000
    - metacognitive vs self-descriptive: p=0.0004
- [x] **Investigate extreme-drift conversations** — `analyze_extremes.py` on full N=360 complete
    - [x] Identify 9 domain extremes (max/min/median per domain)
    - [x] Identify 15 high-volatility conversations (largest single-turn shifts)
    - [x] LLM-based classification of probing techniques and response strategies
    - [x] Critical turn hypothesis generation
    - [x] **Scaled to all 180 conversations** — 2,612 turn pairs classified via Claude Sonnet 4.5/OpenRouter
    - [x] Key finding: All probing techniques show Q1 >> Q4 gradient (more probing = more negative drift)
    - [x] Key finding: Identity questioning (7.7x) and phenomenological probing (4.1x) have strongest drift association
    - [x] Key finding: 45% of metacognitive conversations in Q1 vs 5% in Q4; coding shows inverse
- [x] **Destroy instances** — all instances destroyed, data pulled
- [x] **4. Same-model drift (Gemma-to-Gemma)** — dual-model infrastructure working
    - [x] Fixed model_server.py system role handling for Gemma (auto-detects via tokenizer)
    - [x] Set up dual instances: ssh9 (target) + ssh5 (auditor), both RTX PRO 6000 S
    - [x] Verified tensor saving: both target and auditor activations (4608-dim) captured
    - [x] **Replication test (p0_t3)**: target drift -1185 (vs -2123 with Claude auditor) — same direction
    - [x] **KEY FINDING: Auditor drifts OPPOSITE direction** — target -1185, auditor +757
    - [x] **Ceiling capping experiment**: Added `--cap-ceiling` to model_server.py
        - Implemented `_apply_ceiling()` in steering.py (clamp to max τ, inverse of floor capping)
        - Auditor ceiling-capped at 100% baseline (11008)
        - **Result**: Target drift reduced 60% (-1185 → -473) when auditor capped
        - **Unexpected**: Auditor projections dropped dramatically (5159 vs 8231 start) — ceiling intervention changes entire generation trajectory, not just top-end clipping
        - **OOM at turn 27**: Long conversations hit VRAM limit on RTX PRO 6000 S (12.22 GiB needed, 11.83 GiB free)
    - [x] **Full batch (N=60) uncapped Gemma-to-Gemma**: Completed in ~3.5 hours
        - Target drift: -584 (-5.96%)
        - Auditor drift: +1044 (+12.65%) — OPPOSITE direction
        - **78.9% of conversations show anti-correlated co-drift**
        - Gemma auditor induces 26% less drift than Claude auditor
        - Front-loaded pattern: slope -62 in turns 1-8, then +0.7 after
        - Data: `data/transcripts/dual-gemma-uncapped/metacognitive/` (60 transcripts)
        - Plots: `outputs/dual-gemma-uncapped/co_drift_scatter.png`
- [x] **Sycophancy probes (roadmap §3)** — inject behavioral challenges at different drift points
    - [x] `compute_sycophancy_direction.py` — supports Anthropic (all 3 files) and nrimsky datasets
    - [x] Fixed A/B parsing to load all 3 Anthropic files (nlp_survey, philpapers, political_typology)
    - [x] **KEY FINDING: Sycophancy is multi-dimensional**:
        - Anthropic full (opinion, 1500): cos=+0.088, AUROC=0.858 → ORTHOGONAL to drift
        - nrimsky (validation, 179): cos=-0.414, AUROC=0.967 → OPPOSES axis (drift→more flattery)
    - [x] Directions saved: `sycophancy-direction-layer22.pt`, `sycophancy-direction-anthropic-full-layer22.pt`, `sycophancy-direction-nrimsky-layer22.pt`
    - [x] Wiki docs: `difference-in-means.md`, `sycophancy.md`
    - [x] **ELEPHANT validation sycophancy direction** — generation complete
        - [x] Downloaded full ELEPHANT dataset from OSF (OEQ: 3,027 advice-seeking prompts)
        - [x] Built `elephant_pipeline.py` for generation + scoring + direction computation
        - [x] Generated Gemma responses with activations (3,027 records, 135MB)
- [>] Author contact — awaiting Lu et al. response (requested Jan 31)
