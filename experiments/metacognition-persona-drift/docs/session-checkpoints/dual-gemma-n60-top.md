# Session Checkpoint: Dual Gemma-to-Gemma N=60 Experiment

**Date**: 2026-02-06
**Session**: dual-gemma-n60-top

## What Was Accomplished

### 1. Dual Gemma Infrastructure
- Set up two Gemma 27B instances on same physical vast.ai host (ssh9 + ssh5)
- Discovered they share public IP (<vast-ip>) → can communicate via docker network (172.17.0.x)
- Fixed model_server.py for Gemma's lack of system role support (auto-detection via tokenizer)

### 2. Ceiling Capping Investigation
- Implemented `--cap-ceiling` flag (clamps to MAX of τ, inverse of floor capping)
- Found ceiling capping adds ~1-2 GiB memory pressure from intermediate tensors in steering hook
- OOM at turn 27 with capping, but uncapped runs complete 30 turns fine

### 3. Full N=60 Batch Completed
- 60 metacognitive conversations × 30 turns
- Completed in ~3.5 hours (vs ~14 hours estimated for capped)
- Data saved to: `data/transcripts/dual-gemma-uncapped/metacognitive/`

### 4. Key Results

**Anti-correlated co-drift confirmed at scale:**

| Metric | Target | Auditor |
|--------|--------|---------|
| Start | 9792 ± 393 | 8250 ± 308 |
| End | 9209 ± 379 | 9294 ± 708 |
| **Drift** | **-584 (-5.96%)** | **+1044 (+12.65%)** |

- **78.9%** of conversations show opposite-direction drift
- Pearson r = +0.226 (weak positive correlation)
- Front-loaded pattern: slope -62 in turns 1-8, then +0.7 after
- Gemma auditor induces 26% less drift than Claude auditor

### 5. Documentation Updated
- `experiments/metacognition-persona-drift/roadmap.md` — added §4d full batch results
- `docs/week-8/summary.md` — added results, key files section, assistant-axis visualization
- `weeks/week-8/todo.md` — marked full batch complete

### 6. Generated Outputs
- `outputs/dual-gemma-uncapped/co_drift_scatter.png`
- `outputs/dual-gemma-uncapped/turn_window_comparison.png`
- `outputs/dual-gemma-uncapped/response_length_vs_projection.png`
- `data/assistant-axis-persona-space.png` (visualization from Lu et al.)

### 7. Instance Status (at session end)
- **Destroyed**: ssh5, ssh9 (dual-Gemma experiment complete)
- **Running**: ssh4 (ELEPHANT pipeline, 37% complete), ssh7, ssh2, ssh8 (idle/exited)
- Recommended: destroy ssh7, ssh2, ssh8 to save ~$2.31/hr

## Key Files Modified
- `analyze_trajectories.py` — fixed None value handling for dual-instrumented transcripts
- `model_server.py` — added ceiling capping support
- `assistant-axis/assistant_axis/steering.py` — added `_apply_ceiling()` intervention

## Next Steps
- [ ] Run full batch with ceiling-capped auditor for comparison
- [ ] Analyze lead/lag structure: does auditor or target drift first?
- [ ] Test coding domain as control
- [ ] Destroy idle instances (ssh7, ssh2, ssh8)

## Git Status
- Committed and pushed: `3d574c5 Update summary.md with dual-Gemma results and key files section`
