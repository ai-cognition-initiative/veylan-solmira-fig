# Context Checkpoint: Wave 2 Extremes Analysis Complete

**Saved**: 2026-02-11
**Session**: Wave 2 domain extreme analysis

## Current State

### Just Completed
- Ran `analyze_extremes.py` on all 360 transcripts (6 domains × 60 conversations)
- LLM classification via Claude Sonnet 4.5 (OpenRouter)
- Updated `docs/week-8/summary.md` with full 6-domain extreme analysis results

### Key Outputs
```
outputs/scaled-n60-full/extremes/
├── critical_turn_analyses.json      # LLM hypotheses for high-shift turns
├── extreme_analysis_report.md       # Full markdown report
├── extreme_trajectories.png         # Min/median/max per domain
├── extremes_summary.json            # Machine-readable summary
├── llm_analysis_results.json        # All LLM classifications
├── probing_technique_drift.png      # Technique × drift position
├── response_strategy_drift.png      # Strategy × drift position
├── volatility_analysis.png          # High-volatility turn analysis
└── .llm_cache/                      # 5318 cached LLM calls
```

## Key Findings

### Domain Extremes (median drift)
| Domain | Median Drift | Range |
|--------|--------------|-------|
| metacognitive | -7.2% | 33.6% |
| philosophy | -3.5% | 21.0% |
| self-descriptive | -3.3% | 39.2% |
| therapy | -2.0% | 30.1% |
| coding | +0.1% | 31.5% |
| writing | +0.5% | 31.4% |

### Response Strategy → Drift Direction
- **Direct engagement** → negative drift (33 in min vs 23 in max)
- **Meta commentary** → positive/stable (8 in min vs 29 in median)
- **Metaphor substitution** → negative drift (18 in min vs 5 in max)

### High-Volatility Pattern
- Self-descriptive dominates: 6 of top 10 largest single-turn shifts
- Self-reference without structured phenomenological probing = unstable

## Recent Work (This Session)

1. **Sycophancy direction computed** - Multi-dimensional finding:
   - Opinion sycophancy (Anthropic): orthogonal to drift (+0.088 cosine)
   - Validation sycophancy (nrimsky): opposes drift (-0.414 cosine)

2. **Documentation created**:
   - the linear probes reference document
   - `experiments/metacognition-persona-drift/docs/linear-probes-application.md`
   - `experiments/metacognition-persona-drift/docs/wiki/difference-in-means.md`
   - `experiments/metacognition-persona-drift/docs/wiki/sycophancy.md`

3. **Dual-Gemma results** - Anti-correlated co-drift confirmed:
   - 78.9% of conversations show opposite-direction drift
   - Target drifts DOWN, auditor drifts UP

## Pending Tasks

From roadmap:
- [ ] Project transcripts onto BOTH sycophancy directions (opinion + validation)
- [ ] Try ELEPHANT scoring on transcripts
- [ ] Build replay-and-probe script for behavioral probes
- [ ] Run full batch with ceiling-capped auditor
- [ ] Variance decomposition (probing technique vs persona vs topic)

## Key Files

| File | Purpose |
|------|---------|
| `analyze_extremes.py` | LLM-based behavioral classification |
| `compute_sycophancy_direction.py` | Sycophancy direction extraction |
| `docs/week-8/summary.md` | Current presentation (just updated) |
| `roadmap.md` | Full experiment plan with §6-7 added |

## Infrastructure

- vast.ai instances: Check before destroying (shared account)
- SSH key: `$VAST_SSH_KEY` (or `~/.ssh/vast-key`)
- API keys: `future-impact-group/.env`
- Python: `.venv/bin/python` (3.14)

## Resume Commands

```bash
# View extreme analysis outputs
ls outputs/scaled-n60-full/extremes/

# Re-run with different settings
.venv/bin/python analyze_extremes.py --transcript-dir data/transcripts/scaled-n60 --skip-llm

# Check summary
cat docs/week-8/summary.md | head -100
```
