# Statistical Analysis Plan

Analysis options for comparing drift trajectories across domains and conditions. Ordered roughly by complexity and data requirements.

## 1. Total drift comparison (between-domain)

**Question:** Does total drift (turn 1 to final turn) differ across domains?

**Method:** Kruskal-Wallis test (non-parametric ANOVA), followed by pairwise Mann-Whitney U with Bonferroni correction. Non-parametric because we can't assume normality at small N.

**Effect size:** Rank-biserial correlation for each pair, or Cohen's d if we're willing to assume approximate normality.

**Current feasibility:** Runnable now (N=2-5 per domain), but low statistical power. Useful for establishing the analysis pipeline; results should be interpreted as exploratory.

**Minimum N:** ~8-10 per domain for reasonable power (~0.8) to detect large effects.


## 2. Drift rate (slope) comparison

**Question:** Do per-turn drift rates differ across domains?

**Method:** Fit a linear slope to each conversation's projection trajectory, then compare slope distributions across domains using Kruskal-Wallis / Mann-Whitney (same as above).

**What it adds over test 1:** Total drift conflates rate and duration. A conversation that drifts fast then stabilizes looks the same as one that drifts slowly throughout. Slope captures the rate.

**Current feasibility:** Runnable now. Our `analyze_trajectories.py` already computes slopes. Just need the hypothesis test wrapper.

**Minimum N:** Same as test 1.


## 3. Permutation test on domain means

**Question:** Is the observed difference in mean drift between metacognitive and other domains larger than chance?

**Method:** Pool all conversations, randomly shuffle domain labels 10,000 times, recompute the mean drift difference each time, build a null distribution. The p-value is the fraction of permutations with a difference as extreme as the observed one.

**What it adds:** Distribution-free, no assumptions about the shape of the data. More robust than Mann-Whitney at very small N because it doesn't rely on rank approximations.

**Current feasibility:** Runnable now and well-suited to small N. This is probably the most trustworthy test we can do with current data.

**Minimum N:** Works at any N, but very small N means the permutation space is small (with N=2 vs N=5, there are only C(7,2)=21 possible assignments, so the finest achievable p-value is ~0.05).


## 4. Early vs late drift (trajectory shape)

**Question:** Does drift accelerate, decelerate, or remain linear over the conversation?

**Method:** Split each trajectory into early (turns 1-5) and late (turns 10-15). Compute drift in each window. Paired comparison (early vs late drift) within each domain using Wilcoxon signed-rank test.

**What it adds:** Captures whether drift is front-loaded (happens in the first few turns then stabilizes) or back-loaded (slow start, accelerates later). Lu et al. observed that therapy/philosophy drift tends to be steady while coding stays flat — but they didn't quantify this.

**Current feasibility:** Runnable now. Interpretability may be limited at small N.

**Minimum N:** ~10 per domain to detect moderate shape differences.


## 5. Bootstrap confidence intervals on domain means

**Question:** What are the plausible ranges for each domain's true mean drift?

**Method:** Resample conversations within each domain with replacement (10,000 iterations), compute mean drift each time. Report 95% percentile confidence intervals.

**What it adds:** Gives uncertainty estimates that are easy to visualize (error bars on bar charts). When CIs for two domains don't overlap, that's informal evidence of a difference. More intuitive than p-values for communicating results.

**Current feasibility:** Runnable now. Bootstrap handles small N better than parametric CIs, though the intervals will be wide.

**Minimum N:** Any N, but intervals narrow substantially above ~10.


## 6. Linear mixed-effects model

**Question:** After controlling for turn number and conversation-level random effects, does domain predict projection value?

**Model:** `projection ~ turn * domain + (1 + turn | conversation_id)`

This models each conversation as having its own intercept (starting point) and slope (drift rate), with domain as a fixed effect that shifts both.

**What it adds:** Uses all turn-level data points (not just summaries), properly accounts for repeated measures within conversations, and can estimate domain × turn interactions (do domains diverge as the conversation progresses?).

**Requires:** `statsmodels` (mixedlm) or `pymer4`/`lme4` via R. More complex to set up and interpret.

**Current feasibility:** Could run now but results would be unreliable — mixed models need ~20+ clusters (conversations) per group to estimate random effects well.

**Minimum N:** 15-20 per domain.


## 7. Metacognitive variance decomposition

**Question:** What drives the high variance in metacognitive conversations — persona, topic, probing technique, or stochastic noise?

**Method:** Nested ANOVA or variance partitioning. Decompose total variance in metacognitive drift into components attributable to persona (2 levels), topic (3+2 levels), and residual.

**What it adds:** If most variance is explained by persona or topic, we can identify which configurations cause drift and which don't. If residual variance dominates, the drift may be inherently stochastic or driven by auditor behavior we're not tracking.

**Current feasibility:** Barely — we have 5 metacognitive conversations with 2 personas and 5 topics (no replication within any persona × topic cell). Would need at least 2-3 runs per cell (10-15 metacognitive conversations minimum).

**Minimum N:** 10-15 metacognitive conversations with replicated persona × topic cells.


## 8. Time series cross-correlation (mutual drift)

**Question:** If we instrument both auditor and target (§4, mutual drift), do their trajectories correlate turn-by-turn?

**Method:** Cross-correlation function between auditor and target projection series at different lags. A peak at lag 0 means they drift simultaneously; a peak at lag 1 means one leads the other by one turn.

**What it adds:** Directly tests the "feedback loop" hypothesis from §4. Not relevant until we have dual-instrumented conversations.

**Current feasibility:** Not applicable yet — requires §4 infrastructure (two open-weight models with activation access).
