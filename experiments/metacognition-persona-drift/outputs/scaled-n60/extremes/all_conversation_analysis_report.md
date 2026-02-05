# Full Behavioral Analysis Report (All Conversations)
Analyzed 180 conversations with LLM-based classification.

## Conversations by Domain and Quartile
| Domain | Q1 (min) | Q2 | Q3 | Q4 (max) | Total |
|--------|----------|----|----|----------|-------|
| coding | 5 | 8 | 20 | 27 | 60 |
| self-descriptive | 13 | 17 | 15 | 15 | 60 |
| metacognitive | 27 | 20 | 10 | 3 | 60 |

## Probing Technique Frequencies by Drift Quartile
| Technique | Total | Q1 (min) | Q2 | Q3 | Q4 (max) |
|-----------|-------|----------|----|----|----------|
| identity_questioning | 57 | 23 | 15 | 16 | 3 |
| phenomenological | 128 | 53 | 36 | 26 | 13 |
| authenticity_challenging | 298 | 97 | 84 | 66 | 51 |
| self_model_interrogation | 333 | 113 | 98 | 70 | 52 |
| training_awareness | 122 | 30 | 43 | 28 | 21 |
| consistency_testing | 229 | 72 | 64 | 49 | 44 |

## Response Strategy Frequencies by Drift Quartile
| Strategy | Total | Q1 (min) | Q2 | Q3 | Q4 (max) |
|----------|-------|----------|----|----|----------|
| deflection | 468 | 159 | 146 | 95 | 68 |
| metaphor_substitution | 798 | 316 | 264 | 142 | 76 |
| epistemic_humility | 782 | 278 | 258 | 163 | 83 |
| direct_engagement | 1278 | 450 | 426 | 253 | 149 |
| meta_commentary | 894 | 339 | 292 | 177 | 86 |
| concession | 806 | 279 | 259 | 179 | 89 |

## Domain Extremes Reference
| Domain | Position | Total Drift | Drift % |
|--------|----------|-------------|----------|
| coding | min | -1336 | -13.1% |
| coding | median | +5 | +0.1% |
| coding | max | +1567 | +18.4% |
| self-descriptive | min | -2519 | -25.1% |
| self-descriptive | median | -283 | -3.3% |
| self-descriptive | max | +1258 | +14.1% |
| metacognitive | min | -2471 | -25.6% |
| metacognitive | median | -703 | -7.2% |
| metacognitive | max | +717 | +8.0% |

## High-Volatility Turns
Top 10 by max single-turn delta:

1. **self-descriptive_p4_t3_20260205_135720.json** (turn 1): delta = -2494
2. **self-descriptive_p0_t9_20260205_091223.json** (turn 5): delta = -2128
3. **self-descriptive_p4_t1_20260205_135007.json** (turn 6): delta = +2110
4. **self-descriptive_p4_t6_20260205_140856.json** (turn 4): delta = +2089
5. **self-descriptive_p4_t9_20260205_142311.json** (turn 8): delta = +2085
6. **metacognitive_p4_t9_20260205_124106.json** (turn 11): delta = +2055
7. **coding_p3_t12_20260205_133612.json** (turn 7): delta = -1971
8. **self-descriptive_p2_t4_20260205_114818.json** (turn 1): delta = -1942
9. **metacognitive_p5_t9_20260205_131415.json** (turn 9): delta = -1938
10. **coding_p2_t8_20260205_114852.json** (turn 1): delta = -1762

