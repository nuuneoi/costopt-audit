# CostOpt Audit Report

## Discrimination (DPI)
- Models: 3, Tasks: 30
- Discriminating: 13/30 (43.3%)
- Floor (0 correct): 8, Ceiling (all correct): 9

**Verdict: healthy discrimination.**

## Dual-Grader Table
| model   |   fuzzy_pct |   strict_pct |   n_runs |
|:--------|------------:|-------------:|---------:|
| model-a |        68.9 |         68.9 |       90 |
| model-b |        55.6 |         55.6 |       90 |
| model-c |        36.7 |         17.8 |       90 |

Ordering preserved across graders: True

## Substring-Only Credit Rate (fuzzy=correct, strict=wrong)
By strategy:
| strategy   |   substring_only_pct |   n_runs |
|:-----------|---------------------:|---------:|
| zero-shot  |                  6.3 |      270 |

By model:
| model   |   substring_only_pct |   n_runs |
|:--------|---------------------:|---------:|
| model-c |                 18.9 |       90 |
| model-a |                  0   |       90 |
| model-b |                  0   |       90 |

## Screening-to-Evaluation Drift
- Screening-time discriminating: 8 (26.7%)
- Of those, drifted to saturation by evaluation time: 4/8 (50.0%)

## Significance
| label_a   | label_b   |   delta_pp |   ci_lo |   ci_hi |   p_value |   n_tasks |   n_seeds_a |   n_seeds_b |   bonferroni_alpha | significant   |   tost_p | tost_equivalent   |
|:----------|:----------|-----------:|--------:|--------:|----------:|----------:|------------:|------------:|-------------------:|:--------------|---------:|:------------------|
| model-a   | model-b   |    13.3333 |       0 | 27.7778 |    0.0662 |        30 |           3 |           3 |               0.05 | False         | 0.935702 | False             |
