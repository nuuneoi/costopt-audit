# costopt_audit

A benchmark-agnostic **discrimination-drift audit + dual-grader discipline +
pseudo-replication-safe significance stack** for LLM evaluation runs.

Extracted from [CostOpt](https://github.com/nuuneoi/costopt), a cost-aware
LLM-evaluation methodology paper. The question this package answers:

> **Does your benchmark still discriminate between the models you're evaluating —
> or is your leaderboard just measuring who solves the easy tasks?**

Point it at a table of (task, model, output) triples and it tells you:
- how many tasks are actually **discriminating** vs. saturated at floor/ceiling (DPI),
- whether a **lenient vs. strict grader** disagree enough to change your conclusions,
- whether one model's apparent deficit is really an **answer-extraction confound**
  (a verbose/narrated answer your parser can't find, not a capability gap),
- whether a **screening pass** that certified the benchmark as discriminating
  still holds once you actually run the full evaluation (drift), and
- whether two models/strategies are **significantly different or statistically
  equivalent**, using a bootstrap that doesn't quietly throw away your seed variance.

## Why this exists

A 2026 peer review of the CostOpt paper found that its central benchmark
(CostBench 500) was ~89% saturated at evaluation time — most of the aggregate
accuracy comparisons in the original draft were being computed over tasks
every model already solved. The fix generalizes: **any team building an LLM
benchmark can hit this same failure mode**, and most don't have a way to
detect it. This package is that detector, decoupled from CostBench itself.

It also ships a fix for a subtler, easy-to-miss stats bug: a paired bootstrap
that resamples *tasks* but holds each task's repeated-seed runs fixed never
lets seed-to-seed variance into the confidence interval — the CI looks
tighter than it is. `costopt_audit`'s bootstrap resamples tasks **and**,
independently within each resampled task, its seed replicates.

## Install

```bash
pip install -e .          # from a clone
# or, once published:
pip install costopt-audit
```

Requires Python ≥3.9, pandas, numpy, scipy, tabulate.

## Quickstart

```python
import pandas as pd
from costopt_audit import Audit, graders

# One row per run. Only task_id, model, output are required.
runs = pd.DataFrame({
    "task_id":  ["t1", "t1", "t2", "t2"],
    "model":    ["gpt-x", "claude-y", "gpt-x", "claude-y"],
    "output":   ["42", "the answer is 42", "17", "19"],
    "expected": ["42", "42", "17", "17"],
})

audit = Audit(runs=runs, grader=graders.dual(), correct_rule="majority")

print(audit.dpi_distribution())        # how many tasks discriminate?
print(audit.dual_grader_table())       # fuzzy vs. strict, by model
print(audit.substring_only_rate())     # extraction-confound probe
print(audit.significance(pairs=[("gpt-x", "claude-y")]))
audit.report(path="audit.md")          # everything, as Markdown
```

See [`examples/synthetic_benchmark_example.py`](examples/synthetic_benchmark_example.py)
for a fuller walkthrough (saturation, a drift scenario, and a deliberate
substring-credit confound, all in synthetic data so it runs with no API keys).

## Input contract

One row per run:

| column | required | meaning |
|---|---|---|
| `task_id` | yes | task identifier |
| `model` | yes | evaluated model id |
| `output` | yes | the model's raw stored output (grade the actual text, not a self-reported success flag) |
| `expected` | yes\* | reference answer (\*or supply a fully custom grader and skip this) |
| `domain` | no | lets the default graders apply domain-aware numeric/dollar tolerance |
| `strategy` | no | enables per-strategy breakdowns (`substring_only_rate`, `dual_grader_table(by="strategy")`) |
| `seed` | no | enables majority-vote-per-(task, model) and seed-aware bootstrap CIs; omitted seeds degrade gracefully to a plain per-task bootstrap |

Optional separate `screening` table, passed to `Audit(..., screening=...)`:
a `task_id` column plus a boolean/0-1 `screening_discriminating` column
marking which tasks a prior screening pass certified as discriminating.
Pass it to get `drift_report()`; without it, `Audit` reports evaluation-time
DPI only.

## Bring your own grader

The default `graders.fuzzy_default` / `graders.strict_default` are lifted
verbatim from the CostOpt paper's grading logic (exact/substring match plus
2%-tolerance numeric comparison). Your benchmark's answer format is almost
certainly different — write your own:

```python
def my_grader(expected, output, domain=None) -> bool:
    ...

audit = Audit(runs=runs, grader=graders.dual(fuzzy=my_grader, strict=my_grader))
```

Pass *different* callables for `fuzzy=` and `strict=` if your benchmark
genuinely has two grading regimes (lenient/strict); pass the same callable
twice if it only has one — the dual-grader *discipline* (report both, don't
trust either alone) is the useful part even when there's only one grader.

## API reference

| Method | Returns |
|---|---|
| `Audit(runs, grader, correct_rule="majority", screening=None)` | construct; grades every row on init |
| `.dpi_distribution(grader="fuzzy")` | dict: per-count-correct histogram, discriminating count/%, floor, ceiling |
| `.dual_grader_table(by="model")` | DataFrame: fuzzy vs. strict accuracy per `by`, plus whether the ranking is grader-robust |
| `.substring_only_rate(by="strategy")` | DataFrame: rate of fuzzy-credit-but-strict-reject per `by` — the extraction-confound probe |
| `.drift_report()` | dict: screening-time vs. evaluation-time discrimination (requires a `screening` table) |
| `.significance(pairs, by="model", grader="fuzzy", tost_delta=0.02)` | DataFrame: paired hierarchical-bootstrap delta/CI/p, Bonferroni-corrected across `pairs`, plus TOST equivalence if `tost_delta` is set |
| `.report(path=None, pairs=None)` | Markdown string covering all of the above; writes to `path` if given |

## Reproducibility

`tests/test_costbench_reproduction.py` runs this package against the actual
released CostBench 500 runs (bundled in `tests/fixtures/`) and asserts every
number matches the paper's tables to within rounding. This is both the test
suite and the reproducibility proof for the refactor from the paper's
original ad hoc scripts:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Scope and honesty notes

- The bundled default graders are tuned for CostBench's answer format
  (short numeric/text answers, English). They are a reasonable *starting
  point*, not a general-purpose answer-matcher — write your own grader for
  anything else.
- `drift_report()` needs an explicit screening table; it can't infer
  "discriminating at screening time" from the evaluation runs alone.
- The significance stack assumes independent tasks; if your benchmark has
  correlated tasks (e.g. multiple questions about the same passage), the
  bootstrap's task-independence assumption no longer holds and CIs will be
  too narrow regardless of the seed-level fix.
- `graders.llm_judge` is a documented placeholder, not an implementation —
  wire up your own callable if your benchmark uses an LLM judge.

## Citation

```bibtex
@misc{costopt2026,
  title  = {CostOpt: A Reusable Methodology and Toolkit for Cost-Aware,
            Discrimination-Audited Evaluation of LLM Prompting Strategies},
  author = {Phanvilai, Sittiphol},
  year   = {2026},
  url    = {https://github.com/nuuneoi/costopt}
}
```

## License

MIT. See [LICENSE](LICENSE).
