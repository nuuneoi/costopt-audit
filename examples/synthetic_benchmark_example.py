#!/usr/bin/env python3
"""Portability demo: run costopt_audit on a benchmark that has nothing to do
with CostBench.

This is SYNTHETIC data constructed to exercise every feature (some tasks
saturated, some at floor, some genuinely discriminating; a drift scenario;
a deliberate substring-credit confound) -- it is not a claim about any real
model's performance. The point is that nothing here is CostBench-specific:
swap in your own `runs` DataFrame with the same five columns and everything
below runs unchanged. See the README for the exact input contract.

Run:  python3 examples/synthetic_benchmark_example.py
"""
import random

import pandas as pd

from costopt_audit import Audit, graders


def build_synthetic_runs(n_tasks: int = 30, seeds: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    models = ["model-a", "model-b", "model-c"]
    rows = []
    for i in range(n_tasks):
        task_id = f"task-{i:03d}"
        # 4+ digit answers so the generic substring rule (len(expected) > 3,
        # present only in the fuzzy grader) is what catches a narrated answer
        # below -- not domain-aware numeric tolerance, which both graders have
        # and would credit either way, masking the confound.
        expected = str(rng.randint(1000, 9999))
        # First 6 tasks: every model gets it right (ceiling/saturated).
        # Next 6: every model gets it wrong (floor).
        # Remaining: genuinely discriminating -- each model has its own skill level.
        if i < 6:
            p_correct = {"model-a": 1.0, "model-b": 1.0, "model-c": 1.0}
        elif i < 12:
            p_correct = {"model-a": 0.0, "model-b": 0.0, "model-c": 0.0}
        else:
            p_correct = {"model-a": 0.85, "model-b": 0.55, "model-c": 0.30}

        for model in models:
            for s in range(seeds):
                correct = rng.random() < p_correct[model]
                if correct:
                    # model-c narrates its reasoning instead of returning a bare
                    # value on ~half its correct answers -- a substring-credit
                    # confound just like the paper's Claude tool-calling case.
                    if model == "model-c" and rng.random() < 0.5:
                        output = f"Let me think step by step... the final answer is {expected}."
                    else:
                        output = expected
                else:
                    output = str(rng.randint(1000, 9999))
                rows.append({
                    "task_id": task_id, "model": model, "output": output, "expected": expected,
                    "domain": "general", "strategy": "zero-shot", "seed": s,
                })
    return pd.DataFrame(rows)


def build_synthetic_screening(runs: pd.DataFrame) -> pd.DataFrame:
    """Pretend these tasks were screened for discrimination using a single
    architecture before the full evaluation above ran with three. A few of
    the tasks that looked discriminating at screening time saturate once all
    three models are actually run -- the screening-to-evaluation drift the
    audit is designed to catch."""
    task_ids = sorted(runs["task_id"].unique())
    screening_discriminating = {t: True for t in task_ids[12:20]}  # 8 "passed screening"
    return pd.DataFrame([
        {"task_id": t, "screening_discriminating": screening_discriminating.get(t, False)}
        for t in task_ids
    ])


def main():
    runs = build_synthetic_runs()
    screening = build_synthetic_screening(runs)

    audit = Audit(runs=runs, grader=graders.dual(), correct_rule="majority", screening=screening)

    print("=== DPI distribution (fuzzy) ===")
    print(audit.dpi_distribution())

    print("\n=== Dual-grader table ===")
    print(audit.dual_grader_table())

    print("\n=== Substring-only credit rate (the confound probe) ===")
    print(audit.substring_only_rate(by="model"))

    print("\n=== Screening-to-evaluation drift ===")
    print(audit.drift_report())

    print("\n=== Significance: model-a vs model-b vs model-c ===")
    print(audit.significance(pairs=[("model-a", "model-b"), ("model-b", "model-c"), ("model-a", "model-c")],
                              by="model", tost_delta=None))

    print("\n=== Full report -> audit_report.md ===")
    audit.report(path="audit_report.md", pairs=[("model-a", "model-b")])
    print("wrote audit_report.md")


if __name__ == "__main__":
    main()
