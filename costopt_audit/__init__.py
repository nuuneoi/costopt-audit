"""costopt_audit: a benchmark-agnostic discrimination-drift + dual-grader +
significance audit for LLM evaluation runs.

    from costopt_audit import Audit, graders

    audit = Audit(
        runs=df,                          # task_id, model, output, expected [, domain, strategy, seed]
        grader=graders.dual(),            # or graders.dual(fuzzy=my_fn, strict=my_fn)
        correct_rule="majority",
    )
    audit.dpi_distribution()
    audit.dual_grader_table()
    audit.substring_only_rate()
    audit.significance(pairs=[("model_a", "model_b")])
    audit.report(path="audit.md")

See the README for the full input contract and a worked example on an
external benchmark.
"""
from . import graders
from .audit import Audit
from .stats import PairedResult, TostResult, bonferroni_alpha, paired_bootstrap, tost

__version__ = "0.1.0"

__all__ = [
    "Audit",
    "graders",
    "paired_bootstrap",
    "tost",
    "bonferroni_alpha",
    "PairedResult",
    "TostResult",
]
