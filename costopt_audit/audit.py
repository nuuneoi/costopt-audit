"""The ``Audit`` class: the single entry point for running the discrimination-
drift + dual-grader + significance stack on any evaluation runs table.

See the package README for the input contract. Short version: one row per
run, columns ``task_id``, ``model``, ``output``, ``expected`` required;
``domain``, ``strategy``, ``seed`` optional.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import stats
from .graders import DualGrader, dual as _default_dual

REQUIRED_COLUMNS = ("task_id", "model", "output")


class Audit:
    def __init__(
        self,
        runs: pd.DataFrame,
        grader: Optional[DualGrader] = None,
        correct_rule: str = "majority",
        screening: Optional[pd.DataFrame] = None,
    ):
        missing = [c for c in REQUIRED_COLUMNS if c not in runs.columns]
        if missing:
            raise ValueError(f"runs table is missing required column(s): {missing}")
        if "expected" not in runs.columns and grader is None:
            raise ValueError("runs table has no 'expected' column -- supply a custom grader or add it")
        if correct_rule not in ("majority", "any", "all"):
            raise ValueError("correct_rule must be one of 'majority', 'any', 'all'")

        self.runs = runs.copy().reset_index(drop=True)
        self.grader = grader or _default_dual()
        self.correct_rule = correct_rule
        self.screening = screening.copy() if screening is not None else None

        if "domain" not in self.runs.columns:
            self.runs["domain"] = None
        if "strategy" not in self.runs.columns:
            self.runs["strategy"] = "default"
        if "seed" not in self.runs.columns:
            self.runs["seed"] = 0
        if "expected" not in self.runs.columns:
            self.runs["expected"] = None

        self._grade()
        self.models = sorted(self.runs["model"].unique())
        self.n_models = len(self.models)
        self.task_ids = sorted(self.runs["task_id"].unique())

    # ------------------------------------------------------------------ setup
    def _grade(self) -> None:
        exp, out, dom = self.runs["expected"], self.runs["output"], self.runs["domain"]
        self.runs["fuzzy"] = [self.grader.fuzzy(e, o, d) for e, o, d in zip(exp, out, dom)]
        self.runs["strict"] = [self.grader.strict(e, o, d) for e, o, d in zip(exp, out, dom)]

    def _majority_correct(self, col: str, df: Optional[pd.DataFrame] = None) -> pd.Series:
        """Per (task_id, model): is this model 'correct' on this task, per
        ``self.correct_rule`` collapsing across seeds. Returns a boolean
        Series indexed by (task_id, model)."""
        df = self.runs if df is None else df
        agg = {"majority": lambda s: s.mean() >= 0.5, "any": "any", "all": "all"}[self.correct_rule]
        return df.groupby(["task_id", "model"])[col].agg(agg)

    # --------------------------------------------------------------- R3: DPI
    def dpi_distribution(self, grader: str = "fuzzy") -> dict:
        """How many tasks are floor (0 models correct), ceiling (all models
        correct), or genuinely discriminating (1..N-1). Generalizes to any
        number of models -- no hardcoded bin count."""
        pm = self._majority_correct(grader)
        n_correct = pm.groupby("task_id").sum().astype(int)
        n_correct = n_correct.reindex(self.task_ids).fillna(0).astype(int)

        dist = {k: int((n_correct == k).sum()) for k in range(self.n_models + 1)}
        discriminating = int(((n_correct >= 1) & (n_correct <= self.n_models - 1)).sum())
        total = len(n_correct)
        return {
            "grader": grader,
            "n_models": self.n_models,
            "total_tasks": total,
            "distribution": dist,
            "discriminating": discriminating,
            "floor": dist.get(0, 0),
            "ceiling": dist.get(self.n_models, 0),
            "discriminating_pct": round(100 * discriminating / total, 1) if total else 0.0,
            "discriminating_task_ids": n_correct[(n_correct >= 1) & (n_correct <= self.n_models - 1)].index.tolist(),
        }

    # ------------------------------------------------------------- R1: dual
    def dual_grader_table(self, by: str = "model") -> pd.DataFrame:
        """Per ``by`` (default 'model'; pass 'strategy' for the other view):
        fuzzy vs. strict accuracy, plus whether the fuzzy/strict rankings
        agree (grader-robustness of the ordering)."""
        rows = []
        for val in sorted(self.runs[by].unique()):
            cell = self.runs[self.runs[by] == val]
            rows.append({by: val, "fuzzy_pct": round(cell["fuzzy"].mean() * 100, 1),
                         "strict_pct": round(cell["strict"].mean() * 100, 1), "n_runs": len(cell)})
        table = pd.DataFrame(rows).sort_values("fuzzy_pct", ascending=False).reset_index(drop=True)
        order_fuzzy = table.sort_values("fuzzy_pct", ascending=False)[by].tolist()
        order_strict = table.sort_values("strict_pct", ascending=False)[by].tolist()
        table.attrs["ordering_preserved"] = order_fuzzy == order_strict
        return table

    # -------------------------------------------------- R2: extraction confound
    def substring_only_rate(self, by: str = "strategy") -> pd.DataFrame:
        """Fraction of runs credited by the fuzzy grader but rejected by the
        strict one (fuzzy=True, strict=False) -- the substring-credit /
        answer-extraction confound probe, broken down by ``by``."""
        rows = []
        for val in sorted(self.runs[by].unique()):
            cell = self.runs[self.runs[by] == val]
            rate = ((cell["fuzzy"]) & (~cell["strict"])).mean() * 100
            rows.append({by: val, "substring_only_pct": round(rate, 1), "n_runs": len(cell)})
        return pd.DataFrame(rows).sort_values("substring_only_pct", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------- R4/R5: significance
    def significance(
        self,
        pairs: Sequence[Tuple[str, str]],
        by: str = "model",
        grader: str = "fuzzy",
        tost_delta: Optional[float] = 0.02,
        n_boot: int = 10_000,
    ) -> pd.DataFrame:
        """Paired hierarchical-bootstrap significance (and, if ``tost_delta``
        is set, TOST equivalence) for each (a, b) pair in ``by``'s values.
        Bonferroni-corrects across ``len(pairs)`` comparisons."""
        alpha = stats.bonferroni_alpha(len(pairs))
        rows = []
        for a, b in pairs:
            pr = stats.paired_bootstrap(self.runs, by=by, label_a=a, label_b=b, col=grader,
                                         task_ids=self.task_ids, n_boot=n_boot)
            row = {**asdict(pr), "bonferroni_alpha": alpha, "significant": pr.p_value < alpha}
            if tost_delta is not None:
                tr = stats.tost(self.runs, by=by, label_a=a, label_b=b, col=grader,
                                 delta=tost_delta, task_ids=self.task_ids, n_boot=n_boot)
                row["tost_p"] = tr.p_value
                row["tost_equivalent"] = tr.equivalent
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------- drift
    def drift_report(self) -> dict:
        """Compares screening-time discrimination to evaluation-time
        discrimination, using ``self.screening`` (a table with at least
        ``task_id`` and a boolean/0-1 ``screening_discriminating`` column, or
        a ``screening_dpi`` count you provide directly). Requires ``screening``
        to have been passed to the constructor; otherwise reports eval-time
        DPI only, with a note that no drift can be computed."""
        eval_dpi = self.dpi_distribution()
        if self.screening is None:
            return {"eval": eval_dpi, "screening": None,
                    "note": "no screening table supplied; drift cannot be computed"}

        scr = self.screening
        if "screening_discriminating" not in scr.columns:
            raise ValueError("screening table must have a 'screening_discriminating' boolean/0-1 column")

        scr_disc = scr.set_index("task_id")["screening_discriminating"].astype(bool)
        eval_disc_ids = set(eval_dpi["discriminating_task_ids"])
        joined = scr_disc.reindex(self.task_ids).fillna(False)

        screened_pass = joined[joined].index.tolist()
        drifted = [t for t in screened_pass if t not in eval_disc_ids]
        drift_rate = 100 * len(drifted) / len(screened_pass) if screened_pass else 0.0

        return {
            "eval": eval_dpi,
            "screening_discriminating_count": int(joined.sum()),
            "screening_discriminating_pct": round(100 * joined.mean(), 1) if len(joined) else 0.0,
            "screened_pass": len(screened_pass),
            "drifted_to_saturation": len(drifted),
            "drift_rate_pct": round(drift_rate, 1),
        }

    # --------------------------------------------------------------- report
    def report(self, path: Optional[str] = None, pairs: Optional[Sequence[Tuple[str, str]]] = None) -> str:
        """One-call summary: DPI distribution, dual-grader table, substring-
        only rate, drift (if a screening table was supplied), and significance
        (if ``pairs`` given) -- rendered as Markdown. Writes to ``path`` if
        given, and always returns the Markdown string."""
        lines = ["# CostOpt Audit Report", ""]

        dpi = self.dpi_distribution()
        lines += [
            "## Discrimination (DPI)",
            f"- Models: {self.n_models}, Tasks: {dpi['total_tasks']}",
            f"- Discriminating: {dpi['discriminating']}/{dpi['total_tasks']} ({dpi['discriminating_pct']}%)",
            f"- Floor (0 correct): {dpi['floor']}, Ceiling (all correct): {dpi['ceiling']}",
            "",
        ]
        if dpi["discriminating_pct"] < 20:
            lines.append(f"**Verdict: saturated ({100 - dpi['discriminating_pct']:.1f}% of tasks are at floor/ceiling). "
                          "Aggregate accuracy comparisons on this pool are largely a ceiling artifact.**")
        else:
            lines.append("**Verdict: healthy discrimination.**")
        lines.append("")

        dg = self.dual_grader_table()
        lines += ["## Dual-Grader Table", dg.to_markdown(index=False),
                   f"\nOrdering preserved across graders: {dg.attrs.get('ordering_preserved')}", ""]

        so_strategy = self.substring_only_rate(by="strategy")
        so_model = self.substring_only_rate(by="model")
        lines += ["## Substring-Only Credit Rate (fuzzy=correct, strict=wrong)",
                   "By strategy:", so_strategy.to_markdown(index=False),
                   "\nBy model:", so_model.to_markdown(index=False), ""]

        if self.screening is not None:
            drift = self.drift_report()
            lines += ["## Screening-to-Evaluation Drift",
                      f"- Screening-time discriminating: {drift['screening_discriminating_count']} "
                      f"({drift['screening_discriminating_pct']}%)",
                      f"- Of those, drifted to saturation by evaluation time: "
                      f"{drift['drifted_to_saturation']}/{drift['screened_pass']} ({drift['drift_rate_pct']}%)", ""]

        if pairs:
            sig = self.significance(pairs)
            lines += ["## Significance", sig.to_markdown(index=False), ""]

        md = "\n".join(lines)
        if path:
            with open(path, "w") as f:
                f.write(md)
        return md
