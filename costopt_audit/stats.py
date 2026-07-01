"""Significance / equivalence stack: paired hierarchical bootstrap + TOST +
Bonferroni correction.

The bootstrap resamples tasks AND, independently within each resampled task,
its seed replicates ("cluster bootstrap" over both levels). A bootstrap that
only resamples tasks while holding each task's seed replicates fixed never
lets seed-to-seed run variance enter the resampling distribution -- textbook
pseudo-replication, and the exact defect a 2026 review of the CostOpt paper
flagged as its single highest-priority fix. This module ships the fix by
construction: if your runs table has no ``seed`` column, every task collapses
to one pseudo-replicate and this degrades gracefully to a plain task-level
bootstrap; the more seeds you provide, the more trustworthy the CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as _tdist


@dataclass
class PairedResult:
    label_a: str
    label_b: str
    delta_pp: float
    ci_lo: float
    ci_hi: float
    p_value: float
    n_tasks: int
    n_seeds_a: int
    n_seeds_b: int


@dataclass
class TostResult:
    label_a: str
    label_b: str
    delta_pp: float
    ci_lo: float
    ci_hi: float
    p_value: float
    equivalent: bool
    margin_pp: float


def _task_seed_matrix(df: pd.DataFrame, col: str, task_ids: list) -> np.ndarray:
    """Pivot to a (task x seed) matrix of mean correctness, reindexed to a
    fixed, sorted task_id universe so two groups' matrices align row-for-row."""
    grp = df.groupby(["task_id", "seed"])[col].mean().reset_index()
    mat = grp.pivot(index="task_id", columns="seed", values=col).reindex(task_ids)
    return mat.values


def _hier_resample(mat: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Two-stage cluster bootstrap draw: resample tasks (rows) via idx, then
    independently resample each resampled task's seed replicates (columns)."""
    sub = mat[idx]
    n_rows, n_seeds = sub.shape
    if n_seeds <= 1:
        return sub  # nothing to resample at the seed level
    seed_idx = rng.integers(0, n_seeds, size=(n_rows, n_seeds))
    return np.take_along_axis(sub, seed_idx, axis=1)


def paired_bootstrap(
    df: pd.DataFrame,
    by: str,
    label_a: str,
    label_b: str,
    col: str = "fuzzy",
    task_ids: Optional[list] = None,
    n_boot: int = 10_000,
    seed: int = 7,
) -> PairedResult:
    """Paired hierarchical-bootstrap CI + percentile-bootstrap p-value for the
    accuracy difference between two groups of ``df[by]`` (e.g. two models or
    two strategies), on the SAME task universe."""
    a = df[df[by] == label_a]
    b = df[df[by] == label_b]
    if task_ids is None:
        task_ids = sorted(set(a["task_id"]) | set(b["task_id"]))

    a_mat = _task_seed_matrix(a, col, task_ids)
    b_mat = _task_seed_matrix(b, col, task_ids)
    obs = (np.nanmean(a_mat) - np.nanmean(b_mat)) * 100

    rng = np.random.default_rng(seed)
    n_tasks = a_mat.shape[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n_tasks, n_tasks, replace=True)
        a_res = _hier_resample(a_mat, idx, rng)
        b_res = _hier_resample(b_mat, idx, rng)
        diffs[i] = (np.nanmean(a_res) - np.nanmean(b_res)) * 100

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p = float(min(max(p, 1.0 / n_boot), 1.0))

    return PairedResult(
        label_a=label_a, label_b=label_b, delta_pp=float(obs), ci_lo=float(lo), ci_hi=float(hi),
        p_value=p, n_tasks=n_tasks, n_seeds_a=a_mat.shape[1], n_seeds_b=b_mat.shape[1],
    )


def tost(
    df: pd.DataFrame,
    by: str,
    label_a: str,
    label_b: str,
    col: str = "fuzzy",
    delta: float = 0.02,
    alpha: float = 0.05,
    task_ids: Optional[list] = None,
    n_boot: int = 10_000,
    seed: int = 7,
) -> TostResult:
    """Two-one-sided-test equivalence within +/- ``delta`` (default 2pp),
    using the same hierarchical bootstrap as :func:`paired_bootstrap` for the
    standard-error estimate. This is what licenses "these are equivalent"
    rather than merely "we failed to reject a difference."""
    a = df[df[by] == label_a]
    b = df[df[by] == label_b]
    if task_ids is None:
        task_ids = sorted(set(a["task_id"]) | set(b["task_id"]))

    a_mat = _task_seed_matrix(a, col, task_ids)
    b_mat = _task_seed_matrix(b, col, task_ids)
    m = np.nanmean(a_mat) - np.nanmean(b_mat)

    rng = np.random.default_rng(seed)
    n_tasks = a_mat.shape[0]
    d_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n_tasks, n_tasks, replace=True)
        a_res = _hier_resample(a_mat, idx, rng)
        b_res = _hier_resample(b_mat, idx, rng)
        d_boot[i] = np.nanmean(a_res) - np.nanmean(b_res)

    se = np.std(d_boot, ddof=1)
    if se == 0:
        se = 1e-12
    p_lo = 1 - _tdist.cdf((m + delta) / se, n_tasks - 1)
    p_hi = _tdist.cdf((m - delta) / se, n_tasks - 1)
    p = max(p_lo, p_hi)
    tcrit = _tdist.ppf(1 - alpha, n_tasks - 1)
    lo, hi = (m - tcrit * se) * 100, (m + tcrit * se) * 100
    equivalent = lo > -delta * 100 and hi < delta * 100

    return TostResult(
        label_a=label_a, label_b=label_b, delta_pp=float(m * 100), ci_lo=float(lo), ci_hi=float(hi),
        p_value=float(p), equivalent=bool(equivalent), margin_pp=delta * 100,
    )


def bonferroni_alpha(n_comparisons: int, family_alpha: float = 0.05) -> float:
    return family_alpha / max(n_comparisons, 1)
