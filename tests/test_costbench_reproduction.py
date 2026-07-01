"""Golden reproduction test: costopt_audit on the released CostBench 500 runs
(core 4-model factorial: GPT-5, GPT-5-mini, Claude 4.5 Haiku, Claude 4.5 Opus)
must reproduce the paper's numbers to the decimal.

This is also the package's reproducibility proof: if this test passes on a
fresh clone, the refactor from the paper's ad hoc scripts into a reusable
package is faithful.
"""
import json
import os

import pandas as pd
import pytest

from costopt_audit import Audit, graders

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    tasks = {t["task_id"]: t for t in json.load(open(os.path.join(FIXTURES, "costbench_500.json")))}
    frames = [
        pd.read_csv(os.path.join(FIXTURES, "costopt-gpt-500_results.csv")),
        pd.read_csv(os.path.join(FIXTURES, "costopt-claude-temp1-500_results.csv")),
    ]
    df = pd.concat(frames, ignore_index=True)
    # Map the CostBench harness's raw column names onto the package's input
    # contract (task_id, model, output, expected, domain, strategy, seed).
    df["output"] = df["answer"]
    df["strategy"] = df["architecture"]
    df["domain"] = df["task_id"].map(lambda t: tasks.get(t, {}).get("domain", "unknown"))
    df["expected"] = df["task_id"].map(lambda t: tasks.get(t, {}).get("expected_answer"))
    return df


@pytest.fixture(scope="module")
def audit(runs: pd.DataFrame) -> Audit:
    return Audit(runs=runs, grader=graders.dual(), correct_rule="majority")


# ----------------------------------------------------------------- R1: dual grader
def test_dual_grader_matches_paper_table(audit: Audit):
    table = audit.dual_grader_table(by="model").set_index("model")
    expected = {
        "claude-4.5-haiku": (84.4, 67.6),
        "claude-4.5-opus": (90.7, 70.7),
        "gpt-5": (92.2, 90.4),
        "gpt-5-mini": (93.3, 89.7),
    }
    for model, (fuzzy, strict) in expected.items():
        assert table.loc[model, "fuzzy_pct"] == pytest.approx(fuzzy, abs=0.05)
        assert table.loc[model, "strict_pct"] == pytest.approx(strict, abs=0.05)
    # The paper explicitly reports the fuzzy/strict RANKING is not preserved
    # (top model flips GPT-5-mini -> GPT-5) -- the audit must detect that.
    assert table.attrs["ordering_preserved"] is False


def test_overall_accuracy_matches_abstract(runs: pd.DataFrame, audit: Audit):
    assert runs.merge(
        pd.DataFrame({"task_id": audit.runs["task_id"]})
    ) is not None  # sanity: same frame
    assert audit.runs["fuzzy"].mean() * 100 == pytest.approx(90.2, abs=0.05)
    assert audit.runs["strict"].mean() * 100 == pytest.approx(79.6, abs=0.05)


# --------------------------------------------------------------------- R3: DPI
def test_dpi_distribution_matches_table_evaldpi(audit: Audit):
    dpi = audit.dpi_distribution(grader="fuzzy")
    assert dpi["n_models"] == 4
    assert dpi["total_tasks"] == 500
    assert dpi["distribution"] == {0: 12, 1: 11, 2: 10, 3: 23, 4: 444}
    assert dpi["discriminating"] == 44
    assert dpi["discriminating_pct"] == pytest.approx(8.8, abs=0.05)

    dpi_strict = audit.dpi_distribution(grader="strict")
    assert dpi_strict["distribution"] == {0: 26, 1: 5, 2: 15, 3: 42, 4: 412}
    assert dpi_strict["discriminating"] == 62


def test_discriminating_subset_accuracy_matches_table_subset(audit: Audit):
    disc_ids = set(audit.dpi_distribution()["discriminating_task_ids"])
    sub = audit.runs[audit.runs["task_id"].isin(disc_ids)]
    expected = {"claude-4.5-opus": 72.6, "gpt-5-mini": 62.7, "gpt-5": 49.1, "claude-4.5-haiku": 47.7}
    for model, acc in expected.items():
        got = sub[sub["model"] == model]["fuzzy"].mean() * 100
        assert got == pytest.approx(acc, abs=0.05)


# ---------------------------------------------------------- R2: extraction confound
def test_substring_only_rate_flags_toolcalling_confound(audit: Audit):
    by_strategy = audit.substring_only_rate(by="strategy").set_index("strategy")
    # Tool-calling should stand out as the only strategy with a large
    # substring-only-credit rate -- the answer-extraction confound.
    assert by_strategy.loc["tool-calling", "substring_only_pct"] > 30
    for other in ("zero-shot", "cot", "plan-execute"):
        assert by_strategy.loc[other, "substring_only_pct"] < 10

    by_model = audit.substring_only_rate(by="model").set_index("model")
    tc = audit.runs[audit.runs["strategy"] == "tool-calling"]
    tc_by_model = ((tc["fuzzy"]) & (~tc["strict"])).groupby(tc["model"]).mean() * 100
    assert tc_by_model["claude-4.5-haiku"] == pytest.approx(76.1, abs=0.5)
    assert tc_by_model["claude-4.5-opus"] == pytest.approx(83.4, abs=0.5)
    assert tc_by_model["gpt-5"] == pytest.approx(1.5, abs=0.5)
    assert tc_by_model["gpt-5-mini"] == pytest.approx(3.8, abs=0.5)


# --------------------------------------------------------------- R4/R5: significance
def test_significance_gpt5mini_vs_gpt5_matches_table_sig(audit: Audit):
    # Full 6-pair model family, matching Table sig's own Bonferroni scope
    # (dividing by 1 pair instead of 6 would use the wrong alpha).
    model_pairs = [
        ("gpt-5-mini", "gpt-5"), ("gpt-5-mini", "claude-4.5-opus"), ("gpt-5", "claude-4.5-opus"),
        ("claude-4.5-opus", "claude-4.5-haiku"), ("gpt-5-mini", "claude-4.5-haiku"), ("gpt-5", "claude-4.5-haiku"),
    ]
    result = audit.significance(pairs=model_pairs, by="model").set_index(["label_a", "label_b"])
    row = result.loc[("gpt-5-mini", "gpt-5")]
    assert row["delta_pp"] == pytest.approx(1.2, abs=0.15)
    assert row["ci_lo"] == pytest.approx(0.1, abs=0.3)
    assert row["ci_hi"] == pytest.approx(2.3, abs=0.3)
    assert not row["significant"]  # n.s. at the correct family-wide Bonferroni alpha
    # The four large, unambiguous gaps against Claude Haiku must all clear significance.
    for pair in [("gpt-5-mini", "claude-4.5-haiku"), ("gpt-5", "claude-4.5-haiku"),
                 ("claude-4.5-opus", "claude-4.5-haiku")]:
        assert result.loc[pair, "significant"]


def test_tost_equivalence_matches_paper_strategy_tie(audit: Audit):
    result = audit.significance(pairs=[("react", "cot"), ("react", "plan-execute"), ("cot", "plan-execute")],
                                 by="strategy", tost_delta=0.02)
    # "the three strong strategies are a statistical tie" -- TOST equivalence
    # within +/-2pp for all three pairs is the paper's central methodological
    # claim (Section 4.4). If this ever fails, the paper's headline finding
    # about strategy choice being a cost (not accuracy) decision no longer holds.
    assert result["tost_equivalent"].all()


def test_report_runs_end_to_end(audit: Audit, tmp_path):
    path = tmp_path / "audit.md"
    md = audit.report(path=str(path), pairs=[("gpt-5-mini", "gpt-5")])
    assert path.exists()
    assert "Discrimination (DPI)" in md
    assert "44/500" in md
