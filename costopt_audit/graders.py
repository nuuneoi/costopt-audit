"""Pluggable answer graders.

``fuzzy_default`` and ``strict_default`` are lifted verbatim from the CostOpt
paper's ``compute_robustness.py`` so that running this package on the released
CostBench runs reproduces every paper number bit-for-bit (see
``tests/test_costbench_reproduction.py``). Bring your own callable matching the
``Grader`` signature to grade a different benchmark's answer format.
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Protocol


class Grader(Protocol):
    """A grader decides whether ``output`` is correct given ``expected``.

    ``domain`` is an optional free-text hint (e.g. "math_reasoning", "airline")
    that a grader may use to apply domain-specific tolerance. Graders that
    don't need it can ignore the argument.
    """

    def __call__(self, expected: Optional[str], output: str, domain: Optional[str] = None) -> bool: ...


def normalize_answer(text: Optional[str]) -> str:
    if not text:
        return ""
    text = str(text).strip().lower()
    for p in ("answer:", "the answer is", "result:", "=", "the result is"):
        if text.startswith(p):
            text = text[len(p):].strip()
    return " ".join(text.split()).rstrip(".,;:!?")


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"\$?\s*([\d,]+(?:\.\d{1,4})?)", str(text))
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    m = re.search(r"([-]?\d+(?:\.\d+)?)", str(text))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def fuzzy_default(expected: Optional[str], output: str, domain: Optional[str] = None) -> bool:
    """Lenient grader: exact/substring match plus domain-aware numeric tolerance.

    Can over-credit (a verbose answer that happens to contain the target
    substring) -- pair with ``strict_default`` and report both (Section 3.4,
    "truth-bracketing") rather than trusting either alone.
    """
    if output is None or (isinstance(output, float) and output != output):  # NaN-safe
        return False
    domain = domain or ""
    ne, na = normalize_answer(expected), normalize_answer(output)
    if ne == na:
        return True
    if len(ne) > 3 and ne in na:
        return True
    if len(na) > 3 and na in ne:
        return True
    if domain in {"math_reasoning", "science_reasoning"}:
        a, b = extract_number(ne), extract_number(na)
        if a is not None and b is not None and a != 0 and abs(a - b) / abs(a) < 0.02:
            return True
    if domain == "airline" or "cost" in domain or "fare" in domain:
        a, b = extract_number(ne), extract_number(na)
        if a is not None and b is not None and abs(a - b) < 0.01:
            return True
    if ne in ("yes", "no") and ne in na:
        return True
    if " " not in ne and len(ne) <= 20 and ne in na:
        return True
    return False


def strict_default(expected: Optional[str], output: str, domain: Optional[str] = None) -> bool:
    """``fuzzy_default`` minus every generic substring rule: exact-normalized
    match and domain-aware numeric/dollar tolerance only. Can under-credit a
    correct answer buried in extra formatting or narration."""
    if output is None or (isinstance(output, float) and output != output):
        return False
    domain = domain or ""
    ne, na = normalize_answer(expected), normalize_answer(output)
    if ne == na:
        return True
    if domain in {"math_reasoning", "science_reasoning"}:
        a, b = extract_number(ne), extract_number(na)
        if a is not None and b is not None and a != 0 and abs(a - b) / abs(a) < 0.02:
            return True
    if domain == "airline" or "cost" in domain or "fare" in domain:
        a, b = extract_number(ne), extract_number(na)
        if a is not None and b is not None and abs(a - b) < 0.01:
            return True
    return False


class DualGrader:
    """Bundles a lenient and a strict grader so an :class:`Audit` can report
    both and bracket the truth, instead of presenting one accuracy number as
    ground truth."""

    def __init__(self, fuzzy: Callable[..., bool], strict: Callable[..., bool]):
        self.fuzzy = fuzzy
        self.strict = strict


def dual(fuzzy: Callable[..., bool] = fuzzy_default, strict: Callable[..., bool] = strict_default) -> DualGrader:
    return DualGrader(fuzzy=fuzzy, strict=strict)


def llm_judge(model: str, **kwargs) -> Callable[..., bool]:
    """Convenience stub for an LLM-as-judge grader. Not implemented here (it
    requires an API call and is non-deterministic by nature) -- wire up your
    own callable with the :class:`Grader` signature and pass it to
    ``graders.dual(fuzzy=my_llm_judge, strict=my_llm_judge)`` if you want the
    dual-grader discipline to apply to a judge-graded benchmark too."""
    raise NotImplementedError(
        "llm_judge is a placeholder for the Grader protocol -- supply your own "
        "callable(expected, output, domain) -> bool backed by whatever LLM/API "
        "you use as a judge."
    )
