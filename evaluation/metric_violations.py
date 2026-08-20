"""
Shared metric-violation checking logic.

Used by both results_writer.py (violations.json log), report.py (per-cell
highlighting in the HTML reports), and metric_rows.py (the "violation" field
on each followup.results.json entry) — so all three always agree on what
counts as a metamorphic-relation violation. Lives in its own module rather
than in results_writer.py (its original home) because report.py needs it
too, and results_writer.py already imports from report.py — report.py
importing back from results_writer.py would be circular.

Violation detection is a per-question one-sample t-test: each followup
metric value is tested against a baseline distribution built from repeat
(no-transformation) runs for that same question (config.settings.
REPEAT_BASELINE_DIRS), rather than a fixed +/- delta threshold. See
evaluation.metrics.load_repeat_baseline and
evaluation.metrics.compute_ttest_violation.
"""

from config.settings import TTEST_ALPHA
from evaluation.metrics import load_repeat_baseline, compute_ttest_violation

# Metric names checked against the repeat baseline. Must match both the
# field names in evaluation.metrics.load_repeat_baseline()'s output and the
# keys on the followup row produced by evaluation.metric_rows.serialize_metric_side.
_METRIC_NAMES = ("ncloc", "cyclomatic_complexity", "llm_call_time", "total_tokens", "token_efficiency", "code_sim")

# scipy.stats.ttest_1samp treats the repeat baseline as the sample and the
# follow-up value as popmean. Structural/resource metrics use two-sided tests
# when a change in either direction is relevant. For directional performance
# metrics, ``less`` detects a slower follow-up and ``greater`` detects lower
# efficiency or similarity.
_METRIC_ALTERNATIVES = {
    "ncloc": "two-sided",
    "cyclomatic_complexity": "two-sided",
    "llm_call_time": "less",
    "total_tokens": "two-sided",
    "token_efficiency": "greater",
    "code_sim": "greater",
}

# Static-metric error sentinels: compute_metrics() returns these instead of a
# real value when code couldn't be parsed. They're valid ints but meaningless
# as measurements, so they must never be handed to the t-test as a real
# followup value.
_ERROR_SENTINELS = {
    "ncloc": {-1},
    "cyclomatic_complexity": {-1, -2},
}


def _is_valid_metric_value(metric_name: str, value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value not in _ERROR_SENTINELS.get(metric_name, set())
    )


def check_metric_violations(question_id: str, followup_row: dict) -> dict:
    """
    Test each of _METRIC_NAMES on the followup row against that question's
    repeat-run baseline distribution via a one-sample t-test.

    Returns a dict of {metric_name: ttest_result} for every metric whose
    t-test came back significant (p < config.settings.TTEST_ALPHA).
    ttest_result is the dict returned by compute_ttest_violation (n_baseline,
    baseline_mean, baseline_std, followup, t_stat, p_value, alternative,
    significant).

    A metric is silently skipped (not counted as a violation) when there's
    no baseline for this question at all, the followup value is a static-
    metric error sentinel (-1/-2, meaning the code didn't parse), fewer than
    two baseline samples exist for that specific metric, or the followup
    value is otherwise non-numeric.
    """
    baseline = load_repeat_baseline()
    q_baseline = baseline.get(question_id)
    if not q_baseline:
        return {}

    violations: dict = {}
    for metric_name in _METRIC_NAMES:
        followup_value = followup_row.get(metric_name)
        if not _is_valid_metric_value(metric_name, followup_value):
            continue

        baseline_samples = list(q_baseline.get(metric_name, ()))
        result = compute_ttest_violation(
            baseline_samples,
            followup_value,
            alpha=TTEST_ALPHA,
            alternative=_METRIC_ALTERNATIVES[metric_name],
        )
        if result is not None and result["significant"]:
            violations[metric_name] = result

    return violations


def compute_violation_field(question_id: str, followup_row: dict) -> str:
    """
    Overall tri-state MR verdict for one followup row, written into the
    "violation" field of followup.results.json entries:

      "true"  — the test verdict flipped (mr_holds is False), or at least
                one metric's t-test against the repeat baseline came back
                significant.
      "false" — a test verdict was recorded and held (mr_holds is True) and
                no metric was significant, OR no test verdict exists
                (mr_holds is None) but the question has a repeat baseline to
                compare metrics against and none of them were significant.
      "n/a"   — nothing could actually be compared: no test verdict AND no
                repeat baseline exists for this question at all.
    """
    metric_violations = check_metric_violations(question_id, followup_row)
    mr_holds = followup_row.get("mr_holds")

    if mr_holds is False or metric_violations:
        return "true"
    if mr_holds is True:
        return "false"

    # mr_holds is None: no test-based comparison was made for this row (e.g.
    # no extractable followup code, or no private tests). Only "false" if we
    # at least had baseline data to check metrics against; otherwise nothing
    # was actually compared.
    if not load_repeat_baseline().get(question_id):
        return "n/a"
    return "false"
