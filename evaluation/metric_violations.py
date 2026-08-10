"""
Shared metric-violation checking logic.

Used by both results_writer.py (violations.json log) and report.py
(per-cell highlighting in the HTML reports), so the two always agree on
what counts as a metamorphic-relation violation. Lives in its own module
rather than in results_writer.py (its original home) because report.py
needs it too, and results_writer.py already imports from report.py —
report.py importing back from results_writer.py would be circular.
"""

from config.settings import METRIC_VARIANCE_THRESHOLDS, get_metric_threshold


def check_metric_violations(base_row: dict, followup_row: dict) -> dict:
    """
    Compare base vs followup metric values against METRIC_VARIANCE_THRESHOLDS.

    Every metric except code_sim is checked as a max-allowed |delta| between
    base and followup. code_sim is checked as a minimum allowed value on the
    followup alone, since there's no base code_sim to diff against.

    Returns a dict of {metric_name: {"base":..., "followup":..., "delta":...,
    "threshold":...}} for every metric that breached its threshold. Metrics
    missing on either side, or with no configured threshold, are skipped.
    """
    question_id = followup_row.get("question_id") or base_row.get("question_id")
    violations: dict = {}

    for metric_name in METRIC_VARIANCE_THRESHOLDS:
        threshold = get_metric_threshold(metric_name, question_id)
        if threshold is None:
            continue

        if metric_name == "code_sim":
            followup_value = followup_row.get("code_sim")
            if not isinstance(followup_value, (int, float)):
                continue
            if followup_value < threshold:
                violations["code_sim"] = {
                    "base": None,
                    "followup": followup_value,
                    "delta": None,
                    "threshold": threshold,
                }
            continue

        base_value = base_row.get(metric_name)
        followup_value = followup_row.get(metric_name)
        if not isinstance(base_value, (int, float)) or not isinstance(followup_value, (int, float)):
            continue

        delta = followup_value - base_value
        if abs(delta) > threshold:
            violations[metric_name] = {
                "base": base_value,
                "followup": followup_value,
                "delta": delta,
                "threshold": threshold,
            }

    return violations