"""Shared serializers for the metric rows produced by ``runtests.py``."""

BASE_CSV_FIELDS = [
    "id",
    "extractable_code", "passes_tests", "source_passed",
    "tests_passed", "tests_failed", "tests_total", "test_run_time",
    "source_ncloc", "source_cc",
    "source_mean_distance", "source_std_distance", "source_pairs",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "thinking_tokens", "token_efficiency", "llm_call_time",
]

FOLLOWUP_CSV_FIELDS = [
    "followup_extractable", "followup_passes_tests", "passes_same_tests", "followup_passed",
    "followup_tests_passed", "followup_tests_failed", "followup_tests_total",
    "followup_test_run_time", "followup_ncloc", "followup_cc",
    "followup_mean_distance", "followup_std_distance", "followup_pairs",
    "followup_prompt_tokens", "followup_completion_tokens", "followup_total_tokens",
    "followup_thinking_tokens", "followup_token_efficiency", "followup_llm_call_time",
    "code_sim", "mr_holds",
]


def serialize_metric_side(row: dict, side: str) -> dict:
    """Convert an internal metric row to the stable per-side results schema."""
    if side not in {"source", "followup"}:
        raise ValueError(f"Unknown metric side: {side}")

    followup = side == "followup"
    prefix = "followup_" if followup else "source_"
    variability = row.get(f"{prefix}variability") or {}
    tests_prefix = "followup_" if followup else ""

    output = {
        "question_id": row["id"],
        "question_title": row.get("question_title"),
        "difficulty": row.get("difficulty"),
        "platform": row.get("platform"),
        "extractable_code": row.get(f"{prefix}extractable"),
        "passes_tests": row.get(f"{prefix}passes_tests"),
        "passed": row.get(f"{prefix}passed"),
        "tests_passed": row.get(f"{tests_prefix}tests_passed"),
        "tests_failed": row.get(f"{tests_prefix}tests_failed"),
        "tests_total": row.get(f"{tests_prefix}tests_total"),
        "test_run_time": row.get(f"{prefix}test_run_time"),
        "ncloc": row.get(f"{prefix}ncloc"),
        "cyclomatic_complexity": row.get(f"{prefix}cc"),
        "mean_distance": variability.get("mean_distance"),
        "std_distance": variability.get("std_distance"),
        "n_pairs": variability.get("n_pairs"),
        "prompt_tokens": row.get(f"{prefix}prompt_tokens"),
        "completion_tokens": row.get(f"{prefix}completion_tokens"),
        "total_tokens": row.get(f"{prefix}total_tokens"),
        "thinking_tokens": row.get(f"{prefix}thinking_tokens"),
        "token_efficiency": row.get(f"{prefix}token_efficiency"),
        "llm_call_time": row.get(f"{prefix}llm_call_time"),
    }
    if followup:
        output.update({
            "passes_same_tests": row.get("passes_same_tests"),
            "mr_holds": row.get("mr_holds"),
            "code_sim": row.get("code_sim"),
        })
    return output


def serialize_metric_csv_row(row: dict, has_mr: bool) -> dict:
    """Flatten a metric row to the existing CSV column names."""
    source = serialize_metric_side(row, "source")
    output = {
        "id": source["question_id"],
        "extractable_code": source["extractable_code"],
        "passes_tests": source["passes_tests"],
        "source_passed": source["passed"],
        "tests_passed": source["tests_passed"],
        "tests_failed": source["tests_failed"],
        "tests_total": source["tests_total"],
        "test_run_time": source["test_run_time"],
        "source_ncloc": source["ncloc"],
        "source_cc": source["cyclomatic_complexity"],
        "source_mean_distance": source["mean_distance"],
        "source_std_distance": source["std_distance"],
        "source_pairs": source["n_pairs"],
        "prompt_tokens": source["prompt_tokens"],
        "completion_tokens": source["completion_tokens"],
        "total_tokens": source["total_tokens"],
        "thinking_tokens": source["thinking_tokens"],
        "token_efficiency": source["token_efficiency"],
        "llm_call_time": source["llm_call_time"],
    }
    if has_mr:
        followup = serialize_metric_side(row, "followup")
        output.update({
            "followup_extractable": followup["extractable_code"],
            "followup_passes_tests": followup["passes_tests"],
            "passes_same_tests": followup["passes_same_tests"],
            "followup_passed": followup["passed"],
            "followup_tests_passed": followup["tests_passed"],
            "followup_tests_failed": followup["tests_failed"],
            "followup_tests_total": followup["tests_total"],
            "followup_test_run_time": followup["test_run_time"],
            "followup_ncloc": followup["ncloc"],
            "followup_cc": followup["cyclomatic_complexity"],
            "followup_mean_distance": followup["mean_distance"],
            "followup_std_distance": followup["std_distance"],
            "followup_pairs": followup["n_pairs"],
            "followup_prompt_tokens": followup["prompt_tokens"],
            "followup_completion_tokens": followup["completion_tokens"],
            "followup_total_tokens": followup["total_tokens"],
            "followup_thinking_tokens": followup["thinking_tokens"],
            "followup_token_efficiency": followup["token_efficiency"],
            "followup_llm_call_time": followup["llm_call_time"],
            "code_sim": followup["code_sim"],
            "mr_holds": followup["mr_holds"],
        })
    return output
