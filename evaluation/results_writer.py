"""
Writes split base/followup content files and results JSON files,
then triggers HTML report generation.
"""

import json
from pathlib import Path

from evaluation.report import generate_html_report


def _safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "-")


def _mr_folder_name(relation_name: str, relation_id: int | None) -> str:
    if relation_id is not None:
        return f"mr-{relation_id:02d}-{relation_name}"
    return f"mr-{relation_name}"


def write_base_content(results: list[dict], model_name: str, results_dir: str) -> Path:
    """
    Write base LLM content to results/<model>/base/base.json.
    Strips followup fields.
    """
    FOLLOWUP_FIELDS = {
        "relation", "followup_instruction_prefix", "followup_instruction_suffix",
        "followup_question", "followup_response", "followup_response_samples",
        "followup_error", "followup_prompt_tokens", "followup_completion_tokens",
        "followup_total_tokens", "followup_thinking_tokens", "followup_token_efficiency",
        "followup_llm_call_time",
    }
    out_dir = Path(results_dir) / _safe_name(model_name) / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "base.json"

    stripped = [
        {k: v for k, v in entry.items() if k not in FOLLOWUP_FIELDS}
        for entry in results
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stripped, f, indent=2, ensure_ascii=False)
    print(f"  Base content  -> {out_path}")
    return out_path


def write_followup_content(
    results: list[dict],
    model_name: str,
    relation_name: str,
    relation_id: int | None,
    results_dir: str,
) -> Path:
    """
    Write followup LLM content to results/<model>/mr-XX-name/followup.json.
    Keeps question context + followup fields only.
    """
    KEEP_FIELDS = {
        "question_id", "question_title", "difficulty", "platform", "model_name",
        "question", "public_test_cases", "private_test_cases", "metadata",
        "relation", "followup_instruction_prefix", "followup_instruction_suffix",
        "followup_question", "followup_response", "followup_response_samples",
        "followup_error", "followup_prompt_tokens", "followup_completion_tokens",
        "followup_total_tokens", "followup_thinking_tokens", "followup_token_efficiency",
        "followup_llm_call_time",
    }
    folder = _mr_folder_name(relation_name, relation_id)
    out_dir = Path(results_dir) / _safe_name(model_name) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "followup.json"

    filtered = [
        {k: v for k, v in entry.items() if k in KEEP_FIELDS}
        for entry in results
        if entry.get("followup_response")
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"  Followup content -> {out_path}")
    return out_path


def write_results_json(metric_rows: list[dict], out_path: Path) -> None:
    """Write per-problem metrics and pass/fail counts to a results JSON file."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2, ensure_ascii=False)
    print(f"  Results JSON  -> {out_path}")


def write_all_outputs(
    results: list[dict],
    metric_rows: list[dict],
    model_name: str,
    relation_name: str | None,
    relation_id: int | None,
    results_dir: str,
) -> None:
    """
    Write all split output files and generate the HTML report.
    Called at the end of runtests.py when --run-tests is active.
    """
    has_mr = bool(relation_name)

    # Write content files
    base_content_path = write_base_content(results, model_name, results_dir)
    followup_content_path = None
    if has_mr:
        followup_content_path = write_followup_content(
            results, model_name, relation_name, relation_id, results_dir
        )

    # ── Base results JSON ────────────────────────────────────────────────────
    base_results_path = base_content_path.parent / "base.results.json"
    base_rows = []
    for row in metric_rows:
        src_var = row.get("source_variability") or {}
        base_rows.append({
            "question_id": row["id"],
            "question_title": row.get("question_title"),
            "difficulty": row.get("difficulty"),
            "platform": row.get("platform"),

            # Test pass metrics
            "extractable_code": row.get("source_extractable"),
            "passes_tests": row.get("source_passes_tests"),
            "passed": row.get("source_passed"),          # legacy alias
            "tests_passed": row.get("tests_passed"),
            "tests_failed": row.get("tests_failed"),
            "tests_total": row.get("tests_total"),
            "test_run_time": row.get("source_test_run_time"),

            # Static code metrics
            "ncloc": row.get("source_ncloc"),
            "cyclomatic_complexity": row.get("source_cc"),

            # Variability
            "mean_distance": src_var.get("mean_distance"),
            "std_distance": src_var.get("std_distance"),
            "n_pairs": src_var.get("n_pairs"),

            # Token metrics
            "prompt_tokens": row.get("source_prompt_tokens"),
            "completion_tokens": row.get("source_completion_tokens"),
            "total_tokens": row.get("source_total_tokens"),
            "thinking_tokens": row.get("source_thinking_tokens"),
            "token_efficiency": row.get("source_token_efficiency"),

            # Timing
            "llm_call_time": row.get("source_llm_call_time"),
        })
    write_results_json(base_rows, base_results_path)

    # ── Followup results JSON ────────────────────────────────────────────────
    followup_results_path = None
    if has_mr:
        folder = _mr_folder_name(relation_name, relation_id)
        followup_results_path = (
            Path(results_dir) / _safe_name(model_name) / folder / "followup.results.json"
        )
        followup_rows = []
        for row in metric_rows:
            if row.get("followup_ncloc") is None:
                continue
            flw_var = row.get("followup_variability") or {}
            followup_rows.append({
                "question_id": row["id"],
                "question_title": row.get("question_title"),
                "difficulty": row.get("difficulty"),
                "platform": row.get("platform"),

                # Test pass metrics
                "extractable_code": row.get("followup_extractable"),
                "passes_tests": row.get("followup_passes_tests"),
                "passes_same_tests": row.get("passes_same_tests"),
                "passed": row.get("followup_passed"),    # legacy alias
                "tests_passed": row.get("followup_tests_passed"),
                "tests_failed": row.get("followup_tests_failed"),
                "tests_total": row.get("followup_tests_total"),
                "test_run_time": row.get("followup_test_run_time"),

                # Static code metrics
                "ncloc": row.get("followup_ncloc"),
                "cyclomatic_complexity": row.get("followup_cc"),

                # Variability
                "mean_distance": flw_var.get("mean_distance"),
                "std_distance": flw_var.get("std_distance"),
                "n_pairs": flw_var.get("n_pairs"),

                # MR
                "mr_holds": row.get("mr_holds"),
                "code_sim": row.get("code_sim"),

                # Token metrics
                "prompt_tokens": row.get("followup_prompt_tokens"),
                "completion_tokens": row.get("followup_completion_tokens"),
                "total_tokens": row.get("followup_total_tokens"),
                "thinking_tokens": row.get("followup_thinking_tokens"),
                "token_efficiency": row.get("followup_token_efficiency"),

                # Timing
                "llm_call_time": row.get("followup_llm_call_time"),
            })
        write_results_json(followup_rows, followup_results_path)

    # ── HTML report ──────────────────────────────────────────────────────────
    safe = _safe_name(model_name)
    if has_mr:
        folder = _mr_folder_name(relation_name, relation_id)
        html_path = Path(results_dir) / safe / f"{folder}.html"
    else:
        html_path = Path(results_dir) / safe / "base.html"

    generate_html_report(
        base_results_path=base_results_path,
        followup_results_path=followup_results_path,
        out_path=html_path,
        model_name=model_name,
        relation_name=relation_name,
    )