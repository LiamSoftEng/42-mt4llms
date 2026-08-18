"""
Writes split base/followup content files and results JSON files,
then triggers HTML report generation.
"""

import json
import time
from pathlib import Path

from evaluation.report import generate_html_report
from evaluation.paths import build_results_dir, mr_folder_name, safe_name
from evaluation.metric_violations import check_metric_violations


def _merge_by_question_id(existing_path: Path, new_entries: list[dict]) -> list[dict]:
    """Merge new_entries into whatever is already at existing_path, keyed by
    question_id. New entries replace existing ones with the same id; anything
    else already on disk is preserved."""
    existing: list[dict] = []
    if existing_path.exists():
        try:
            with open(existing_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing = [e for e in loaded if isinstance(e, dict)]
        except (OSError, json.JSONDecodeError):
            existing = []

    incoming_ids = {e.get("question_id") for e in new_entries}
    return [e for e in existing if e.get("question_id") not in incoming_ids] + new_entries


def write_base_content(results: list[dict], model_name: str, results_dir: str) -> Path:
    """
    Write base LLM content to results/<model>/base/base.json.
    Strips followup fields. Merges into any existing base.json rather
    than overwriting it, since `results` here may only be a subset
    (e.g. a single --mr run's questions), while base.json is meant to
    be a running ledger across all runs.
    """
    FOLLOWUP_FIELDS = {
        "relation", "followup_instruction_prefix", "followup_instruction_suffix",
        "followup_question", "followup_response", "followup_response_samples",
        "followup_error", "followup_prompt_tokens", "followup_completion_tokens",
        "followup_total_tokens", "followup_thinking_tokens", "followup_token_efficiency",
        "followup_llm_call_time",
    }
    out_dir = Path(results_dir) / safe_name(model_name) / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "base.json"

    stripped = [
        {k: v for k, v in entry.items() if k not in FOLLOWUP_FIELDS}
        for entry in results
    ]

    merged = _merge_by_question_id(out_path, stripped)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
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
    folder = mr_folder_name(relation_name, relation_id)
    out_dir = Path(results_dir) / safe_name(model_name) / folder
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


def write_results_json(metric_rows: list[dict], out_path: Path, merge: bool = False) -> None:
    """Write per-problem metrics and pass/fail counts to a results JSON file.

    If merge=True, merges into any existing file at out_path by question_id
    instead of overwriting it outright — used for base.results.json, which
    is a running ledger across all --mr runs, not a per-run artifact.
    """
    rows = _merge_by_question_id(out_path, metric_rows) if merge else metric_rows
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
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
    write_results_json(base_rows, base_results_path, merge=True)

    # ── Followup results JSON ────────────────────────────────────────────────
    followup_results_path = None
    if has_mr:
        folder = mr_folder_name(relation_name, relation_id)
        followup_results_path = (
            Path(results_dir) / safe_name(model_name) / folder / "followup.results.json"
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
    safe = safe_name(model_name)
    if has_mr:
        out_dir = build_results_dir(model_name, relation_name, relation_id)
        html_path = out_dir / f"{out_dir.name}.html"
    else:
        html_path = Path(results_dir) / safe / "base" / "base.html"

    generate_html_report(
        base_results_path=base_results_path,
        followup_results_path=followup_results_path,
        out_path=html_path,
        model_name=model_name,
        relation_name=relation_name,
    )


def write_violations_log(model_name: str, results_dir: str) -> Path:
    """
    Scan each per-MR followup results file for violated metamorphic relations,
    where a violation is either a failed test-verdict match (mr_holds == False)
    or a metric delta exceeding its configured threshold (METRIC_VARIANCE_THRESHOLDS).
    Source-side metrics are joined from the base running ledger by question_id.

    The log is JSONL (one object per line), matching the helper_call_log
    format, with a timestamp first on each record.
    """
    model_dir = Path(results_dir) / safe_name(model_name)
    out_path = model_dir / "violations.json"

    base_by_qid: dict[str, dict] = {}
    base_path = model_dir / "base" / "base.results.json"
    if base_path.exists():
        try:
            with open(base_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                base_by_qid = {
                    e.get("question_id"): e
                    for e in loaded
                    if isinstance(e, dict) and isinstance(e.get("question_id"), str) and e["question_id"]
                }
        except (OSError, json.JSONDecodeError):
            base_by_qid = {}

    violations: list[dict] = []
    if model_dir.exists():
        for folder in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("mr-")):
            flw_path = folder / "followup.results.json"
            if not flw_path.exists():
                continue
            try:
                with open(flw_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                qid = row.get("question_id")
                b = base_by_qid.get(qid, {})

                test_violation = row.get("mr_holds") is False
                metric_violations = check_metric_violations(b, row)

                if not test_violation and not metric_violations:
                    continue

                violations.append({
                    "model": model_name,
                    "relation": folder.name,
                    "question_id": qid,
                    "question_title": row.get("question_title"),
                    "difficulty": row.get("difficulty"),
                    "platform": row.get("platform"),
                    "source_passed": b.get("passed"),
                    "source_tests_passed": b.get("tests_passed"),
                    "source_tests_failed": b.get("tests_failed"),
                    "source_tests_total": b.get("tests_total"),
                    "followup_passed": row.get("passed"),
                    "followup_tests_passed": row.get("tests_passed"),
                    "followup_tests_failed": row.get("tests_failed"),
                    "followup_tests_total": row.get("tests_total"),
                    "passes_same_tests": row.get("passes_same_tests"),
                    "code_sim": row.get("code_sim"),
                    "test_violation": test_violation,
                    "metric_violations": metric_violations,
                })

    violations.sort(key=lambda v: (v.get("relation") or "", v.get("question_id") or ""))
    model_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for v in violations:
            f.write(json.dumps({"timestamp": now, **v}, ensure_ascii=False) + "\n")
    print(f"  Violations log -> {out_path} ({len(violations)} violations)")
    return out_path


def write_consolidated_outputs(model_name: str, results_dir: str) -> None:
    """
    Write the post-test consolidated outputs for a model:
      - merged HTML report of base + all per-MR runs
      - violations log
    Safe to call repeatedly; reads whatever results exist on disk.
    """
    from evaluation.report import generate_merged_html_report

    generate_merged_html_report(model_name, results_dir)
    write_violations_log(model_name, results_dir)