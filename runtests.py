"""
Run all test cases against model responses.
If the results file contains followup responses (from --mr), tests both
and computes MR verdict.

Usage:
    python runtests.py --results results/openai/base/base.json
    python runtests.py --results results/openai/base/base.json --n 5
    python runtests.py --results results/openai/base/base.json --stdout
    python runtests.py --results results/openai/mr-05-polarity_negative_formulation.json --followup-only
    python runtests.py --results results/openai/base/base.json --csv results/openai/metrics.csv
"""

import json
import argparse
import csv
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from evaluation.executor import (
    extract_code, get_all_tests, get_func_name, run_test,
)
from evaluation.metrics import (
    compute_metrics,
    compute_code_similarity,
    compute_pairwise_code_variability,
    compute_extractable_code,
    compute_passes_same_tests,
    compute_passes_tests,
    compute_token_count,
    compute_token_efficiency,
)
from evaluation.report import format_decimal
from evaluation.metric_rows import BASE_CSV_FIELDS, FOLLOWUP_CSV_FIELDS, serialize_metric_csv_row


def _response_samples(entry: dict, sample_key: str, primary_key: str) -> list[str]:
    samples = entry.get(sample_key)
    if isinstance(samples, list) and samples:
        return [s for s in samples if isinstance(s, str)]
    primary = entry.get(primary_key, "")
    return [primary] if isinstance(primary, str) and primary else []


def _code_samples(responses: list[str]) -> list[str | None]:
    return [extract_code(r) for r in responses]


def _delta(value) -> str:
    return f"{value:+}" if isinstance(value, (int, float)) else value


def _variation_greater(source_mean_dist, followup_mean_dist) -> str:
    if isinstance(source_mean_dist, (int, float)) and isinstance(followup_mean_dist, (int, float)):
        return "YES" if followup_mean_dist > source_mean_dist else "NO"
    return "N/A"


def _write_metrics_csv(csv_path: str | Path, metric_rows: list[dict], has_mr: bool) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = BASE_CSV_FIELDS + (FOLLOWUP_CSV_FIELDS if has_mr else [])

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(serialize_metric_csv_row(row, has_mr))
    print(f"\n  Wrote metrics CSV -> {path}")


def test_response(
    response: str,
    entry: dict,
    func_name: str | None,
    debug: bool,
    label: str = "source",
) -> tuple[bool | None, int, int, int, str | None, list[tuple], float]:
    """
    Run all tests for one response.

    Returns:
        all_passed      bool | None  — None means skipped (no code / no tests)
        tests_passed    int
        tests_failed    int
        tests_total     int
        code            str | None   — extracted code block
        test_results    list[tuple]  — raw (passed, actual, error, elapsed) per test
        total_elapsed   float        — cumulative wall-clock seconds across all tests
    """
    code = extract_code(response)
    if not code:
        print(f"  [{label}] SKIP — no code block found")
        if debug:
            print(f"    Response: {response[:200]}")
        return None, 0, 0, 0, None, [], 0.0

    tests = get_all_tests(entry)
    if not tests:
        print(f"  [{label}] SKIP — no test cases")
        return None, 0, 0, 0, None, [], 0.0

    tests_passed = 0
    tests_failed = 0
    test_results: list[tuple] = []
    total_elapsed = 0.0

    for i, test in enumerate(tests):
        testtype = test.get("testtype", "stdin")
        passed, actual, error, elapsed = run_test(code, test, func_name, debug=debug)
        test_results.append((passed, actual, error, elapsed))
        total_elapsed += elapsed

        if passed:
            tests_passed += 1
        else:
            tests_failed += 1
            print(f"  [{label}] Test {i+1} [{testtype}]: FAIL")
            if error:
                print(f"    Error:    {error[:200]}")
            else:
                print(f"    Expected: {test.get('output', '')[:80].strip()}")
                print(f"    Got:      {actual[:80].strip()}")

    all_passed = tests_failed == 0
    total = tests_passed + tests_failed
    print(f"  [{label}] {tests_passed}/{total} passed  ({total_elapsed:.2f}s)")
    return all_passed, tests_passed, tests_failed, total, code, test_results, total_elapsed


def evaluate_response(
    response: str,
    entry: dict,
    func_name: str | None,
    debug: bool,
    label: str,
    sample_key: str,
    primary_key: str,
) -> dict:
    """Execute and measure one source or followup response."""
    passed, tests_passed, tests_failed, tests_total, code, test_results, run_time = test_response(
        response, entry, func_name, debug, label=label
    )
    metrics = compute_metrics(code)
    return {
        "passed": passed,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_total": tests_total,
        "test_run_time": round(run_time, 4),
        "code": code,
        "test_results": test_results,
        "extractable": compute_extractable_code(response),
        "passes_tests": compute_passes_tests(test_results),
        "ncloc": metrics["ncloc"],
        "cc": metrics["cyclomatic_complexity"],
        "variability": compute_pairwise_code_variability(
            _code_samples(_response_samples(entry, sample_key, primary_key))
        ),
    }


def _metric_side(evaluation: dict, side: str) -> dict:
    """Map an evaluated response to the internal metric-row field names."""
    prefix = "followup_" if side == "followup" else "source_"
    tests_prefix = "followup_" if side == "followup" else ""
    return {
        f"{prefix}extractable": evaluation["extractable"],
        f"{prefix}passes_tests": evaluation["passes_tests"],
        f"{prefix}passed": evaluation["passed"],
        f"{tests_prefix}tests_passed": evaluation["tests_passed"],
        f"{tests_prefix}tests_failed": evaluation["tests_failed"],
        f"{tests_prefix}tests_total": evaluation["tests_total"],
        f"{prefix}test_run_time": evaluation["test_run_time"],
        f"{prefix}ncloc": evaluation["ncloc"],
        f"{prefix}cc": evaluation["cc"],
        f"{prefix}variability": evaluation["variability"],
    }


def run_tests(
    results_path: str,
    n: int = None,
    debug: bool = False,
    followup_only: bool = False,
    csv_path: str | None = None,
    write_outputs: bool = False,
) -> list[dict]:
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if n:
        results = results[:n]

    total = passed = failed = skipped = 0
    mr_holds_count = mr_violated_count = 0
    has_mr = any(r.get("relation") for r in results) and not followup_only
    metric_rows = []

    for entry in results:
        qid = entry["question_id"]
        print(f"\n{'='*60}")
        print(f"  {qid} — {entry['question_title']} ({entry['difficulty']} / {entry['platform']})")
        print(f"{'='*60}")

        total += 1

        if entry.get("error"):
            print(f"  [SKIP] LLM error: {entry['error']}")
            skipped += 1
            continue

        func_name = get_func_name(entry)

        # Pull token fields from entry (written by runner.py)
        src_tokens = compute_token_count({
            "prompt_tokens": entry.get("prompt_tokens"),
            "completion_tokens": entry.get("completion_tokens"),
            "total_tokens": entry.get("total_tokens"),
            "thinking_tokens": entry.get("thinking_tokens"),
        })
        src_token_efficiency = compute_token_efficiency({
            "completion_tokens": entry.get("completion_tokens"),
            "thinking_tokens": entry.get("thinking_tokens"),
        })
        src_llm_call_time = entry.get("llm_call_time")

        flw_tokens = compute_token_count({
            "prompt_tokens": entry.get("followup_prompt_tokens"),
            "completion_tokens": entry.get("followup_completion_tokens"),
            "total_tokens": entry.get("followup_total_tokens"),
            "thinking_tokens": entry.get("followup_thinking_tokens"),
        })
        flw_token_efficiency = compute_token_efficiency({
            "completion_tokens": entry.get("followup_completion_tokens"),
            "thinking_tokens": entry.get("followup_thinking_tokens"),
        })
        flw_llm_call_time = entry.get("followup_llm_call_time")

        # ── followup-only mode ───────────────────────────────────────────────
        if followup_only:
            primary_response = entry.get("followup_response") or entry.get("response", "")
            label = "followup" if entry.get("followup_response") else "source"
            evaluation = evaluate_response(
                primary_response,
                entry,
                func_name,
                debug,
                label,
                "followup_response_samples" if label == "followup" else "response_samples",
                "followup_response" if label == "followup" else "response",
            )
            if evaluation["passed"] is None:
                skipped += 1
                continue
            if evaluation["passed"]:
                passed += 1
            else:
                failed += 1

            metric_rows.append({
                "id": qid,
                "question_title": entry.get("question_title"),
                "difficulty": entry.get("difficulty"),
                "platform": entry.get("platform"),
                **_metric_side(evaluation, "source"),
                "source_prompt_tokens": src_tokens["prompt_tokens"],
                "source_completion_tokens": src_tokens["completion_tokens"],
                "source_total_tokens": src_tokens["total_tokens"],
                "source_thinking_tokens": src_tokens["thinking_tokens"],
                "source_token_efficiency": src_token_efficiency,
                "source_llm_call_time": src_llm_call_time,
            })
            continue

        # ── Source ──────────────────────────────────────────────────────────
        src_response = entry["response"]
        source = evaluate_response(
            src_response,
            entry,
            func_name,
            debug,
            "source",
            "response_samples",
            "response",
        )

        if source["passed"] is None:
            skipped += 1
            continue
        if source["passed"]:
            passed += 1
        else:
            failed += 1

        row: dict = {
            "id": qid,
            "question_title": entry.get("question_title"),
            "difficulty": entry.get("difficulty"),
            "platform": entry.get("platform"),

            **_metric_side(source, "source"),

            # Source token metrics
            "source_prompt_tokens": src_tokens["prompt_tokens"],
            "source_completion_tokens": src_tokens["completion_tokens"],
            "source_total_tokens": src_tokens["total_tokens"],
            "source_thinking_tokens": src_tokens["thinking_tokens"],
            "source_token_efficiency": src_token_efficiency,
            "source_llm_call_time": src_llm_call_time,

            # Followup placeholders
            "followup_extractable": None,
            "followup_passes_tests": None,
            "passes_same_tests": None,
            "followup_passed": None,
            "followup_tests_passed": None, "followup_tests_failed": None, "followup_tests_total": None,
            "followup_test_run_time": None,
            "followup_ncloc": None, "followup_cc": None,
            "followup_variability": None,
            "followup_prompt_tokens": None,
            "followup_completion_tokens": None,
            "followup_total_tokens": None,
            "followup_thinking_tokens": None,
            "followup_token_efficiency": None,
            "followup_llm_call_time": None,
            "code_sim": None, "mr_holds": None,
        }

        # ── Followup ─────────────────────────────────────────────────────────
        if entry.get("relation") and entry.get("followup_response"):
            flw_response = entry["followup_response"]
            followup = evaluate_response(
                flw_response,
                entry,
                func_name,
                debug,
                "followup",
                "followup_response_samples",
                "followup_response",
            )

            if followup["passed"] is not None:
                holds = source["passed"] == followup["passed"]
                verdict = "HOLDS" if holds else "VIOLATED"
                print(f"  MR [{entry['relation']}]: {verdict}")
                if holds:
                    mr_holds_count += 1
                else:
                    mr_violated_count += 1
                row["mr_holds"] = holds

            row.update(_metric_side(followup, "followup"))

            row["passes_same_tests"] = compute_passes_same_tests(
                source["test_results"], followup.get("test_results", [])
            )
            row["followup_prompt_tokens"] = flw_tokens["prompt_tokens"]
            row["followup_completion_tokens"] = flw_tokens["completion_tokens"]
            row["followup_total_tokens"] = flw_tokens["total_tokens"]
            row["followup_thinking_tokens"] = flw_tokens["thinking_tokens"]
            row["followup_token_efficiency"] = flw_token_efficiency
            row["followup_llm_call_time"] = flw_llm_call_time

            if source["code"] and followup.get("code"):
                row["code_sim"] = compute_code_similarity(source["code"], followup["code"])

        metric_rows.append(row)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Results: {Path(results_path).name}")
    print(f"  Model:   {results[0]['model_name'] if results else 'unknown'}")
    print(f"  PASSED:  {passed}/{total}")
    print(f"  FAILED:  {failed}/{total}")
    print(f"  SKIPPED: {skipped}/{total}")
    if has_mr:
        mr_total = mr_holds_count + mr_violated_count
        print(f"  MR [{results[0].get('relation')}] HOLDS:    {mr_holds_count}/{mr_total}")
        print(f"  MR [{results[0].get('relation')}] VIOLATED: {mr_violated_count}/{mr_total}")

    if metric_rows:
        print()
        if has_mr:
            print(
                f"  {'ID':<15} {'Src':>6} {'Flw':>6} {'SameTst':>7} {'NCLOC':>6} {'dNCL':>5} "
                f"{'CC':>6} {'dCC':>5} {'Sim':>7} {'SrcDst':>7} {'FlwDst':>7} {'Var>':>5} "
                f"{'TkEff':>6} {'LLMs':>6} {'MR':>8}"
            )
            print(
                f"  {'-'*15} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*5} "
                f"{'-'*6} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*5} "
                f"{'-'*6} {'-'*6} {'-'*8}"
            )
            for r in metric_rows:
                src_var = r.get("source_variability") or {}
                flw_var = r.get("followup_variability") or {}
                src_dist = src_var.get("mean_distance")
                flw_dist = flw_var.get("mean_distance")
                ncloc_str = f"{r['source_ncloc']}/{r['followup_ncloc'] if r['followup_ncloc'] is not None else 'N/A'}"
                cc_str = f"{format_decimal(r['source_cc'], 1)}/{format_decimal(r['followup_cc'], 1) if r['followup_cc'] is not None else 'N/A'}"
                mr = ("HOLDS" if r["mr_holds"] else "VIOLATED") if r["mr_holds"] is not None else "N/A"
                src_res = f"{r['tests_passed']}/{r['tests_total']}" if r.get("tests_total") else "N/A"
                flw_res = (
                    f"{r.get('followup_tests_passed', '?')}/{r.get('followup_tests_total', '?')}"
                    if r.get("followup_tests_total") else "N/A"
                )
                same_tst = (
                    "YES" if r.get("passes_same_tests") is True
                    else "NO" if r.get("passes_same_tests") is False
                    else "N/A"
                )
                eff = format_decimal(r.get("source_token_efficiency"), 3) if r.get("source_token_efficiency") is not None else "N/A"
                llm_t = f"{r['source_llm_call_time']:.1f}s" if r.get("source_llm_call_time") is not None else "N/A"
                d_ncloc = _delta(
                    r["followup_ncloc"] - r["source_ncloc"]
                    if r["followup_ncloc"] is not None and r["source_ncloc"] >= 0 and r["followup_ncloc"] >= 0
                    else "N/A"
                )
                d_cc = _delta(
                    round(r["followup_cc"] - r["source_cc"], 2)
                    if r["followup_cc"] is not None and r["source_cc"] >= 0 and r["followup_cc"] >= 0
                    else "N/A"
                )
                print(
                    f"  {r['id']:<15} {src_res:>6} {flw_res:>6} {same_tst:>7} {ncloc_str:>6} {d_ncloc:>5} "
                    f"{cc_str:>6} {d_cc:>5} "
                    f"{format_decimal(r.get('code_sim'), 4):>7} {format_decimal(src_dist, 4):>7} {format_decimal(flw_dist, 4):>7} "
                    f"{_variation_greater(src_dist, flw_dist):>5} "
                    f"{eff:>6} {llm_t:>6} {mr:>8}"
                )
        else:
            print(
                f"  {'ID':<15} {'Tests':>8} {'Extract':>7} {'NCLOC':>6} {'CC':>6} "
                f"{'MeanDst':>8} {'StdDst':>7} {'TkEff':>6} {'LLMs':>6} {'TestT':>6}"
            )
            print(
                f"  {'-'*15} {'-'*8} {'-'*7} {'-'*6} {'-'*6} "
                f"{'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*6}"
            )
            for r in metric_rows:
                src_var = r.get("source_variability") or {}
                tests_str = f"{r.get('tests_passed', '?')}/{r.get('tests_total', '?')}"
                ext = "YES" if r.get("source_extractable") else "NO"
                eff = format_decimal(r.get("source_token_efficiency"), 3) if r.get("source_token_efficiency") is not None else "N/A"
                llm_t = f"{r['source_llm_call_time']:.1f}s" if r.get("source_llm_call_time") is not None else "N/A"
                test_t = f"{r['source_test_run_time']:.2f}s" if r.get("source_test_run_time") is not None else "N/A"
                print(
                    f"  {r['id']:<15} {tests_str:>8} {ext:>7} {r['source_ncloc']:>6} {r['source_cc']:>6} "
                    f"{format_decimal(src_var.get('mean_distance'), 4):>8} {format_decimal(src_var.get('std_distance'), 4):>7} "
                    f"{eff:>6} {llm_t:>6} {test_t:>6}"
                )

    if csv_path:
        _write_metrics_csv(csv_path, metric_rows, has_mr)

    if write_outputs and results:
        from evaluation.results_writer import write_all_outputs
        from config.settings import RESULTS_DIR
        model_name = results[0]["model_name"]
        relation_name = results[0].get("relation") or None
        relation_id = None
        stem = Path(results_path).stem
        if stem.startswith("mr-"):
            parts = stem.split("-")
            if len(parts) >= 2:
                try:
                    relation_id = int(parts[1])
                except ValueError:
                    pass
        write_all_outputs(results, metric_rows, model_name, relation_name, relation_id, RESULTS_DIR)

    print(f"{'='*60}")
    return metric_rows


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test cases against model responses")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N problems")
    parser.add_argument("--debug", action="store_true", help="Print extracted code and generated scripts")
    parser.add_argument("--out", type=str, default=None, help="Path to write test output log")
    parser.add_argument("--csv", type=str, default=None, help="Path to write metrics CSV")
    parser.add_argument("--stdout", action="store_true", help="Also print output to the terminal")
    parser.add_argument("--followup-only", action="store_true", help="Test followup responses only")
    parser.add_argument("--write-outputs", action="store_true", help="Write split files and HTML report")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(args.results).with_suffix(".tests.log")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        stream = Tee(sys.stdout, f) if args.stdout else f
        with redirect_stdout(stream), redirect_stderr(stream):
            run_tests(
                args.results, args.n,
                debug=args.debug,
                followup_only=args.followup_only,
                csv_path=args.csv,
                write_outputs=args.write_outputs,
            )
