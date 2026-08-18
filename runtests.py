"""
Run all test cases against model responses.
If the results file contains followup responses (from --mr), tests both
and computes MR verdict.

Usage:
    python runtests.py --results results/grok/base/base.json
    python runtests.py --results results/grok/base/base.json --n 5
    python runtests.py --results results/grok/base/base.json --stdout
    python runtests.py --results results/grok/mr-05-context_extraneous_narrative.json --followup-only
    python runtests.py --results results/grok/mr-05-context_extraneous_narrative.json --csv results/grok/metrics.csv
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


def _response_samples(entry: dict, sample_key: str, primary_key: str) -> list[str]:
    samples = entry.get(sample_key)
    if isinstance(samples, list) and samples:
        return [s for s in samples if isinstance(s, str)]
    primary = entry.get(primary_key, "")
    return [primary] if isinstance(primary, str) and primary else []


def _code_samples(responses: list[str]) -> list[str | None]:
    return [extract_code(r) for r in responses]


def _fmt(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "N/A"


def _delta(value) -> str:
    return f"{value:+}" if isinstance(value, (int, float)) else value


def _variation_greater(source_mean_dist, followup_mean_dist) -> str:
    if isinstance(source_mean_dist, (int, float)) and isinstance(followup_mean_dist, (int, float)):
        return "YES" if followup_mean_dist > source_mean_dist else "NO"
    return "N/A"


def _write_metrics_csv(csv_path: str | Path, metric_rows: list[dict], has_mr: bool) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "id",
        "extractable_code", "passes_tests", "source_passed",
        "tests_passed", "tests_failed", "tests_total",
        "test_run_time",
        "source_ncloc", "source_cc",
        "source_mean_distance", "source_std_distance", "source_pairs",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "thinking_tokens", "token_efficiency",
        "llm_call_time",
    ]
    mr_extra_fields = [
        "followup_extractable", "followup_passes_tests", "passes_same_tests", "followup_passed",
        "followup_tests_passed", "followup_tests_failed", "followup_tests_total",
        "followup_test_run_time",
        "followup_ncloc", "followup_cc",
        "followup_mean_distance", "followup_std_distance", "followup_pairs",
        "followup_prompt_tokens", "followup_completion_tokens", "followup_total_tokens",
        "followup_thinking_tokens", "followup_token_efficiency",
        "followup_llm_call_time",
        "code_sim", "mr_holds",
    ]
    fieldnames = base_fields + (mr_extra_fields if has_mr else [])

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            src_var = row.get("source_variability") or {}
            flw_var = row.get("followup_variability") or {}
            output = {
                "id": row.get("id"),
                "extractable_code": row.get("source_extractable"),
                "passes_tests": row.get("source_passes_tests"),
                "source_passed": row.get("source_passed"),
                "tests_passed": row.get("tests_passed"),
                "tests_failed": row.get("tests_failed"),
                "tests_total": row.get("tests_total"),
                "test_run_time": row.get("source_test_run_time"),
                "source_ncloc": row.get("source_ncloc"),
                "source_cc": row.get("source_cc"),
                "source_mean_distance": src_var.get("mean_distance"),
                "source_std_distance": src_var.get("std_distance"),
                "source_pairs": src_var.get("n_pairs"),
                "prompt_tokens": row.get("source_prompt_tokens"),
                "completion_tokens": row.get("source_completion_tokens"),
                "total_tokens": row.get("source_total_tokens"),
                "thinking_tokens": row.get("source_thinking_tokens"),
                "token_efficiency": row.get("source_token_efficiency"),
                "llm_call_time": row.get("source_llm_call_time"),
                # MR fields
                "followup_extractable": row.get("followup_extractable"),
                "followup_passes_tests": row.get("followup_passes_tests"),
                "passes_same_tests": row.get("passes_same_tests"),
                "followup_passed": row.get("followup_passed"),
                "followup_tests_passed": row.get("followup_tests_passed"),
                "followup_tests_failed": row.get("followup_tests_failed"),
                "followup_tests_total": row.get("followup_tests_total"),
                "followup_test_run_time": row.get("followup_test_run_time"),
                "followup_ncloc": row.get("followup_ncloc"),
                "followup_cc": row.get("followup_cc"),
                "followup_mean_distance": flw_var.get("mean_distance"),
                "followup_std_distance": flw_var.get("std_distance"),
                "followup_pairs": flw_var.get("n_pairs"),
                "followup_prompt_tokens": row.get("followup_prompt_tokens"),
                "followup_completion_tokens": row.get("followup_completion_tokens"),
                "followup_total_tokens": row.get("followup_total_tokens"),
                "followup_thinking_tokens": row.get("followup_thinking_tokens"),
                "followup_token_efficiency": row.get("followup_token_efficiency"),
                "followup_llm_call_time": row.get("followup_llm_call_time"),
                "code_sim": row.get("code_sim"),
                "mr_holds": row.get("mr_holds"),
            }
            writer.writerow({field: output.get(field) for field in fieldnames})
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
            all_passed, tp, tf, tt, code, test_results, run_time = test_response(
                primary_response, entry, func_name, debug, label=label
            )
            if all_passed is None:
                skipped += 1
                continue
            if all_passed:
                passed += 1
            else:
                failed += 1

            metrics = compute_metrics(code)
            sample_key = "followup_response_samples" if label == "followup" else "response_samples"
            primary_key = "followup_response" if label == "followup" else "response"
            variability = compute_pairwise_code_variability(
                _code_samples(_response_samples(entry, sample_key, primary_key))
            )
            metric_rows.append({
                "id": qid,
                "question_title": entry.get("question_title"),
                "difficulty": entry.get("difficulty"),
                "platform": entry.get("platform"),
                "source_extractable": compute_extractable_code(primary_response),
                "source_passes_tests": compute_passes_tests(test_results),
                "source_passed": all_passed,
                "tests_passed": tp, "tests_failed": tf, "tests_total": tt,
                "source_test_run_time": round(run_time, 4),
                "source_ncloc": metrics["ncloc"],
                "source_cc": metrics["cyclomatic_complexity"],
                "source_variability": variability,
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
        src_passed, src_tp, src_tf, src_tt, src_code, src_test_results, src_run_time = test_response(
            src_response, entry, func_name, debug, label="source"
        )

        if src_passed is None:
            skipped += 1
            continue
        if src_passed:
            passed += 1
        else:
            failed += 1

        src_metrics = compute_metrics(src_code)
        src_var = compute_pairwise_code_variability(
            _code_samples(_response_samples(entry, "response_samples", "response"))
        )

        row: dict = {
            "id": qid,
            "question_title": entry.get("question_title"),
            "difficulty": entry.get("difficulty"),
            "platform": entry.get("platform"),

            # Source test metrics
            "source_extractable": compute_extractable_code(src_response),
            "source_passes_tests": compute_passes_tests(src_test_results),
            "source_passed": src_passed,
            "tests_passed": src_tp, "tests_failed": src_tf, "tests_total": src_tt,
            "source_test_run_time": round(src_run_time, 4),

            # Source code metrics
            "source_ncloc": src_metrics["ncloc"],
            "source_cc": src_metrics["cyclomatic_complexity"],
            "source_variability": src_var,

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
            flw_passed, flw_tp, flw_tf, flw_tt, flw_code, flw_test_results, flw_run_time = test_response(
                flw_response, entry, func_name, debug, label="followup"
            )

            if flw_passed is not None:
                holds = src_passed == flw_passed
                verdict = "HOLDS" if holds else "VIOLATED"
                print(f"  MR [{entry['relation']}]: {verdict}")
                if holds:
                    mr_holds_count += 1
                else:
                    mr_violated_count += 1
                row["mr_holds"] = holds
                row["followup_passed"] = flw_passed
                row["followup_tests_passed"] = flw_tp
                row["followup_tests_failed"] = flw_tf
                row["followup_tests_total"] = flw_tt
                row["followup_test_run_time"] = round(flw_run_time, 4)

            row["followup_extractable"] = compute_extractable_code(flw_response)
            row["followup_passes_tests"] = compute_passes_tests(flw_test_results)
            row["passes_same_tests"] = compute_passes_same_tests(src_test_results, flw_test_results)

            flw_metrics = compute_metrics(flw_code)
            row["followup_ncloc"] = flw_metrics["ncloc"]
            row["followup_cc"] = flw_metrics["cyclomatic_complexity"]
            row["followup_variability"] = compute_pairwise_code_variability(
                _code_samples(_response_samples(entry, "followup_response_samples", "followup_response"))
            )
            row["followup_prompt_tokens"] = flw_tokens["prompt_tokens"]
            row["followup_completion_tokens"] = flw_tokens["completion_tokens"]
            row["followup_total_tokens"] = flw_tokens["total_tokens"]
            row["followup_thinking_tokens"] = flw_tokens["thinking_tokens"]
            row["followup_token_efficiency"] = flw_token_efficiency
            row["followup_llm_call_time"] = flw_llm_call_time

            if src_code and flw_code:
                row["code_sim"] = compute_code_similarity(src_code, flw_code)

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
                cc_str = f"{_fmt(r['source_cc'], 1)}/{_fmt(r['followup_cc'], 1) if r['followup_cc'] is not None else 'N/A'}"
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
                eff = _fmt(r.get("source_token_efficiency"), 3) if r.get("source_token_efficiency") is not None else "N/A"
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
                    f"{_fmt(r.get('code_sim'), 4):>7} {_fmt(src_dist, 4):>7} {_fmt(flw_dist, 4):>7} "
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
                eff = _fmt(r.get("source_token_efficiency"), 3) if r.get("source_token_efficiency") is not None else "N/A"
                llm_t = f"{r['source_llm_call_time']:.1f}s" if r.get("source_llm_call_time") is not None else "N/A"
                test_t = f"{r['source_test_run_time']:.2f}s" if r.get("source_test_run_time") is not None else "N/A"
                print(
                    f"  {r['id']:<15} {tests_str:>8} {ext:>7} {r['source_ncloc']:>6} {r['source_cc']:>6} "
                    f"{_fmt(src_var.get('mean_distance'), 4):>8} {_fmt(src_var.get('std_distance'), 4):>7} "
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


class _Tee:
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
        stream = _Tee(sys.stdout, f) if args.stdout else f
        with redirect_stdout(stream), redirect_stderr(stream):
            run_tests(
                args.results, args.n,
                debug=args.debug,
                followup_only=args.followup_only,
                csv_path=args.csv,
                write_outputs=args.write_outputs,
            )
