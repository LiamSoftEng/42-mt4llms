"""
Run all test cases against model responses.
If the results file contains followup responses (from --mr), tests both
and computes MR verdict.

Usage:
    python runtests.py --results results/google_gemma-2-2b-it/base.json
    python runtests.py --results results/google_gemma-2-2b-it/base.json --n 5
    python runtests.py --results results/google_gemma-2-2b-it/base.json --n 1 --debug
    python runtests.py --results results/google_gemma-2-2b-it/base.json --stdout
    python runtests.py --results results/google_gemma-2-2b-it/base.json --out results/google_gemma-2-2b-it/base.tests.log
    python runtests.py --results results/google_gemma-2-2b-it/mr-04-persona_role_prompting.json --followup-only
"""

import json
import argparse
import csv
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from evaluation.executor import extract_code, get_all_tests, get_func_name, run_test
from evaluation.metrics import compute_metrics, compute_code_similarity, compute_pairwise_code_variability


def _response_samples(entry: dict, sample_key: str, primary_key: str) -> list[str]:
    samples = entry.get(sample_key)
    if isinstance(samples, list) and samples:
        return [sample for sample in samples if isinstance(sample, str)]
    primary = entry.get(primary_key, "")
    return [primary] if isinstance(primary, str) and primary else []


def _code_samples(responses: list[str]) -> list[str | None]:
    return [extract_code(response) for response in responses]


def _format_float(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "N/A"


def _format_delta(value) -> str:
    return f"{value:+}" if isinstance(value, (int, float)) else value


def _variation_greater(source_mean_dist, followup_mean_dist) -> str:
    if isinstance(source_mean_dist, (int, float)) and isinstance(followup_mean_dist, (int, float)):
        return "YES" if followup_mean_dist > source_mean_dist else "NO"
    return "N/A"


def _write_metrics_csv(csv_path: str | Path, metric_rows: list[dict], has_mr: bool) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source_passed",
        "followup_passed",
        "mr_holds",
        "source_ncloc",
        "source_cc",
        "followup_ncloc",
        "followup_cc",
        "code_sim",
        "source_mean_distance",
        "source_std_distance",
        "source_pairs",
        "followup_mean_distance",
        "followup_std_distance",
        "followup_pairs",
        "variation_greater_than_source",
    ]
    if not has_mr:
        fieldnames = [
            "id",
            "source_passed",
            "source_ncloc",
            "source_cc",
            "source_mean_distance",
            "source_std_distance",
            "source_pairs",
        ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            source_var = row.get("source_variability") or {}
            followup_var = row.get("followup_variability") or {}
            source_mean_dist = source_var.get("mean_distance")
            followup_mean_dist = followup_var.get("mean_distance")
            output = {
                "id": row.get("id"),
                "source_passed": row.get("source_passed"),
                "followup_passed": row.get("followup_passed"),
                "mr_holds": row.get("mr_holds"),
                "source_ncloc": row.get("source_ncloc"),
                "source_cc": row.get("source_cc"),
                "followup_ncloc": row.get("followup_ncloc"),
                "followup_cc": row.get("followup_cc"),
                "code_sim": row.get("code_sim"),
                "source_mean_distance": source_mean_dist,
                "source_std_distance": source_var.get("std_distance"),
                "source_pairs": source_var.get("n_pairs"),
                "followup_mean_distance": followup_mean_dist,
                "followup_std_distance": followup_var.get("std_distance"),
                "followup_pairs": followup_var.get("n_pairs"),
                "variation_greater_than_source": _variation_greater(source_mean_dist, followup_mean_dist),
            }
            writer.writerow({field: output.get(field) for field in fieldnames})

    print(f"\n  Wrote metrics CSV -> {path}")


def test_response(response, entry, func_name, debug, label="source"):
    code = extract_code(response)
    if not code:
        print(f"  [{label}] SKIP — no code block found")
        if debug:
            print(f"    Response: {response[:200]}")
        return None, 0, None

    tests = get_all_tests(entry)
    if not tests:
        print(f"  [{label}] SKIP — no test cases")
        return None, 0, None

    all_passed = True
    for i, test in enumerate(tests):
        testtype = test.get("testtype", "stdin")
        passed, actual, error = run_test(code, test, func_name, debug=debug)
        if passed:
            print(f"  [{label}] Test {i+1} [{testtype}]: PASS")
        else:
            all_passed = False
            print(f"  [{label}] Test {i+1} [{testtype}]: FAIL")
            if error:
                print(f"    Error:    {error[:200]}")
            else:
                print(f"    Expected: {test.get('output', '')[:80].strip()}")
                print(f"    Got:      {actual[:80].strip()}")

    return all_passed, len(tests), code


def run_tests(
    results_path: str,
    n: int = None,
    debug: bool = False,
    followup_only: bool = False,
    csv_path: str | None = None,
):
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if n:
        results = results[:n]

    total = passed = failed = skipped = 0
    mr_holds = mr_violated = 0
    has_mr = any(r.get("relation") for r in results)
    if followup_only:
        has_mr = False
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

        if followup_only:
            primary_response = entry.get("followup_response") or entry.get("response", "")
            label = "followup" if entry.get("followup_response") else "source"
            source_passed, n_tests, source_code = test_response(primary_response, entry, func_name, debug, label=label)

            if source_passed is None:
                skipped += 1
                continue

            if source_passed:
                passed += 1
            else:
                failed += 1

            source_metrics = compute_metrics(source_code)
            sample_key = "followup_response_samples" if label == "followup" else "response_samples"
            primary_key = "followup_response" if label == "followup" else "response"
            source_variability = compute_pairwise_code_variability(
                _code_samples(_response_samples(entry, sample_key, primary_key))
            )
            metric_rows.append(
                {
                    "id": qid,
                    "source_passed": source_passed,
                    "source_ncloc": source_metrics["ncloc"],
                    "source_cc": source_metrics["cyclomatic_complexity"],
                    "source_variability": source_variability,
                }
            )
            continue

        # Test source response
        source_passed, n_tests, source_code = test_response(entry["response"], entry, func_name, debug, label="source")

        if source_passed is None:
            skipped += 1
            continue

        if source_passed:
            passed += 1
        else:
            failed += 1

        source_metrics = compute_metrics(source_code)
        source_variability = compute_pairwise_code_variability(
            _code_samples(_response_samples(entry, "response_samples", "response"))
        )
        row = {
            "id": qid,
            "source_passed": source_passed,
            "followup_passed": None,
            "source_ncloc": source_metrics["ncloc"],
            "source_cc": source_metrics["cyclomatic_complexity"],
            "source_variability": source_variability,
            "followup_ncloc": None,
            "followup_cc": None,
            "followup_variability": None,
            "code_sim": None,
            "mr_holds": None,
        }

        # Test followup response if MR was applied
        if entry.get("relation") and entry.get("followup_response"):
            followup_passed, _, followup_code = test_response(
                entry["followup_response"], entry, func_name, debug, label="followup"
            )

            if followup_passed is not None:
                holds = source_passed == followup_passed
                verdict = "HOLDS" if holds else "VIOLATED"
                print(f"  MR [{entry['relation']}]: {verdict}")
                if holds:
                    mr_holds += 1
                else:
                    mr_violated += 1
                row["mr_holds"] = holds
                row["followup_passed"] = followup_passed

            followup_metrics = compute_metrics(followup_code)
            row["followup_ncloc"] = followup_metrics["ncloc"]
            row["followup_cc"] = followup_metrics["cyclomatic_complexity"]
            row["followup_variability"] = compute_pairwise_code_variability(
                _code_samples(_response_samples(entry, "followup_response_samples", "followup_response"))
            )
            if source_code and followup_code:
                row["code_sim"] = compute_code_similarity(source_code, followup_code)

        metric_rows.append(row)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Results: {Path(results_path).name}")
    print(f"  Model:   {results[0]['model_name'] if results else 'unknown'}")
    print(f"  PASSED:  {passed}/{total}")
    print(f"  FAILED:  {failed}/{total}")
    print(f"  SKIPPED: {skipped}/{total}")
    if has_mr:
        mr_total = mr_holds + mr_violated
        print(f"  MR [{results[0].get('relation')}] HOLDS:    {mr_holds}/{mr_total}")
        print(f"  MR [{results[0].get('relation')}] VIOLATED: {mr_violated}/{mr_total}")

    if metric_rows:
        print()
        if has_mr:
            print(
                f"  {'ID':<15} {'Src NCLOC':>10} {'Src CC':>8} {'Flw NCLOC':>10} "
                f"{'Flw CC':>8} {'CodeSim':>8} {'SrcDist':>8} {'SrcStd':>8} "
                f"{'FlwDist':>8} {'FlwStd':>8} {'dNCLOC':>8} {'dCC':>8} {'Var>Src':>8} {'MR':>8}"
            )
            print(
                f"  {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
                f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}"
            )
            for r in metric_rows:
                delta_ncloc = (
                    r["followup_ncloc"] - r["source_ncloc"]
                    if r["followup_ncloc"] is not None and r["source_ncloc"] >= 0 and r["followup_ncloc"] >= 0
                    else "N/A"
                )
                delta_cc = (
                    round(r["followup_cc"] - r["source_cc"], 2)
                    if r["followup_cc"] is not None and r["source_cc"] >= 0 and r["followup_cc"] >= 0
                    else "N/A"
                )
                mr = ("HOLDS" if r["mr_holds"] else "VIOLATED") if r["mr_holds"] is not None else "N/A"
                flw_ncloc = r["followup_ncloc"] if r["followup_ncloc"] is not None else "N/A"
                flw_cc = r["followup_cc"] if r["followup_cc"] is not None else "N/A"
                source_var = r.get("source_variability") or {}
                followup_var = r.get("followup_variability") or {}
                source_mean_dist = source_var.get("mean_distance")
                followup_mean_dist = followup_var.get("mean_distance")
                variation_greater = _variation_greater(source_mean_dist, followup_mean_dist)
                print(
                    f"  {r['id']:<15} {r['source_ncloc']:>10} {r['source_cc']:>8} "
                    f"{str(flw_ncloc):>10} {str(flw_cc):>8} {_format_float(r['code_sim']):>8} "
                    f"{_format_float(source_mean_dist):>8} {_format_float(source_var.get('std_distance')):>8} "
                    f"{_format_float(followup_mean_dist):>8} {_format_float(followup_var.get('std_distance')):>8} "
                    f"{_format_delta(delta_ncloc):>8} {_format_delta(delta_cc):>8} {variation_greater:>8} {mr:>8}"
                )
        else:
            print(f"  {'ID':<15} {'NCLOC':>10} {'CC':>8} {'MeanDist':>8} {'StdDist':>8} {'Pairs':>8}")
            print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            for r in metric_rows:
                source_var = r.get("source_variability") or {}
                print(
                    f"  {r['id']:<15} {r['source_ncloc']:>10} {r['source_cc']:>8} "
                    f"{_format_float(source_var.get('mean_distance')):>8} "
                    f"{_format_float(source_var.get('std_distance')):>8} {source_var.get('n_pairs', 0):>8}"
                )
        if csv_path:
            _write_metrics_csv(csv_path, metric_rows, has_mr)
    print(f"{'='*60}")


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
    parser.add_argument(
        "--followup-only",
        action="store_true",
        help="Test followup responses only (for MR-only runs)",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(args.results).with_suffix(".tests.log")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        stream = _Tee(sys.stdout, f) if args.stdout else f
        with redirect_stdout(stream), redirect_stderr(stream):
            run_tests(args.results, args.n, debug=args.debug, followup_only=args.followup_only, csv_path=args.csv)
