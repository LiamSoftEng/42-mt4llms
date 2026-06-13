"""
Run all test cases against model responses.
If the results file contains followup responses (from --mr), tests both
and computes MR verdict.

Usage:
    python runtests.py --results results/grok/base.json
    python runtests.py --results results/grok/base.json --n 5
    python runtests.py --results results/grok/base.json --stdout
    python runtests.py --results results/grok/mr-05-context_extraneous_narrative.json --followup-only
    python runtests.py --results results/grok/mr-05-context_extraneous_narrative.json --csv results/grok/metrics.csv
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
    fieldnames = [
        "id", "source_passed", "followup_passed", "mr_holds",
        "source_ncloc", "source_cc", "followup_ncloc", "followup_cc",
        "code_sim", "source_mean_distance", "source_std_distance", "source_pairs",
        "followup_mean_distance", "followup_std_distance", "followup_pairs",
        "variation_greater_than_source",
    ]
    if not has_mr:
        fieldnames = ["id", "source_passed", "source_ncloc", "source_cc",
                      "source_mean_distance", "source_std_distance", "source_pairs"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            src_var = row.get("source_variability") or {}
            flw_var = row.get("followup_variability") or {}
            src_dist = src_var.get("mean_distance")
            flw_dist = flw_var.get("mean_distance")
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
                "source_mean_distance": src_dist,
                "source_std_distance": src_var.get("std_distance"),
                "source_pairs": src_var.get("n_pairs"),
                "followup_mean_distance": flw_dist,
                "followup_std_distance": flw_var.get("std_distance"),
                "followup_pairs": flw_var.get("n_pairs"),
                "variation_greater_than_source": _variation_greater(src_dist, flw_dist),
            }
            writer.writerow({field: output.get(field) for field in fieldnames})
    print(f"\n  Wrote metrics CSV -> {path}")


def test_response(response, entry, func_name, debug, label="source"):
    code = extract_code(response)
    if not code:
        print(f"  [{label}] SKIP — no code block found")
        if debug:
            print(f"    Response: {response[:200]}")
        return None, 0, 0, 0, None

    tests = get_all_tests(entry)
    if not tests:
        print(f"  [{label}] SKIP — no test cases")
        return None, 0, 0, 0, None

    tests_passed = 0
    tests_failed = 0
    for i, test in enumerate(tests):
        testtype = test.get("testtype", "stdin")
        passed, actual, error = run_test(code, test, func_name, debug=debug)
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
    print(f"  [{label}] {tests_passed}/{total} passed")
    return all_passed, tests_passed, tests_failed, total, code


def run_tests(
    results_path: str,
    n: int = None,
    debug: bool = False,
    followup_only: bool = False,
    csv_path: str | None = None,
    write_outputs: bool = False,
):
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if n:
        results = results[:n]

    total = passed = failed = skipped = 0
    mr_holds = mr_violated = 0
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

        if followup_only:
            primary_response = entry.get("followup_response") or entry.get("response", "")
            label = "followup" if entry.get("followup_response") else "source"
            all_passed, tp, tf, tt, code = test_response(primary_response, entry, func_name, debug, label=label)

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
                "source_passed": all_passed,
                "tests_passed": tp, "tests_failed": tf, "tests_total": tt,
                "source_ncloc": metrics["ncloc"],
                "source_cc": metrics["cyclomatic_complexity"],
                "source_variability": variability,
            })
            continue

        # Source
        src_passed, src_tp, src_tf, src_tt, src_code = test_response(
            entry["response"], entry, func_name, debug, label="source"
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

        row = {
            "id": qid,
            "question_title": entry.get("question_title"),
            "difficulty": entry.get("difficulty"),
            "platform": entry.get("platform"),
            "source_passed": src_passed,
            "tests_passed": src_tp, "tests_failed": src_tf, "tests_total": src_tt,
            "source_ncloc": src_metrics["ncloc"],
            "source_cc": src_metrics["cyclomatic_complexity"],
            "source_variability": src_var,
            "followup_passed": None,
            "followup_tests_passed": None, "followup_tests_failed": None, "followup_tests_total": None,
            "followup_ncloc": None, "followup_cc": None,
            "followup_variability": None,
            "code_sim": None, "mr_holds": None,
        }

        # Followup
        if entry.get("relation") and entry.get("followup_response"):
            flw_passed, flw_tp, flw_tf, flw_tt, flw_code = test_response(
                entry["followup_response"], entry, func_name, debug, label="followup"
            )

            if flw_passed is not None:
                holds = src_passed == flw_passed
                verdict = "HOLDS" if holds else "VIOLATED"
                print(f"  MR [{entry['relation']}]: {verdict}")
                if holds:
                    mr_holds += 1
                else:
                    mr_violated += 1
                row["mr_holds"] = holds
                row["followup_passed"] = flw_passed
                row["followup_tests_passed"] = flw_tp
                row["followup_tests_failed"] = flw_tf
                row["followup_tests_total"] = flw_tt

            flw_metrics = compute_metrics(flw_code)
            row["followup_ncloc"] = flw_metrics["ncloc"]
            row["followup_cc"] = flw_metrics["cyclomatic_complexity"]
            row["followup_variability"] = compute_pairwise_code_variability(
                _code_samples(_response_samples(entry, "followup_response_samples", "followup_response"))
            )
            if src_code and flw_code:
                row["code_sim"] = compute_code_similarity(src_code, flw_code)

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
                f"  {'ID':<15} {'Src':>6} {'Flw':>6} {'NCLOC':>6} {'dNCL':>6} "
                f"{'CC':>6} {'dCC':>6} {'Sim':>7} {'SrcDst':>7} {'FlwDst':>7} {'Var>':>5} {'MR':>8}"
            )
            print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*8}")
            for r in metric_rows:
                src_var = r.get("source_variability") or {}
                flw_var = r.get("followup_variability") or {}
                src_dist = src_var.get("mean_distance")
                flw_dist = flw_var.get("mean_distance")
                ncloc_str = f"{r['source_ncloc']}/{r['followup_ncloc'] if r['followup_ncloc'] is not None else 'N/A'}"
                cc_str = f"{_fmt(r['source_cc'], 1)}/{_fmt(r['followup_cc'], 1) if r['followup_cc'] is not None else 'N/A'}"
                mr = ("HOLDS" if r["mr_holds"] else "VIOLATED") if r["mr_holds"] is not None else "N/A"
                src_res = f"{r['tests_passed']}/{r['tests_total']}" if r.get("tests_total") else "N/A"
                flw_res = f"{r.get('followup_tests_passed', '?')}/{r.get('followup_tests_total', '?')}" if r.get("followup_tests_total") else "N/A"
                print(
                    f"  {r['id']:<15} {src_res:>6} {flw_res:>6} {ncloc_str:>6} "
                    f"{_delta(r['followup_ncloc'] - r['source_ncloc'] if r['followup_ncloc'] is not None and r['source_ncloc'] >= 0 and r['followup_ncloc'] >= 0 else 'N/A'):>6} "
                    f"{cc_str:>6} "
                    f"{_delta(round(r['followup_cc'] - r['source_cc'], 2) if r['followup_cc'] is not None and r['source_cc'] >= 0 and r['followup_cc'] >= 0 else 'N/A'):>6} "
                    f"{_fmt(r.get('code_sim'), 4):>7} {_fmt(src_dist, 4):>7} {_fmt(flw_dist, 4):>7} "
                    f"{_variation_greater(src_dist, flw_dist):>5} {mr:>8}"
                )
        else:
            print(f"  {'ID':<15} {'Tests':>8} {'NCLOC':>6} {'CC':>6} {'MeanDist':>9} {'StdDist':>8} {'Pairs':>6}")
            print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*6} {'-'*9} {'-'*8} {'-'*6}")
            for r in metric_rows:
                src_var = r.get("source_variability") or {}
                tests_str = f"{r.get('tests_passed', '?')}/{r.get('tests_total', '?')}"
                print(
                    f"  {r['id']:<15} {tests_str:>8} {r['source_ncloc']:>6} {r['source_cc']:>6} "
                    f"{_fmt(src_var.get('mean_distance'), 4):>9} {_fmt(src_var.get('std_distance'), 4):>8} "
                    f"{src_var.get('n_pairs', 0):>6}"
                )

    if csv_path:
        _write_metrics_csv(csv_path, metric_rows, has_mr)

    if write_outputs and results:
        from evaluation.results_writer import write_all_outputs
        from config.settings import RESULTS_DIR
        model_name = results[0]["model_name"]
        relation_name = results[0].get("relation") or None
        # Derive relation_id from filename e.g. mr-05-name.json -> 5
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