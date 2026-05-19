"""
Run all test cases against model responses.
If the results file contains followup responses (from --mr), tests both
and computes MR verdict.

Usage:
    python runtests.py --results results/google_gemma-2-2b-it.json
    python runtests.py --results results/google_gemma-2-2b-it.json --n 5
    python runtests.py --results results/google_gemma-2-2b-it.json --n 1 --debug
    python runtests.py --results results/google_gemma-2-2b-it.json --stdout
    python runtests.py --results results/google_gemma-2-2b-it.json --out results/google_gemma-2-2b-it.tests.log
"""

import json
import argparse
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from evaluation.executor import extract_code, get_all_tests, get_func_name, run_test


def test_response(response, entry, func_name, debug, label="source"):
    code = extract_code(response)
    if not code:
        print(f"  [{label}] SKIP — no code block found")
        if debug:
            print(f"    Response: {response[:200]}")
        return None, 0

    tests = get_all_tests(entry)
    if not tests:
        print(f"  [{label}] SKIP — no test cases")
        return None, 0

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

    return all_passed, len(tests)


def run_tests(results_path: str, n: int = None, debug: bool = False):
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if n:
        results = results[:n]

    total = passed = failed = skipped = 0
    mr_holds = mr_violated = 0
    has_mr = any(r.get("relation") for r in results)

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

        # Test source response
        source_passed, n_tests = test_response(entry["response"], entry, func_name, debug, label="source")

        if source_passed is None:
            skipped += 1
            continue

        if source_passed:
            passed += 1
        else:
            failed += 1

        # Test followup response if MR was applied
        if entry.get("relation") and entry.get("followup_response"):
            followup_passed, _ = test_response(entry["followup_response"], entry, func_name, debug, label="followup")

            if followup_passed is not None:
                holds = source_passed == followup_passed
                verdict = "HOLDS" if holds else "VIOLATED"
                print(f"  MR [{entry['relation']}]: {verdict}")
                if holds:
                    mr_holds += 1
                else:
                    mr_violated += 1

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
    parser.add_argument("--stdout", action="store_true", help="Also print output to the terminal")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(args.results).with_suffix(".tests.log")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        stream = _Tee(sys.stdout, f) if args.stdout else f
        with redirect_stdout(stream), redirect_stderr(stream):
            run_tests(args.results, args.n, debug=args.debug)