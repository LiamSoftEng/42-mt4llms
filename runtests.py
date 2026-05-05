"""
Run all test cases (public, stdin + functional) against model responses.

Usage:
    python run_tests.py --results results/google_gemma-2-2b-it.json
    python run_tests.py --results results/google_gemma-2-2b-it.json --n 5
    python run_tests.py --results results/google_gemma-2-2b-it.json --n 1 --debug
"""

import json
import argparse
from pathlib import Path

from evaluation.executor import extract_code, get_all_tests, get_func_name, run_test


def run_tests(results_path: str, n: int = None, debug: bool = False):
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if n:
        results = results[:n]

    total_problems = 0
    passed_problems = 0
    failed_problems = 0
    skipped_problems = 0

    for entry in results:
        qid = entry["question_id"]
        title = entry["question_title"]
        difficulty = entry["difficulty"]
        platform = entry["platform"]

        print(f"\n{'='*60}")
        print(f"  {qid} — {title} ({difficulty} / {platform})")
        print(f"{'='*60}")

        total_problems += 1

        if entry.get("error"):
            print(f"  [SKIP] LLM error: {entry['error']}")
            skipped_problems += 1
            continue

        code = extract_code(entry["response"])
        if not code:
            print("  [SKIP] No Python code block found in response")
            if debug:
                print(f"  Response was: {entry['response'][:300]}")
            skipped_problems += 1
            continue

        if debug:
            print(f"\n  Extracted code:\n{code}\n")

        tests = get_all_tests(entry)
        if not tests:
            print("  [SKIP] No test cases available")
            skipped_problems += 1
            continue

        func_name = get_func_name(entry)

        problem_passed = True
        for i, test in enumerate(tests):
            testtype = test.get("testtype", "stdin")
            passed, actual, error = run_test(code, test, func_name, debug=debug)

            if passed:
                print(f"  Test {i+1} [{testtype}]: PASS")
            else:
                problem_passed = False
                print(f"  Test {i+1} [{testtype}]: FAIL")
                if error:
                    print(f"    Error:    {error[:200]}")
                else:
                    print(f"    Expected: {test.get('output', '')[:80].strip()}")
                    print(f"    Got:      {actual[:80].strip()}")

        if problem_passed:
            passed_problems += 1
        else:
            failed_problems += 1

    print(f"\n{'='*60}")
    print(f"  Results: {Path(results_path).name}")
    print(f"  Model:   {results[0]['model_name'] if results else 'unknown'}")
    print(f"  PASSED:  {passed_problems}/{total_problems}")
    print(f"  FAILED:  {failed_problems}/{total_problems}")
    print(f"  SKIPPED: {skipped_problems}/{total_problems}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test cases against model responses")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N problems")
    parser.add_argument("--debug", action="store_true", help="Print extracted code and generated scripts")
    args = parser.parse_args()

    run_tests(args.results, args.n, debug=args.debug)