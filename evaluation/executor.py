"""
Runs public test cases against model responses.
"""
import json
import re
import time


def extract_code(response: str) -> str | None:
    """
    Pull the Python code block out of the model response.

    Tries, in order:
      1. ```python ... ``` fenced block (preferred, most explicit)
      2. ``` ... ``` fenced block with no language tag
      3. Raw response treated as code, if it has no fence markers at all
         (some models emit bare code with no markdown fencing)
    Returns None only if the response is empty or contains fence markers
    that don't resolve into a matched block (i.e. genuinely unparseable).
    """
    if not response:
        return None

    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    stripped = response.strip()
    if stripped and "```" not in stripped:
        return stripped

    return None


def get_all_tests(entry: dict) -> list[dict]:
    """Decode private test cases from the raw base64 string at test time."""
    from data.decoder import decode_private_test_cases
    raw = entry.get("private_test_cases", "")
    return decode_private_test_cases(raw)


def get_func_name(entry: dict) -> str | None:
    """Extract the function name from metadata (used for functional problems)."""
    raw = entry.get("metadata", "{}")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
            if isinstance(raw, str):
                raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    return raw.get("func_name", None)


def run_test(
    code: str,
    test: dict,
    func_name: str | None,
    debug: bool = False,
) -> tuple[bool, str, str, float]:
    """
    Run a single test case against the extracted code.
    Returns (passed, actual_output, error, elapsed_seconds).
    elapsed_seconds is the wall-clock time spent inside the harness.
    """
    testtype = test.get("testtype", "stdin")
    inp = test.get("input", "")
    expected = test.get("output", "")

    t0 = time.perf_counter()

    if testtype == "stdin":
        from evaluation.harness.stdin import run, check
        actual, error = run(code, inp)
        passed = check(actual, expected) if not error else False

    elif testtype == "functional":
        if not func_name:
            elapsed = time.perf_counter() - t0
            return False, "", "No func_name in metadata", elapsed
        from evaluation.harness.functional import run, check
        actual, error = run(code, func_name, inp, debug=debug)
        passed = check(actual, expected) if not error else False

    else:
        elapsed = time.perf_counter() - t0
        return False, "", f"Unknown testtype: {testtype}", elapsed

    elapsed = time.perf_counter() - t0
    return passed, actual, error, elapsed