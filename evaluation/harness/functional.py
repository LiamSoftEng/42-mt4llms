"""
Harness for functional style problems (LeetCode).
The model returns a Solution class with a method. We construct a self-contained
script that calls the function with test inputs and prints the result.
"""

import subprocess
import sys
import tempfile
import os
import json


RUNNER_TEMPLATE = """from typing import List, Dict, Tuple, Optional, Set
{code}

import json

sol = Solution()
result = sol.{func_name}({args})
print(json.dumps(result, separators=(',', ':')))
"""


def parse_args(test_input: str) -> str:
    """
    LeetCode inputs are newline-separated JSON values, one per argument.
    e.g. '"[1,2,3]"\n"4"' -> '[1,2,3], 4'
    Parse each line as JSON so Python receives the correct types.
    """
    args = []
    for line in test_input.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            # Each line is a JSON-encoded value
            parsed = json.loads(line)
            args.append(repr(parsed))
        except json.JSONDecodeError:
            # Fall back to using the raw line
            args.append(line)
    return ", ".join(args)


def run(code: str, func_name: str, test_input: str, timeout: int = 100, debug: bool = False) -> tuple[str, str]:
    """
    Execute a functional-style solution.

    Returns:
        (stdout, stderr)
    """
    args = parse_args(test_input)
    script = RUNNER_TEMPLATE.format(code=code, func_name=func_name, args=args)

    if debug:
        print("\n--- GENERATED SCRIPT ---")
        print(script)
        print("--- END SCRIPT ---\n")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT"
    finally:
        os.unlink(tmp_path)


def check(actual: str, expected: str) -> bool:
    """
    Compare actual vs expected output.
    Parses both as JSON for structural comparison, falls back to string compare.
    Also handles LeetCode's 'true'/'false' vs Python's 'True'/'False'.
    """
    # Normalise Python booleans to JSON booleans for comparison
    actual = actual.strip().replace("True", "true").replace("False", "false")
    expected = expected.strip().replace("True", "true").replace("False", "false")

    try:
        actual_val = json.loads(actual)
        expected_val = json.loads(expected)
        if isinstance(actual_val, list) and isinstance(expected_val, list):
            if sorted(str(x) for x in actual_val) == sorted(str(x) for x in expected_val):
                return True
        return actual_val == expected_val
    except (json.JSONDecodeError, TypeError):
        return actual == expected