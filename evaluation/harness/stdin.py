"""
Harness for stdin/stdout style problems (Codeforces).
The model's code is run as a subprocess with test input piped to stdin,
and stdout is compared against the expected output.
"""

import subprocess
import sys
import tempfile
import os


def run(code: str, test_input: str, timeout: int = 100) -> tuple[str, str]:
    """
    Execute code with the given stdin input.

    Returns:
        The subprocess's ``(stdout, stderr)`` text.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT"
    finally:
        os.unlink(tmp_path)


def check(actual: str, expected: str) -> bool:
    """Compare outputs line by line, ignoring trailing whitespace."""
    actual_lines = [l.strip() for l in actual.strip().splitlines()]
    expected_lines = [l.strip() for l in expected.strip().splitlines()]
    return actual_lines == expected_lines
