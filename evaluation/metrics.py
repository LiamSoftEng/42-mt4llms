"""
Code metrics computed on extracted code blocks using radon.
"""

from radon.raw import analyze
from radon.complexity import cc_visit


def compute_ncloc(code: str) -> int:
    """
    Non-comment lines of code - blank lines and comment lines excluded.
    Uses radon's lloc (logical lines of code).
    Returns -1 if code cannot be parsed.
    """
    try:
        result = analyze(code)
        return result.lloc
    except Exception:
        return -1


def compute_cyclomatic_complexity(code: str) -> float:
    """
    Average cyclomatic complexity across all functions/methods in the code.
    Returns -1 if code cannot be parsed or contains no functions.
    """
    try:
        blocks = cc_visit(code)
        if not blocks:
            return -1
        return round(sum(b.complexity for b in blocks) / len(blocks), 2)
    except Exception:
        return -1


def compute_metrics(code: str | None) -> dict:
    """
    Compute all metrics for a code block.
    Returns a dict with ncloc and cyclomatic_complexity.
    """
    if not code:
        return {"ncloc": -1, "cyclomatic_complexity": -1}
    return {
        "ncloc": compute_ncloc(code),
        "cyclomatic_complexity": compute_cyclomatic_complexity(code),
    }
