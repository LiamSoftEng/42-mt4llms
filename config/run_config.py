"""
Runnable pipeline configuration.

Run the pipeline:
    python config/run_config.py
"""

import subprocess
import sys
from pathlib import Path


CONFIG = {
    "models": ["openai"],
    "n": 2,
    "after": None,
    "difficulty": ["easy"],
    "mr": [1],  # Use a list of IDs, one ID, "all", or None.
    "mr_only": False,
    "run_tests": True,
    "repeats": 1,
    "resume": False,
}

FLAG_NAMES = {
    "models": "--models",
    "n": "--n",
    "after": "--after",
    "difficulty": "--difficulty",
    "mr": "--mr",
    "mr_only": "--mr-only",
    "run_tests": "--run-tests",
    "repeats": "--repeats",
    "resume": "--resume",
}


def _is_empty(value) -> bool:
    return value is None or value == "" or value == []


def build_command(config: dict) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / "main.py")]

    for key, flag in FLAG_NAMES.items():
        value = config.get(key)
        if _is_empty(value):
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(item) for item in value)
            continue
        cmd.extend([flag, str(value)])

    return cmd


def main() -> int:
    cmd = build_command(CONFIG)
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
