import json
import subprocess
import sys
from pathlib import Path

from metamorphic.relations_list import RELATIONS
from config.settings import GROK_MODEL, RESULTS_DIR


def _safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "-")


def _load_run_results(base_name: str) -> list[dict] | None:
    src = Path(RESULTS_DIR) / f"{base_name}.json"
    if not src.exists():
        print(f"Expected results file not found: {src}")
        return None
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    base_name = _safe_name(GROK_MODEL)
    out_path = Path(RESULTS_DIR) / f"{base_name}_all_mrs.json"
    combined_results: list[dict] = []
    mr_ids = sorted(RELATIONS.keys())
    for mr_id in mr_ids:
        print(f"\n=== Running MR {mr_id}: {RELATIONS[mr_id]['name']} ===")
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--models",
                "grok",
                "--n",
                "1",
                "--mr",
                str(mr_id),
            ],
            check=False,
        )
        if result.returncode != 0:
            print(f"MR {mr_id} failed with exit code {result.returncode}")
            return result.returncode
        run_results = _load_run_results(base_name)
        if run_results is None:
            return 1
        combined_results.extend(run_results)

    Path(RESULTS_DIR).mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all MR results -> {out_path}")

    tests_out_path = Path(RESULTS_DIR) / f"{base_name}_all_mrs.tests.log"
    test_result = subprocess.run(
        [
            sys.executable,
            "runtests.py",
            "--results",
            str(out_path),
            "--out",
            str(tests_out_path),
        ],
        check=False,
    )
    if test_result.returncode != 0:
        print(f"Tests failed with exit code {test_result.returncode}")
        return test_result.returncode
    print(f"Saved test log -> {tests_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
