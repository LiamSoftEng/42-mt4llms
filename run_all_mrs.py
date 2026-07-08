import json
import subprocess
import sys
from pathlib import Path

from metamorphic.relations_list import RELATIONS
from config.settings import GROK_MODEL, DEEPSEEK_MODEL
from evaluation.paths import build_results_path


def _load_run_results(model_name: str, relation_name: str, relation_id: int) -> list[dict] | None:
    src = build_results_path(model_name, relation_name, relation_id)
    if not src.exists():
        print(f"Expected results file not found: {src}")
        return None
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    base_results_path = build_results_path(DEEPSEEK_MODEL, None, None)
    model_dir = base_results_path.parent.parent
    out_dir = model_dir / "all_mrs"
    out_path = out_dir / "all_mrs.json"
    combined_results: list[dict] = []
    mr_ids = sorted(RELATIONS.keys())
    for mr_id in mr_ids:
        mr_name = RELATIONS[mr_id]["name"]
        print(f"\n=== Running MR {mr_id}: {mr_name} ===")
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--models",
                "deepseek",
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
        run_results = _load_run_results(DEEPSEEK_MODEL, mr_name, mr_id)
        if run_results is None:
            return 1
        combined_results.extend(run_results)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all MR results -> {out_path}")

    tests_out_path = out_dir / "all_mrs.tests.log"
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
