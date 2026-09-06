import argparse
import subprocess
import sys

from cost_tracker import get_model_call_counts, reset_model_call_counts
from data.loader import load_livecode_bench
from evaluation.paths import build_results_path
from evaluation.runner import PipelineRunner
from llms import ClaudeModel, GeminiModel, OpenAIModel
from metamorphic.relations_list import RELATIONS

MODEL_CHOICES = {
    "claude": ClaudeModel,
    "openai": OpenAIModel,
    "gemini": GeminiModel,
}

def get_relation(mr_id: int):
    if mr_id not in RELATIONS:
        available = list(RELATIONS.keys())
        raise ValueError(f"Unknown MR: {mr_id}. Available: {available}")
    entry = RELATIONS[mr_id]
    return entry["class"]()


def parse_mr(value: str) -> int | str:
    if value.lower() == "all":
        return "all"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("MR must be an integer ID or 'all'") from exc


def resolve_relation_ids(values: list[int | str] | None) -> list[int]:
    if not values:
        return []
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with individual MR IDs")
        return sorted(RELATIONS)

    relation_ids = list(dict.fromkeys(values))
    unknown = [mr_id for mr_id in relation_ids if mr_id not in RELATIONS]
    if unknown:
        raise ValueError(f"Unknown MR(s): {unknown}. Available: {sorted(RELATIONS)}")
    return relation_ids

def _run_tests(
    model_name: str,
    relation_name: str | None,
    relation_id: int | None,
    followup_only: bool,
    difficulty: str | None = None,
) -> int:
    results_path = build_results_path(model_name, relation_name, relation_id, difficulty)
    if not results_path.exists():
        print(f"Results file not found for tests: {results_path}")
        return 1
    cmd = [sys.executable, "runtests.py", "--results", str(results_path), "--write-outputs"]
    if followup_only:
        cmd.append("--followup-only")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Tests failed for {model_name} (exit {result.returncode})")
    return result.returncode

def parse_args():
    parser = argparse.ArgumentParser(description="LLM MT Pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES.keys(),
        default=list(MODEL_CHOICES.keys()),
        help="Which models to run (default: all). e.g. --models claude openai",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit number of samples. e.g. --n 10",
    )
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help="Only include problems on or after this date. e.g. --after 2024-01-01",
    )
    parser.add_argument(
        "--difficulty",
        nargs="+",
        choices=["easy", "medium", "hard"],
        default=None,
        help=(
            "Run each listed difficulty separately, applying --n to each one. "
            "e.g. --difficulty easy medium"
        ),
    )
    parser.add_argument(
        "--mr",
        nargs="+",
        type=parse_mr,
        default=None,
        metavar="ID|all",
        help="Apply one or more metamorphic relations. e.g. --mr 1 4 7 or --mr all",
    )
    parser.add_argument(
        "--mr-only",
        action="store_true",
        help="When used with --mr, reuse shared base results and generate any missing bases",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run runtests.py on each model's results after generation",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Generate each source/followup prompt this many times to measure prompt variability",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing results file by skipping entries with required code blocks",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    try:
        relation_ids = resolve_relation_ids(args.mr)
    except ValueError as exc:
        print(exc)
        return 2
    if args.mr_only and not relation_ids:
        print("--mr-only requires --mr to be set")
        return 2
    if args.repeats < 1:
        print("--repeats must be at least 1")
        return 2

    models = [MODEL_CHOICES[name]() for name in args.models]
    runs = relation_ids or [None]
    difficulty_runs = list(dict.fromkeys(args.difficulty)) if args.difficulty else [None]
    print(
        f"Running models: {[model.model_name for model in models]}, "
        f"difficulties: {difficulty_runs if args.difficulty else ['all']}, "
        f"MRs: {relation_ids or ['base']}"
    )
    reset_model_call_counts()

    try:
        for difficulty in difficulty_runs:
            dataset = load_livecode_bench(
                n_samples=args.n,
                after=args.after,
                difficulty=difficulty,
            )
            for relation_id in runs:
                relation = get_relation(relation_id) if relation_id is not None else None
                runner = PipelineRunner(
                    models=models,
                    dataset=dataset,
                    relation=relation,
                    relation_id=relation_id,
                    mr_only=args.mr_only,
                    prompt_repeats=args.repeats,
                    resume=args.resume,
                    difficulty=difficulty,
                )
                runner.run()

                if args.run_tests:
                    for model in models:
                        relation_name = relation.name if relation else None
                        exit_code = _run_tests(
                            model.model_name,
                            relation_name,
                            relation_id,
                            args.mr_only,
                            difficulty,
                        )
                        if exit_code != 0:
                            return exit_code
    finally:
        call_counts = get_model_call_counts()
        total_calls = call_counts["main"] + call_counts["helper"]
        print(
            f"\nModel calls (total): main={call_counts['main']}, "
            f"helper={call_counts['helper']}, combined={total_calls}"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
