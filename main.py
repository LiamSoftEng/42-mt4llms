import argparse
import subprocess
import sys

from data.loader import load_livecode_bench
from evaluation.paths import build_results_path
from evaluation.runner import PipelineRunner
from llms import ClaudeModel, OpenAIModel, GeminiModel, DeepSeekModel, GrokModel
from metamorphic.relations_list import RELATIONS

MODEL_CHOICES = {
    "claude": ClaudeModel,
    "openai": OpenAIModel,
    "gemini": GeminiModel,
    "deepseek": DeepSeekModel,
    "grok": GrokModel,
}

def get_relation(mr_id: int):
    if mr_id not in RELATIONS:
        available = list(RELATIONS.keys())
        raise ValueError(f"Unknown MR: {mr_id}. Available: {available}")
    entry = RELATIONS[mr_id]
    return entry["class"]()

def _run_tests(model_name: str, relation_name: str | None, relation_id: int | None, followup_only: bool) -> int:
    results_path = build_results_path(model_name, relation_name, relation_id)
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
        "--mr",
        type=int,
        nargs="+",
        default=None,
        help="Apply one or more metamorphic relations, run sequentially. e.g. --mr 1 4 9",
    )
    parser.add_argument(
        "--mr-only",
        action="store_true",
        help="When used with --mr, skip the base LLM call and generate followup only",
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

    dataset = load_livecode_bench(n_samples=args.n, after=args.after)
    models = [MODEL_CHOICES[name]() for name in args.models]

    if args.mr_only and not args.mr:
        print("--mr-only requires --mr to be set")
        return 2
    if args.repeats < 1:
        print("--repeats must be at least 1")
        return 2

    # None -> single base-only run. Otherwise run each MR id sequentially,
    # reusing the same models/dataset for every relation.
    mr_ids = args.mr if args.mr else [None]

    exit_code = 0
    for mr_id in mr_ids:
        relation = get_relation(mr_id) if mr_id is not None else None

        print(f"\n{'='*60}")
        print(f"Running models: {[m.model_name for m in models]}, on metamorphic relation: {mr_id if mr_id is not None else 'None'}")
        print(f"{'='*60}")

        runner = PipelineRunner(
            models=models,
            dataset=dataset,
            relation=relation,
            relation_id=mr_id,
            mr_only=args.mr_only,
            prompt_repeats=args.repeats,
            resume=args.resume,
        )
        runner.run()

        if args.run_tests:
            for model in models:
                relation_name = relation.name if relation else None
                code = _run_tests(model.model_name, relation_name, mr_id, args.mr_only)
                if code != 0 and exit_code == 0:
                    exit_code = code
            if exit_code != 0:
                break

    if args.run_tests:
        from config.settings import RESULTS_DIR
        from evaluation.results_writer import write_consolidated_outputs
        for model in models:
            print(f"\nConsolidating outputs for {model.model_name}")
            write_consolidated_outputs(model.model_name, RESULTS_DIR)

    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())