import argparse
from data.loader import load_livecode_bench
from evaluation.runner import PipelineRunner
from llms import ClaudeModel, OpenAIModel, GeminiModel, DeepSeekModel, GrokModel
import importlib
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
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class"])
    return cls()


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
        default=None,
        help="Apply a metamorphic relation alongside the normal run. e.g. --mr 1",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    dataset = load_livecode_bench(n_samples=args.n, after=args.after)
    models = [MODEL_CHOICES[name]() for name in args.models]
    relation = get_relation(args.mr) if args.mr is not None else None

    print(f"Running models: {[m.model_name for m in models]}, on metamorphic relation: {args.mr is not None}")

    runner = PipelineRunner(models=models, dataset=dataset, relation=relation)
    runner.run()

if __name__ == "__main__":
    main()