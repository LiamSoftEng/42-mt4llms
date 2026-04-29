import argparse
from data.loader import load_livecode_bench
from evaluation.runner import PipelineRunner
from llms import ClaudeModel, OpenAIModel, GeminiModel, DeepSeekModel

MODEL_CHOICES = {
    "claude": ClaudeModel,
    "openai": OpenAIModel,
    "gemini": GeminiModel,
    "deepseek": DeepSeekModel,
}

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
        help="Limit number of samples (default: full dataset). e.g. --n 10",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    dataset = load_livecode_bench(n_samples=args.n)

    models = [MODEL_CHOICES[name]() for name in args.models]
    print(f"Running models: {[m.model_name for m in models]}")

    runner = PipelineRunner(models=models, dataset=dataset)
    runner.run()

if __name__ == "__main__":
    main()