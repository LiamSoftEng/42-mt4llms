import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

from llms.base import BaseLLM
from data.loader import build_prompt
from config.settings import RESULTS_DIR


@dataclass
class LLMResult:
    question_id: str
    question_title: str
    difficulty: str
    platform: str
    model_name: str
    prompt: str
    response: str
    public_test_cases: str   # raw JSON string from dataset
    private_test_cases: str  # raw base64 string from dataset (decoded at test time)
    metadata: str            # raw JSON string from dataset
    error: str = ""


class PipelineRunner:
    def __init__(self, models: List[BaseLLM], dataset, delay: float = 0.5):
        self.models = models
        self.dataset = dataset
        self.delay = delay
        Path(RESULTS_DIR).mkdir(exist_ok=True)

    def run(self):
        for model in self.models:
            print(f"\n{'='*50}")
            print(f"Running model: {model.model_name}")
            print(f"{'='*50}")

            results = []

            for i, sample in enumerate(self.dataset):
                prompt = build_prompt(sample)
                print(prompt)
                print(f"  [{i+1}/{len(self.dataset)}] {sample['question_id']} ({sample['difficulty']})", end=" ... ")

                try:
                    response = model.generate(prompt)
                    status = "ok"
                    error = ""
                except Exception as e:
                    response = ""
                    error = str(e)
                    status = "error"

                print(status)

                results.append(asdict(LLMResult(
                    question_id=sample["question_id"],
                    question_title=sample["question_title"],
                    difficulty=sample["difficulty"],
                    platform=sample["platform"],
                    model_name=model.model_name,
                    prompt=prompt,
                    response=response,
                    public_test_cases=json.dumps(sample.get("public_test_cases", [])),
                    private_test_cases=sample.get("private_test_cases", ""),
                    metadata=json.dumps(sample.get("metadata", {})),
                    error=error,
                )))

                time.sleep(self.delay)

            self._save(model.model_name, results)

        print("\nAll models complete.")

    def _save(self, model_name: str, results: list):
        safe_name = model_name.replace("/", "_").replace(".", "-")
        out_path = Path(RESULTS_DIR) / f"{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(results)} results -> {out_path}")