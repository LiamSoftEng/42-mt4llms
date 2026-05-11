import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional

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
    public_test_cases: str
    private_test_cases: str
    metadata: str
    error: str = ""
    # MT fields — only populated when --mr is used
    relation: str = ""
    followup_prompt: str = ""
    followup_response: str = ""
    followup_error: str = ""


class PipelineRunner:
    def __init__(self, models: List[BaseLLM], dataset, delay: float = 0.5, relation=None):
        self.models = models
        self.dataset = dataset
        self.delay = delay
        self.relation = relation
        Path(RESULTS_DIR).mkdir(exist_ok=True)

    def run(self):
        for model in self.models:
            print(f"\n{'='*50}")
            print(f"Running model: {model.model_name}")
            print(f"{'='*50}")

            results = []

            for i, sample in enumerate(self.dataset):
                prompt = build_prompt(sample)
                print(f"  [{i+1}/{len(self.dataset)}] {sample['question_id']} ({sample['difficulty']})", end=" ... ")

                # Source response
                try:
                    response = model.generate(prompt)
                    error = ""
                except Exception as e:
                    response = ""
                    error = str(e)

                # Followup response if MR is set
                followup_prompt = ""
                followup_response = ""
                followup_error = ""
                if self.relation:
                    followup_prompt = self.relation.transform(prompt)
                    try:
                        followup_response = model.generate(followup_prompt)
                    except Exception as e:
                        followup_error = str(e)
                    print(f"ok (+ {self.relation.name})")
                else:
                    print("ok")

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
                    relation=self.relation.name if self.relation else "",
                    followup_prompt=followup_prompt,
                    followup_response=followup_response,
                    followup_error=followup_error,
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