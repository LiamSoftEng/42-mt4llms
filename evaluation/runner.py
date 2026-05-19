import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

from llms.base import BaseLLM
from data.loader import build_code_question, build_prompt, INSTRUCTION_PREFIX, INSTRUCTION_SUFFIX
from config.settings import RESULTS_DIR


@dataclass
class LLMResult:
    question_id: str
    question_title: str
    difficulty: str
    platform: str
    model_name: str
    instruction_prefix: str
    instruction_suffix: str
    question: str
    response: str
    public_test_cases: str
    private_test_cases: str
    metadata: str
    error: str = ""
    # MT fields — only populated when --mr is used
    relation: str = ""
    followup_instruction_prefix: str = ""
    followup_instruction_suffix: str = ""
    followup_question: str = ""
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
                question = build_code_question(sample)
                prompt = build_prompt(question)

                print(f"  [{i+1}/{len(self.dataset)}] {sample['question_id']} ({sample['difficulty']})", end=" ... ")

                # Source response
                try:
                    response = model.generate(prompt)
                    error = ""
                except Exception as e:
                    response = ""
                    error = str(e)

                # Followup response if MR is set
                followup_question = ""
                followup_instruction_prefix = ""
                followup_instruction_suffix = ""
                followup_response = ""
                followup_error = ""

                if self.relation:
                    target = getattr(self.relation, "target", "question")

                    if target == "question":
                        followup_question = self.relation.transform(question)
                        followup_instruction_prefix = INSTRUCTION_PREFIX
                        followup_instruction_suffix = INSTRUCTION_SUFFIX
                        followup_prompt = build_prompt(followup_question)

                    elif target == "instruction":
                        followup_question = question
                        followup_instruction_prefix, followup_instruction_suffix = self.relation.transform(
                            INSTRUCTION_PREFIX, INSTRUCTION_SUFFIX
                        )
                        followup_prompt = build_prompt(question, followup_instruction_prefix, followup_instruction_suffix)

                    try:
                        followup_response = model.generate(followup_prompt)
                    except Exception as e:
                        followup_error = str(e)

                    print(f"ok (+ {self.relation.name} [{target}])")
                else:
                    print("ok")

                results.append(asdict(LLMResult(
                    question_id=sample["question_id"],
                    question_title=sample["question_title"],
                    difficulty=sample["difficulty"],
                    platform=sample["platform"],
                    model_name=model.model_name,
                    instruction_prefix=INSTRUCTION_PREFIX,
                    instruction_suffix=INSTRUCTION_SUFFIX,
                    question=question,
                    response=response,
                    public_test_cases=json.dumps(sample.get("public_test_cases", [])),
                    private_test_cases=sample.get("private_test_cases", ""),
                    metadata=json.dumps(sample.get("metadata", {})),
                    error=error,
                    relation=self.relation.name if self.relation else "",
                    followup_instruction_prefix=followup_instruction_prefix,
                    followup_instruction_suffix=followup_instruction_suffix,
                    followup_question=followup_question,
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