import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

from llms.base import BaseLLM
from data.loader import build_code_question, build_prompt, INSTRUCTION_PREFIX, INSTRUCTION_SUFFIX
from config.settings import RESULTS_DIR


def _safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "-")


def build_results_path(model_name: str, relation_name: str | None = None, relation_id: int | None = None) -> Path:
    model_dir = Path(RESULTS_DIR) / _safe_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    if relation_name:
        if relation_id is not None:
            filename = f"mr-{relation_id:02d}-{relation_name}.json"
        else:
            filename = f"mr-{relation_name}.json"
    else:
        filename = "base.json"
    return model_dir / filename


def _extract_usage(usage: dict | None) -> tuple[int | None, int | None, int | None]:
    if not usage:
        return None, None, None
    return (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


def _accumulate_usage(totals: dict, usage: dict | None) -> None:
    if not usage:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is not None:
            totals[key] += value
            totals["reported"] = True


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
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # MT fields — only populated when --mr is used
    relation: str = ""
    followup_instruction_prefix: str = ""
    followup_instruction_suffix: str = ""
    followup_question: str = ""
    followup_response: str = ""
    followup_error: str = ""
    followup_prompt_tokens: int | None = None
    followup_completion_tokens: int | None = None
    followup_total_tokens: int | None = None


class PipelineRunner:
    def __init__(self, models: List[BaseLLM], dataset, delay: float = 0.5, relation=None, relation_id: int | None = None):
        self.models = models
        self.dataset = dataset
        self.delay = delay
        self.relation = relation
        self.relation_id = relation_id
        Path(RESULTS_DIR).mkdir(exist_ok=True)

    def run(self):
        for model in self.models:
            print(f"\n{'='*50}")
            print(f"Running model: {model.model_name}")
            print(f"{'='*50}")

            results = []
            token_totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reported": False,
            }

            for i, sample in enumerate(self.dataset):
                question = build_code_question(sample)
                prompt = build_prompt(question)

                print(f"  [{i+1}/{len(self.dataset)}] {sample['question_id']} ({sample['difficulty']})", end=" ... ")

                # Source response
                try:
                    response = model.generate(prompt)
                    error = ""
                    usage = model.last_usage
                except Exception as e:
                    response = ""
                    error = str(e)
                    usage = None

                prompt_tokens, completion_tokens, total_tokens = _extract_usage(usage)
                _accumulate_usage(token_totals, usage)

                # Followup response if MR is set
                followup_question = ""
                followup_instruction_prefix = ""
                followup_instruction_suffix = ""
                followup_response = ""
                followup_error = ""
                followup_prompt_tokens = None
                followup_completion_tokens = None
                followup_total_tokens = None

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
                        followup_usage = model.last_usage
                        followup_prompt_tokens, followup_completion_tokens, followup_total_tokens = _extract_usage(
                            followup_usage
                        )
                        _accumulate_usage(token_totals, followup_usage)
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
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    relation=self.relation.name if self.relation else "",
                    followup_instruction_prefix=followup_instruction_prefix,
                    followup_instruction_suffix=followup_instruction_suffix,
                    followup_question=followup_question,
                    followup_response=followup_response,
                    followup_error=followup_error,
                    followup_prompt_tokens=followup_prompt_tokens,
                    followup_completion_tokens=followup_completion_tokens,
                    followup_total_tokens=followup_total_tokens,
                )))

                time.sleep(self.delay)

            self._save(model.model_name, results)

            if token_totals["reported"]:
                print(
                    "  Token usage (total): "
                    f"prompt={token_totals['prompt_tokens']}, "
                    f"completion={token_totals['completion_tokens']}, "
                    f"total={token_totals['total_tokens']}"
                )
            else:
                print("  Token usage (total): not reported by provider")

        print("\nAll models complete.")

    def _save(self, model_name: str, results: list):
        relation_name = self.relation.name if self.relation else None
        out_path = build_results_path(model_name, relation_name, self.relation_id)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(results)} results -> {out_path}")