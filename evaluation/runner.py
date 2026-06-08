import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List

from llms.base import BaseLLM
from data.loader import build_code_question, build_prompt, INSTRUCTION_PREFIX, INSTRUCTION_SUFFIX
from evaluation.executor import extract_code
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


def _load_base_results(model_name: str) -> dict[str, dict]:
    base_path = build_results_path(model_name, None, None)
    if not base_path.exists():
        print(f"  Base results not found for MR-only run: {base_path}")
        return {}
    try:
        with open(base_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f"  Failed to read base results for MR-only run: {base_path}")
        return {}
    if not isinstance(data, list):
        print(f"  Unexpected base results format (expected list): {base_path}")
        return {}
    return {
    qid: entry
    for entry in data
    if isinstance(entry, dict)
    and isinstance((qid := entry.get("question_id")), str)
    and qid
}


def _load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f"  Existing results could not be read, starting fresh: {path}")
        return []
    if not isinstance(data, list):
        print(f"  Existing results are not a list, starting fresh: {path}")
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def _entry_has_code(entry: dict, response_key: str) -> bool:
    response = entry.get(response_key, "")
    return isinstance(response, str) and extract_code(response) is not None


def _entry_complete_for_resume(entry: dict, relation=None, mr_only: bool = False) -> bool:
    if not isinstance(entry.get("question_id"), str) or not entry["question_id"]:
        return False
    if mr_only:
        return _entry_has_code(entry, "followup_response")
    if relation:
        return _entry_has_code(entry, "response") and _entry_has_code(entry, "followup_response")
    return _entry_has_code(entry, "response")


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
    response_samples: list[str] = field(default_factory=list)
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
    followup_response_samples: list[str] = field(default_factory=list)
    followup_prompt_tokens: int | None = None
    followup_completion_tokens: int | None = None
    followup_total_tokens: int | None = None


class PipelineRunner:
    def __init__(
        self,
        models: List[BaseLLM],
        dataset,
        delay: float = 0.5,
        relation=None,
        relation_id: int | None = None,
        mr_only: bool = False,
        prompt_repeats: int = 1,
        resume: bool = False,
    ):
        self.models = models
        self.dataset = dataset
        self.delay = delay
        self.relation = relation
        self.relation_id = relation_id
        self.mr_only = mr_only
        self.prompt_repeats = max(1, prompt_repeats)
        self.resume = resume
        Path(RESULTS_DIR).mkdir(exist_ok=True)

    def run(self):
        for model in self.models:
            print(f"\n{'='*50}")
            print(f"Running model: {model.model_name}")
            print(f"{'='*50}")

            out_path = build_results_path(
                model.model_name,
                self.relation.name if self.relation else None,
                self.relation_id,
            )
            results = _load_existing_results(out_path) if self.resume else []
            completed_ids = {
                entry.get("question_id")
                for entry in results
                if _entry_complete_for_resume(entry, self.relation, self.mr_only)
            }
            if self.resume and completed_ids:
                print(f"  Resume enabled: loaded {len(completed_ids)} completed code results from {out_path}")

            token_totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reported": False,
            }

            base_results = {}
            if self.relation and self.mr_only:
                base_results = _load_base_results(model.model_name)

            for i, sample in enumerate(self.dataset):
                if self.resume and sample["question_id"] in completed_ids:
                    print(
                        f"  [{i+1}/{len(self.dataset)}] {sample['question_id']} "
                        f"({sample['difficulty']}) ... skipped (already complete)"
                    )
                    continue

                question = build_code_question(sample)
                prompt = build_prompt(question)

                print(f"  [{i+1}/{len(self.dataset)}] {sample['question_id']} ({sample['difficulty']})", end=" ... ")

                response = ""
                response_samples = []
                error = ""
                usage = None
                if not (self.relation and self.mr_only):
                    # Source response
                    for repeat_index in range(self.prompt_repeats):
                        try:
                            current_response = model.generate(prompt)
                            current_usage = model.last_usage
                        except Exception as e:
                            current_response = ""
                            current_usage = None
                            if repeat_index == 0:
                                error = str(e)

                        response_samples.append(current_response)
                        if repeat_index == 0:
                            response = current_response
                            usage = current_usage
                        _accumulate_usage(token_totals, current_usage)
                        if repeat_index < self.prompt_repeats - 1:
                            time.sleep(self.delay)

                if self.relation and self.mr_only and base_results:
                    base_entry = base_results.get(sample["question_id"])
                    if base_entry:
                        response = base_entry.get("response", "")
                        response_samples = base_entry.get("response_samples") or ([response] if response else [])
                        error = base_entry.get("error", "")

                prompt_tokens, completion_tokens, total_tokens = _extract_usage(usage)

                # Followup response if MR is set
                followup_question = ""
                followup_instruction_prefix = ""
                followup_instruction_suffix = ""
                followup_response = ""
                followup_response_samples = []
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

                    for repeat_index in range(self.prompt_repeats):
                        try:
                            current_followup_response = model.generate(followup_prompt)
                            followup_usage = model.last_usage
                        except Exception as e:
                            current_followup_response = ""
                            followup_usage = None
                            if repeat_index == 0:
                                followup_error = str(e)

                        followup_response_samples.append(current_followup_response)
                        if repeat_index == 0:
                            followup_response = current_followup_response
                            followup_prompt_tokens, followup_completion_tokens, followup_total_tokens = _extract_usage(
                                followup_usage
                            )
                        _accumulate_usage(token_totals, followup_usage)
                        if repeat_index < self.prompt_repeats - 1:
                            time.sleep(self.delay)

                    if self.mr_only:
                        print(f"ok (mr-only + {self.relation.name} [{target}], repeats={self.prompt_repeats})")
                    else:
                        print(f"ok (+ {self.relation.name} [{target}], repeats={self.prompt_repeats})")
                else:
                    print(f"ok (repeats={self.prompt_repeats})" if self.prompt_repeats > 1 else "ok")

                result_entry = asdict(LLMResult(
                    question_id=sample["question_id"],
                    question_title=sample["question_title"],
                    difficulty=sample["difficulty"],
                    platform=sample["platform"],
                    model_name=model.model_name,
                    instruction_prefix=INSTRUCTION_PREFIX,
                    instruction_suffix=INSTRUCTION_SUFFIX,
                    question=question,
                    response=response,
                    response_samples=response_samples,
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
                    followup_response_samples=followup_response_samples,
                    followup_error=followup_error,
                    followup_prompt_tokens=followup_prompt_tokens,
                    followup_completion_tokens=followup_completion_tokens,
                    followup_total_tokens=followup_total_tokens,
                ))
                results = [entry for entry in results if entry.get("question_id") != sample["question_id"]]
                results.append(result_entry)
                completed_ids.add(sample["question_id"])
                self._save_to_path(out_path, results, quiet=True)

                time.sleep(self.delay)

            self._save_to_path(out_path, results)

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

    def _save_to_path(self, out_path: Path, results: list, quiet: bool = False):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if not quiet:
            print(f"\n  Saved {len(results)} results -> {out_path}")
