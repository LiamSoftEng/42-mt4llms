import json
import time
from dataclasses import asdict, dataclass, field, replace
from typing import List
from pathlib import Path

from llms.base import BaseLLM
from data.loader import build_code_question, build_prompt, INSTRUCTION_PREFIX, INSTRUCTION_SUFFIX
from evaluation.executor import extract_code
from evaluation.paths import build_results_path, build_repeat_path, build_repeats_dir
from config.settings import RESULTS_DIR


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


def _extract_usage(usage: dict | None) -> tuple[int | None, int | None, int | None, int | None]:
    """Return (prompt_tokens, completion_tokens, total_tokens, thinking_tokens)."""
    if not usage:
        return None, None, None, None
    return (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
        usage.get("thinking_tokens"),
    )


def _accumulate_usage(totals: dict, usage: dict | None) -> None:
    if not usage:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "thinking_tokens"):
        value = usage.get(key)
        if value is not None:
            totals[key] = totals.get(key, 0) + value
            totals["reported"] = True


def _token_efficiency(usage: dict | None) -> float | None:
    _, completion_tokens, _, thinking_tokens = _extract_usage(usage)
    if thinking_tokens and completion_tokens:
        return round(completion_tokens / thinking_tokens, 6)
    return None


@dataclass
class GeneratedResponses:
    responses: list[str] = field(default_factory=list)
    usages: list[dict | None] = field(default_factory=list)
    call_times: list[float | None] = field(default_factory=list)
    finish_reasons: list[str | None] = field(default_factory=list)
    error: str = ""

    @property
    def primary_response(self) -> str:
        return self.responses[0] if self.responses else ""

    @property
    def primary_usage(self) -> dict | None:
        return self.usages[0] if self.usages else None

    @property
    def primary_call_time(self) -> float | None:
        return self.call_times[0] if self.call_times else None

    @property
    def primary_finish_reason(self) -> str | None:
        return self.finish_reasons[0] if self.finish_reasons else None


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

    # Token counts — source
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    thinking_tokens: int | None = None          # reasoning/thinking budget consumed
    token_efficiency: float | None = None       # completion_tokens / thinking_tokens

    # Timing — source
    llm_call_time: float | None = None          # wall-clock seconds for generate()
    finish_reason: str | None = None

    # MT fields — only populated when --mr is used
    relation: str = ""
    followup_instruction_prefix: str = ""
    followup_instruction_suffix: str = ""
    followup_question: str = ""
    followup_response: str = ""
    followup_error: str = ""
    followup_response_samples: list[str] = field(default_factory=list)

    # Token counts — followup
    followup_prompt_tokens: int | None = None
    followup_completion_tokens: int | None = None
    followup_total_tokens: int | None = None
    followup_thinking_tokens: int | None = None
    followup_token_efficiency: float | None = None

    # Timing — followup
    followup_llm_call_time: float | None = None
    followup_finish_reason: str | None = None


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

    def _generate_responses(
        self,
        model: BaseLLM,
        prompt: str,
        token_totals: dict,
    ) -> GeneratedResponses:
        generated = GeneratedResponses()
        for repeat_index in range(self.prompt_repeats):
            try:
                response = model.generate(prompt)
                usage = model.last_usage
                call_time = model.last_call_time
                finish_reason = model.last_finish_reason
            except Exception as exc:
                response = ""
                usage = None
                call_time = None
                finish_reason = None
                if repeat_index == 0:
                    generated.error = str(exc)

            generated.responses.append(response)
            generated.usages.append(usage)
            generated.call_times.append(call_time)
            generated.finish_reasons.append(finish_reason)
            _accumulate_usage(token_totals, usage)
            if repeat_index < self.prompt_repeats - 1:
                time.sleep(self.delay)
        return generated

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

            token_totals: dict = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "thinking_tokens": 0,
                "reported": False,
            }

            base_results = {}
            if self.relation and self.mr_only:
                base_results = _load_base_results(model.model_name)

            # For a plain base run (no --mr) with --repeats > 1, each repeat is
            # also written out as its own full results file under base/repeats/,
            # so it can be tested and compared independently of the others.
            write_repeat_files = self.relation is None and self.prompt_repeats > 1
            repeat_results: list[list[dict]] = (
                [[] for _ in range(self.prompt_repeats)] if write_repeat_files else []
            )

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

                # ── Source response ──────────────────────────────────────────
                source = GeneratedResponses()

                if not (self.relation and self.mr_only):
                    source = self._generate_responses(model, prompt, token_totals)

                response = source.primary_response
                response_samples = source.responses
                error = source.error
                usage = source.primary_usage
                llm_call_time = source.primary_call_time
                finish_reason = source.primary_finish_reason

                if self.relation and self.mr_only and base_results:
                    base_entry = base_results.get(sample["question_id"])
                    if base_entry:
                        response = base_entry.get("response", "")
                        response_samples = base_entry.get("response_samples") or ([response] if response else [])
                        error = base_entry.get("error", "")
                        llm_call_time = base_entry.get("llm_call_time")
                        finish_reason = base_entry.get("finish_reason")

                prompt_tokens, completion_tokens, total_tokens, thinking_tokens = _extract_usage(usage)
                token_efficiency = _token_efficiency(usage)

                # ── Followup response ────────────────────────────────────────
                followup_question = ""
                followup_instruction_prefix = ""
                followup_instruction_suffix = ""
                followup_response = ""
                followup_response_samples = []
                followup_error = ""
                followup_prompt_tokens = None
                followup_completion_tokens = None
                followup_total_tokens = None
                followup_thinking_tokens = None
                followup_token_efficiency = None
                followup_llm_call_time: float | None = None
                followup_finish_reason: str | None = None

                if self.relation:
                    target = getattr(self.relation, "target", "PD")

                    if target == "PD":
                        followup_question = self.relation.transform(question)
                        followup_instruction_prefix = INSTRUCTION_PREFIX
                        followup_instruction_suffix = INSTRUCTION_SUFFIX
                        followup_prompt = build_prompt(followup_question)
                    elif target == "I":
                        followup_question = question
                        followup_instruction_prefix, followup_instruction_suffix = self.relation.transform(
                            INSTRUCTION_PREFIX, INSTRUCTION_SUFFIX
                        )
                        followup_prompt = build_prompt(question, followup_instruction_prefix, followup_instruction_suffix)

                    followup = self._generate_responses(model, followup_prompt, token_totals)
                    followup_response = followup.primary_response
                    followup_response_samples = followup.responses
                    followup_error = followup.error
                    (
                        followup_prompt_tokens,
                        followup_completion_tokens,
                        followup_total_tokens,
                        followup_thinking_tokens,
                    ) = _extract_usage(followup.primary_usage)
                    followup_token_efficiency = _token_efficiency(followup.primary_usage)
                    followup_llm_call_time = followup.primary_call_time
                    followup_finish_reason = followup.primary_finish_reason

                    if self.mr_only:
                        print(f"ok (mr-only + {self.relation.name} [{target}], repeats={self.prompt_repeats})")
                    else:
                        print(f"ok (+ {self.relation.name} [{target}], repeats={self.prompt_repeats})")
                else:
                    print(f"ok (repeats={self.prompt_repeats})" if self.prompt_repeats > 1 else "ok")

                result = LLMResult(
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
                    thinking_tokens=thinking_tokens,
                    token_efficiency=token_efficiency,
                    llm_call_time=llm_call_time,
                    finish_reason=finish_reason,
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
                    followup_thinking_tokens=followup_thinking_tokens,
                    followup_token_efficiency=followup_token_efficiency,
                    followup_llm_call_time=followup_llm_call_time,
                    followup_finish_reason=followup_finish_reason,
                )
                result_entry = asdict(result)
                results = [entry for entry in results if entry.get("question_id") != sample["question_id"]]
                results.append(result_entry)
                completed_ids.add(sample["question_id"])
                self._save_to_path(out_path, results, quiet=True)

                # ── Per-repeat result files (base run only, repeats > 1) ───────
                if write_repeat_files:
                    for idx, rep_response in enumerate(response_samples):
                        rep_usage = source.usages[idx] if idx < len(source.usages) else None
                        rep_ptok, rep_ctok, rep_ttok, rep_thtok = _extract_usage(rep_usage)
                        rep_entry = asdict(replace(
                            result,
                            response=rep_response,
                            response_samples=[],
                            error=error if idx == 0 else "",
                            prompt_tokens=rep_ptok,
                            completion_tokens=rep_ctok,
                            total_tokens=rep_ttok,
                            thinking_tokens=rep_thtok,
                            token_efficiency=_token_efficiency(rep_usage),
                            llm_call_time=source.call_times[idx] if idx < len(source.call_times) else None,
                            finish_reason=(
                                source.finish_reasons[idx] if idx < len(source.finish_reasons) else None
                            ),
                        ))
                        repeat_results[idx] = [
                            e for e in repeat_results[idx] if e.get("question_id") != sample["question_id"]
                        ]
                        repeat_results[idx].append(rep_entry)
                        self._save_to_path(build_repeat_path(model.model_name, idx), repeat_results[idx], quiet=True)

                time.sleep(self.delay)

            self._save_to_path(out_path, results)

            if write_repeat_files:
                print(f"  Saved {self.prompt_repeats} per-repeat result files -> {build_repeats_dir(model.model_name)}")

            if token_totals["reported"]:
                thinking_total = token_totals.get("thinking_tokens", 0) or 0
                thinking_str = f", thinking={thinking_total}" if thinking_total else ""
                print(
                    "  Token usage (total): "
                    f"prompt={token_totals['prompt_tokens']}, "
                    f"completion={token_totals['completion_tokens']}, "
                    f"total={token_totals['total_tokens']}"
                    f"{thinking_str}"
                )
            else:
                print("  Token usage (total): not reported by provider")

        print("\nAll models complete.")

    def _save_to_path(self, out_path: Path, results: list, quiet: bool = False):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if not quiet:
            print(f"\n  Saved {len(results)} results -> {out_path}")
