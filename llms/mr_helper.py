from openai import OpenAI
from config.settings import OPENAI_API_KEY, MR_HELPER_MODEL, RESULTS_DIR
import json
import time
from pathlib import Path

HELPER_LOG_PATH = Path(RESULTS_DIR) / "helper_call_log.jsonl"

def _log_helper_call(system: str, user: str, response_text: str, finish_reason, usage) -> None:
    HELPER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HELPER_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": time.time(),
            "system_prompt": system,
            "user_input": user,
            "raw_response": response_text,
            "finish_reason": finish_reason,
            "usage": usage,
        }) + "\n")

class MRHelper:

    def __init__(self, model: str = MR_HELPER_MODEL):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self._model_name = model
        self._last_usage: dict | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def last_usage(self) -> dict | None:
        return self._last_usage
    
    @property
    def last_finish_reason(self):
        return getattr(self, "_last_finish_reason", None)

    def transform(self, system: str, user: str) -> str:
        request = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            print(f"  [mr_helper] call failed ({exc}), returning input unchanged")
            self._last_usage = None
            self._last_finish_reason = None
            _log_helper_call(system, user, "", None, None)
            return user

        choice = response.choices[0]
        self._last_finish_reason = choice.finish_reason

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        else:
            self._last_usage = None

        result = choice.message.content.strip()
        _log_helper_call(system, user, result, self._last_finish_reason, self._last_usage)
        return result


_helper: MRHelper | None = None


def _get_helper() -> MRHelper:
    global _helper
    if _helper is None:
        _helper = MRHelper()
    return _helper


def mr_llm_transform(system: str, user: str) -> str:
    return _get_helper().transform(system, user)
