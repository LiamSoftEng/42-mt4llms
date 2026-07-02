from openai import OpenAI
from config.settings import OPENAI_API_KEY, MR_HELPER_MODEL


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

    def transform(self, system: str, user: str) -> str:
        """
        Call the MR helper LLM with a system + user prompt.
        Returns the raw text response, or the original user content on failure
        so a broken helper never kills a full pipeline run.
        """
        request = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._model_name.startswith("gpt-5"):
            request["max_completion_tokens"] = 2048
            request["reasoning_effort"] = "low"
        else:
            request["max_tokens"] = 2048

        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            print(f"  [mr_helper] call failed ({exc}), returning input unchanged")
            self._last_usage = None
            return user

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        else:
            self._last_usage = None

        return response.choices[0].message.content.strip()


_helper: MRHelper | None = None


def _get_helper() -> MRHelper:
    global _helper
    if _helper is None:
        _helper = MRHelper()
    return _helper


def mr_llm_transform(system: str, user: str) -> str:
    return _get_helper().transform(system, user)