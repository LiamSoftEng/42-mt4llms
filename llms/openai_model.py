from openai import OpenAI
from .base import BaseLLM
from config.settings import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE


class OpenAIModel(BaseLLM):
    def __init__(self, model: str = OPENAI_MODEL):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        request = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        # if "gpt-5-nano" not in self._model_name:
            # request["temperature"] = TEMPERATURE
        if self._model_name.startswith("gpt-5"):
            # gpt-5 models require max_completion_tokens instead of max_tokens.
            request["max_completion_tokens"] = MAX_TOKENS
            request["reasoning_effort"] = "low"  # Optional, but can help reduce costs for simpler tasks
        else:
            request["max_tokens"] = MAX_TOKENS
        response = self.client.chat.completions.create(**request)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._set_last_usage({
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            })
        else:
            self._set_last_usage(None)
        return response.choices[0].message.content