import time
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
            request["max_completion_tokens"] = MAX_TOKENS
            request["reasoning_effort"] = "low"
        else:
            request["max_tokens"] = MAX_TOKENS

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(**request)
        self._set_last_call_time(time.perf_counter() - t0)

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            thinking_tokens = None
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                thinking_tokens = getattr(details, "reasoning_tokens", None)
            
            self._set_last_usage({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "thinking_tokens": thinking_tokens,
            })
        else:
            self._set_last_usage(None)
        return response.choices[0].message.content