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

        response = self._record_timed_call(lambda: self.client.chat.completions.create(**request))
        self._set_openai_compatible_usage(response)
        return response.choices[0].message.content
