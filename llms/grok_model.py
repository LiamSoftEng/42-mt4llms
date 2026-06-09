from openai import OpenAI
from .base import BaseLLM
from config.settings import GROK_API_KEY, GROK_MODEL, MAX_TOKENS, TEMPERATURE


class GrokModel(BaseLLM):
    def __init__(self, model: str = GROK_MODEL):
        self.client = OpenAI(
            api_key=GROK_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
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
