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
        response = self._record_timed_call(
            lambda: self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
        )
        self._set_openai_compatible_usage(response)
        return response.choices[0].message.content
