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
        response = self.client.chat.completions.create(
            model=self._model_name,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content