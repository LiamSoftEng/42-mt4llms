import anthropic
from .base import BaseLLM
from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS, TEMPERATURE


class ClaudeModel(BaseLLM):
    def __init__(self, model: str = CLAUDE_MODEL):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self._model_name,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text