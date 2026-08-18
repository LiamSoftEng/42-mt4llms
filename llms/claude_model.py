import anthropic
from .base import BaseLLM
from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS


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
        usage = getattr(message, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "input_tokens", None)
            completion_tokens = getattr(usage, "output_tokens", None)
            total_tokens = None
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            self._set_last_usage({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            })
        else:
            self._set_last_usage(None)
        return message.content[0].text
