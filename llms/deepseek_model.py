from openai import OpenAI
from .base import BaseLLM


class DeepSeekModel(BaseLLM):
    def __init__(self, model: str = "google/gemma-2-2b-it"):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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",
        )
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True,
        )

        response = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response