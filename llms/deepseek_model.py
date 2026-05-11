from openai import OpenAI
from .base import BaseLLM
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, MAX_TOKENS, TEMPERATURE

#NOTE: This ended up not actually using the deepseek LLM, its just a tester with free online api keys (google gemma in this case)
class DeepSeekModel(BaseLLM):
    def __init__(self, model: str = DEEPSEEK_MODEL):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=DEEPSEEK_API_KEY,
        )
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            top_p=0.7,
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        response = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response