import google.generativeai as genai
from .base import BaseLLM
from config.settings import GOOGLE_API_KEY, GEMINI_MODEL, MAX_TOKENS, TEMPERATURE


class GeminiModel(BaseLLM):
    def __init__(self, model: str = GEMINI_MODEL):
        genai.configure(api_key=GOOGLE_API_KEY)
        self._model_name = model
        self.client = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        response = self.client.generate_content(prompt)
        return response.text