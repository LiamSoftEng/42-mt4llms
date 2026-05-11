from google import genai
from google.genai import types
from .base import BaseLLM
from config.settings import GOOGLE_API_KEY, GEMINI_MODEL, MAX_TOKENS, TEMPERATURE

class GeminiModel(BaseLLM):
    def __init__(self, model: str = GEMINI_MODEL):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self._model_name = model
 
    @property
    def model_name(self) -> str:
        return self._model_name
 
    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        return response.text
