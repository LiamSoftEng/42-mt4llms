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
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            completion_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)
            self._set_last_usage({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            })
        else:
            self._set_last_usage(None)
        return response.text
