from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for all LLM wrappers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the response text."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier string for this model (used in results output)."""
        pass