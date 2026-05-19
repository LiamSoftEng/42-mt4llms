from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for all LLM wrappers."""

    @property
    def last_usage(self) -> dict | None:
        """Return the most recent token usage, if available."""
        return getattr(self, "_last_usage", None)

    def _set_last_usage(self, usage: dict | None) -> None:
        """Store token usage from the most recent request."""
        self._last_usage = usage

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the response text."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier string for this model (used in results output)."""
        pass