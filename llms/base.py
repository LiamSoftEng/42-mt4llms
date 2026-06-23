from abc import ABC, abstractmethod

from cost_tracker import update_costs


class BaseLLM(ABC):
    """Abstract base class for all LLM wrappers."""

    @property
    def last_usage(self) -> dict | None:
        """Return the most recent token usage, if available."""
        return getattr(self, "_last_usage", None)
    
    @property
    def last_call_time(self) -> float | None:
        """Return wall-clock seconds for the most recent generate() call."""
        return getattr(self, "_last_call_time", None)

    def _set_last_call_time(self, elapsed: float) -> None:
        """Store wall-clock duration of the most recent generate() call."""
        self._last_call_time = elapsed

    def _set_last_usage(self, usage: dict | None) -> None:
        """Store token usage from the most recent request."""
        self._last_usage = usage
        update_costs(self.model_name, usage)

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the response text."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier string for this model (used in results output)."""
        pass