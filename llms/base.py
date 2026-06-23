from abc import ABC, abstractmethod
import time
from typing import Callable, TypeVar

from cost_tracker import update_costs

T = TypeVar("T")


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

    def _record_timed_call(self, call: Callable[[], T]) -> T:
        """Run a provider request and store its elapsed wall-clock time."""
        t0 = time.perf_counter()
        try:
            return call()
        finally:
            self._set_last_call_time(time.perf_counter() - t0)

    def _set_openai_compatible_usage(self, response) -> None:
        """Store token usage from OpenAI-compatible response objects."""
        usage = getattr(response, "usage", None)
        if usage is None:
            self._set_last_usage(None)
            return

        thinking_tokens = None
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            thinking_tokens = getattr(details, "reasoning_tokens", None)

        self._set_last_usage({
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "thinking_tokens": thinking_tokens,
        })

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the response text."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier string for this model (used in results output)."""
        pass
