NOISE = "Note: This problem was originally presented in an international programming contest."


class MR1:

    @property
    def name(self) -> str:
        return "dead_context"

    def transform(self, prompt: str) -> str:
        """Insert an irrelevant sentence after the first line of the prompt."""
        lines = prompt.split("\n")
        lines.insert(1, NOISE)
        return "\n".join(lines)

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed