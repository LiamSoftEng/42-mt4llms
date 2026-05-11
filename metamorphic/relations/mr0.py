class MR0:

    @property
    def name(self) -> str:
        return "do_nothing"

    def transform(self, prompt: str) -> str:
        return prompt

    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed