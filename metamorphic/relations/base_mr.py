class BaseMR:
    target = "PD"
 
    def transform(self, text: str) -> str:
        raise NotImplementedError
 
    def verify(self, source_passed: bool, followup_passed: bool) -> bool:
        return source_passed == followup_passed