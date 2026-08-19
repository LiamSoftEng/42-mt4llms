class BaseMR:
    target = "PD"

    def transform(self, text: str) -> str:
        raise NotImplementedError
