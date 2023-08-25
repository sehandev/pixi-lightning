class BatchSize:
    def __init__(
        self,
        train: int = 1,
        val: int = 1,
        test: int = 1,
    ) -> None:
        self.train = train
        self.val = val
        self.test = test
