import pandas as pd
from run_baseline import BaselineRunner
from sklearn.linear_model import LogisticRegression


class LogRegRunner(BaselineRunner):
    def pipeline(self, split: tuple[pd.DataFrame, pd.DataFrame], fold: int) -> dict:
        return self._pipeline(split, fold, impute=True, one_hot=True, standardize=True)

    def create_model(self, *args, **kwargs) -> LogisticRegression:
        print("\n>>> creating model\n")

        if getattr(self.config, "penalty", "none") == "none":
            self.config.penalty = None

        return LogisticRegression(
            solver="saga",
            C=getattr(self.config, "C", 1.0),
            max_iter=getattr(self.config, "max_iter", 100),
            penalty=getattr(self.config, "penalty", "l2"),
            fit_intercept=getattr(self.config, "fit_intercept", True),
        )


if __name__ == "__main__":
    args, config = LogRegRunner.load_args_and_config()

    runner = LogRegRunner(config, args.operation)
    runner.run()
