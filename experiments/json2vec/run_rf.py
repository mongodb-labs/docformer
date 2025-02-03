import pandas as pd
from run_baseline import BaselineRunner
from sklearn.ensemble import RandomForestClassifier


class RandomForestRunner(BaselineRunner):
    def pipeline(self, split: tuple[pd.DataFrame, pd.DataFrame], fold: int) -> dict:
        return self._pipeline(split, fold, impute=True, one_hot=True, standardize=False)

    def create_model(self, *args, **kwargs) -> RandomForestClassifier:
        print("\n>>> creating model\n")

        if getattr(self.config, "max_depth", 0) == 0:
            self.config.max_depth = None

        if getattr(self.config, "max_features", "none") == "none":
            self.config.max_features = None

        return RandomForestClassifier(
            n_estimators=getattr(self.config, "n_estimators", 100),
            max_features=getattr(self.config, "max_features", "sqrt"),
            max_depth=getattr(self.config, "max_depth", None),
            min_samples_split=getattr(self.config, "min_samples_split", 2),
        )


if __name__ == "__main__":
    args, config = RandomForestRunner.load_args_and_config()

    runner = RandomForestRunner(config, args.operation)
    runner.run()
