import pandas as pd
from lightgbm import LGBMClassifier
from run_baseline import BaselineRunner


class LightGBMRunner(BaselineRunner):
    def pipeline(self, split: tuple[pd.DataFrame, pd.DataFrame], fold: int) -> dict:
        return self._pipeline(split, fold, impute=False, one_hot=True, standardize=False)

    def create_model(self, *args, **kwargs) -> LGBMClassifier:
        print("\n>>> creating model\n")

        # hyperparameters set to their default lightgbm values unless overridden by config
        return LGBMClassifier(
            random_state=42,
            verbose=-1,
            num_leaves=getattr(self.config, "num_leaves", 31),
            max_depth=getattr(self.config, "max_depth", -1),
            learning_rate=getattr(self.config, "learning_rate", 0.1),
            n_estimators=getattr(self.config, "n_estimators", 100),
            min_child_weight=getattr(self.config, "min_child_weight", 1e-3),
            subsample=getattr(self.config, "subsample", 1.0),
            colsample_bytree=getattr(self.config, "colsample_bytree", 1.0),
            reg_alpha=getattr(self.config, "alpha", 0.0),
            reg_lambda=getattr(self.config, "lambda", 0.0),
        )


if __name__ == "__main__":
    args, config = LightGBMRunner.load_args_and_config()

    runner = LightGBMRunner(config, args.operation)
    runner.run()
