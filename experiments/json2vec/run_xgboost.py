import pandas as pd
from run_baseline import BaselineRunner
from xgboost import XGBClassifier


class XGBoostRunner(BaselineRunner):
    def pipeline(self, split: tuple[pd.DataFrame, pd.DataFrame], fold: int) -> dict:
        return self._pipeline(split, fold, impute=False, one_hot=True, standardize=False)

    def create_model(self, *args, **kwargs) -> XGBClassifier:
        print("\n>>> creating model\n")

        # hyperparameters set to their default xgboost values unless overridden by config
        return XGBClassifier(
            seed=42,
            enable_categorical=True,
            use_label_encoder=False,
            n_estimators=getattr(self.config, "n_estimators", 100),
            learning_rate=getattr(self.config, "learning_rate", 0.3),  # e-7 to e-1
            max_depth=getattr(self.config, "max_depth", 6),
            subsample=getattr(self.config, "subsample", 1.0),
            colsample_bytree=getattr(self.config, "colsample_bytree", 1.0),
            colsample_bylevel=getattr(self.config, "colsample_bylevel", 1.0),
            min_child_weight=getattr(self.config, "min_child_weight", 1.0),
            reg_alpha=getattr(self.config, "alpha", 0.0),
            reg_lambda=getattr(self.config, "lambda", 1.0),
            gamma=getattr(self.config, "gamma", 0.0),
            eval_metric="logloss",
        )


if __name__ == "__main__":
    args, config = XGBoostRunner.load_args_and_config()

    runner = XGBoostRunner(config, args.operation)
    runner.run()
