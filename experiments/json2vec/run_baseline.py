from typing import Any

import joblib
import pandas as pd
from pymongo import MongoClient
from runner import BaseRunner
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from utils import IdentityTransformer

from origami.utils.common import flatten_docs
from origami.utils.guild import load_secrets, print_guild_scalars


class BaselineRunner(BaseRunner):
    def fetch_df(self):
        print("\n>>> fetching data\n")
        secrets = load_secrets()

        # load documents from MongoDB
        client = MongoClient(secrets["MONGO_URI"])
        collection = client[self.config.data.db][self.config.data.coll]
        docs = list(collection.find({}, projection=self.config.data.projection))

        # flatten documents and create DataFrame
        df = pd.DataFrame(flatten_docs(docs))

        return df

    def _pipeline(
        self, split: tuple[pd.DataFrame, pd.DataFrame], fold: int, impute: bool, one_hot: bool, standardize: bool
    ) -> dict:
        print(f"\n>>> executing pipeline for fold {fold}\n")

        X_train, X_test = split

        # extract target
        y_train = X_train[self.config.data.target_field]
        y_test = X_test[self.config.data.target_field]

        # remove target from features
        X_train.drop(self.config.data.target_field, axis=1, inplace=True)
        X_test.drop(self.config.data.target_field, axis=1, inplace=True)

        # preprocess numeric and categorical features
        num_features = X_train.select_dtypes("number").columns
        num_steps = [("identity", IdentityTransformer())]

        if impute:
            num_steps.append(("imputer", SimpleImputer(strategy="median")))
        if standardize:
            num_steps.append(("scaler", StandardScaler()))

        numeric_transformer = Pipeline(steps=num_steps)

        cat_features = X_train.select_dtypes("object").columns

        # replace all categorical nan values with "n/a" string
        X_train[cat_features] = X_train[cat_features].fillna("n/a")
        X_test[cat_features] = X_test[cat_features].fillna("n/a")

        # convert categorical features to strings
        if one_hot:
            X_train[cat_features] = X_train[cat_features].astype("string")
            X_test[cat_features] = X_test[cat_features].astype("string")
            cat_steps = [("encoder", OneHotEncoder(handle_unknown="ignore"))]
        else:
            cat_steps = [("identity", IdentityTransformer())]

        categorical_transformer = Pipeline(steps=cat_steps)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_features),
                ("cat", categorical_transformer, cat_features),
            ],
            remainder="passthrough",
            sparse_threshold=0,
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # label-encode targets
        label_encoder = LabelEncoder()

        label_encoder.fit(pd.concat((y_train, y_test), axis=0))
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def train(self, model, data: dict, fold: int) -> None:
        print(f"\n>>> training model for fold {fold}\n")

        model.fit(data["X_train"], data["y_train"])
        # if this is a 'train' run, also print out evaluation results because .fit() is silent
        if self.operation == "train":
            self.eval(model, data, fold)

    def train_all(self):
        """scikit-learn models don't print anything when trained, so we can just train them within this
        guild run directly."""
        print("\n>>> training models on all folds\n")
        df = self.fetch_df()
        splits = self.get_splits(df)
        for k, split in enumerate(splits):
            data = self.pipeline(split, fold=k)
            self.save_data(data, fold=k)
            model = self.create_model()
            self.train(model, data, fold=k)
            self.save_model(model, fold=k)

    def eval(self, model, data: dict, fold: int) -> None:
        print(f"\n>>> evaluating model for fold {fold}\n")
        train_acc = accuracy_score(data["y_train"], model.predict(data["X_train"]))
        test_acc = accuracy_score(data["y_test"], model.predict(data["X_test"]))

        print_guild_scalars(
            fold=fold,
            train_acc=f"{train_acc:.4f}",
            test_acc=f"{test_acc:.4f}",
        )
        return {"train_acc": train_acc, "test_acc": test_acc}

    def save_model(self, model: Any, fold: int) -> None:
        print(f"\n>>> saving model for fold {fold}\n")

        joblib.dump(model, f"model_fold_{fold}.joblib")

    def load_model(self, data: dict, fold: int) -> Any:
        print(f"\n>>> loading model for fold {fold}\n")

        model = joblib.load(f"model_fold_{fold}.joblib")
        return model


if __name__ == "__main__":
    args, config = BaselineRunner.load_args_and_config()

    runner = BaselineRunner(config, args.operation)
    runner.run()
