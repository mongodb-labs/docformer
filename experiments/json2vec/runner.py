import argparse
import json
import os
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pymongo import MongoClient
from sklearn.model_selection import KFold, train_test_split
from utils import flatten_config

from origami.utils import set_seed
from origami.utils.config import TopLevelConfig
from origami.utils.guild import detect_remote, get_run_objects, load_secrets, print_guild_scalars


class BaseRunner:
    """This class is the basis for running different models. It takes care of loading the config, routing
    for the different operations (data, train, eval, all, ...) and contains config validation and
    cross-validation logic."""

    def __init__(self, config: OmegaConf, operation: str) -> None:
        self.config = config
        self.is_local = not detect_remote()
        self.operation = operation

        # set seed
        set_seed(config.seed)

    @staticmethod
    def load_args_and_config() -> tuple[argparse.Namespace, OmegaConf]:
        """
        Load the command line arguments and configuration from the specified YAML files.

        This static method takes no parameters and returns a tuple containing two objects:
        - args: An instance of the `argparse.Namespace` class representing the parsed command line arguments.
        - config: An instance of the `OmegaConf` class representing the loaded configuration.

        It validates the different sub-configs (model, train, pipeline) against the dataclasses defined in
        axon/gpt/config.py.

        Run scripts should call this method in their `__main__` section like so:

            if __name__ == "__main__":
                args, config = MyRunner.load_args_and_config()
                runner = MyRunner(config, args.operation)
                runner.run()

        """

        p = ArgumentParser()
        p.add_argument("operation", type=str, choices=["data", "train", "eval", "all", "embed"])
        args = p.parse_args()

        # load flags from yaml file
        config = OmegaConf.load("flags.yml")

        # load data config based on dataset and overwrite any config parameters from flags
        data_config = OmegaConf.load(Path("./datasets") / f"{config.dataset}.yml")

        # validate against the dataclasses in config.py
        schema = OmegaConf.create(TopLevelConfig)

        # will raise if config is invalid
        validated = OmegaConf.merge(
            schema,
            {"model": config.model, "train": config.train, "pipeline": config.pipeline, "data": data_config.data},
        )

        # merge back in top-level keys (dataset, seed)
        config = OmegaConf.merge(config, validated)

        return args, config

    def run(self):
        """
        This function is responsible for executing the main logic of the program based on the
        value of the `self.operation` attribute.

        - data: fetches the data from MongoDB and create a dataset_fold_{n}.pickle.gz file
                for each fold.
        - train: trains a model on a single fold of the data
        - eval: evaluates a model on a single fold of the data
        - all: wrapper that prepares the data, trains and evaluates a model on all folds
        """

        # verify checks. if verify raises an exception, exit with an error
        try:
            self.verify()
        except Exception as e:
            print(e)
            sys.exit(1)

        if self.operation == "data":
            df = self.fetch_df()
            splits = self.get_splits(df)
            for k, split in enumerate(splits):
                data = self.pipeline(split, fold=k)
                self.save_data(data, fold=k)

        elif self.operation == "train":
            data = self.load_data(fold=self.config.fold)
            model = self.create_model(data)
            self.train(model, data, fold=self.config.fold)
            self.save_model(model, fold=self.config.fold)

        elif self.operation == "eval":
            data = self.load_data(fold=self.config.fold)
            model = self.load_model(data, fold=self.config.fold)
            self.eval(model, data, fold=self.config.fold)

        elif self.operation == "all":
            df = self.fetch_df()
            splits = self.get_splits(df)
            if len(splits) == 1:
                # for a single split, call train directly
                data = self.pipeline(splits[0], fold=0)
                self.save_data(data, fold=0)
                model = self.create_model(data)
                self.train(model, data, fold=0)
                self.save_model(model, fold=0)
            else:
                # for cross-validation, train all models, potentially as separate guild runs
                self.train_all()

            # evaluate on each split
            evals = []
            for k in range(len(splits)):
                data = self.load_data(fold=k)
                model = self.load_model(data, fold=k)
                evals.append(self.eval(model, data, fold=k))

            # print results
            if len(splits) > 1:
                self.print_cv_results(evals)

        elif self.operation == "embed":
            self.load_data()
            self.load_model()
            self.embed()

    def print_cv_results(self, evals: list[dict]) -> dict:
        """Print cross-validation results for evaluations on all folds (mean, std, min, max for train and test)."""
        print("\n>>> cross-validation results\n")

        keys = list(evals[0].keys())
        scalars = {}
        for key in keys:
            scalars[f"{key}_mean"] = np.mean([e[key] for e in evals])
            scalars[f"{key}_std"] = np.std([e[key] for e in evals])
            scalars[f"{key}_min"] = np.min([e[key] for e in evals])
            scalars[f"{key}_max"] = np.max([e[key] for e in evals])

        # print rounded scalars
        print_guild_scalars(**{k: f"{v:.4f}" for k, v in scalars.items()})
        return scalars

    def get_splits(self, df) -> tuple:
        """Creates or loads train/test splits. If cross_val is "none", a single split is created based on
        train.test_split and train.shuffle_split. If cross_val is the string "catalog", it
        will look up the split points from the catalog collection (currently only supported for OpenML datasets).
        If cross_val is "5-fold", it will create 5 folds of train/test.

        The splits are saved in a splits.json file.
        """
        try:
            splits = self.load_splits()
            print(f"loaded splits.json with {len(splits)} splits for cross-validation.")
            return tuple((df.iloc[train], df.iloc[test]) for train, test in splits)

        except FileNotFoundError:
            print("creating splits...", end=" ")

        if self.config.cross_val == "none":
            # no cv, simple split in train/test set
            split = train_test_split(
                range(len(df)), test_size=self.config.train.test_split, shuffle=self.config.train.shuffle_split
            )
            splits = (split,)
            self.save_splits(splits)
            print("using single train/test split.")

            return tuple((df.iloc[train], df.iloc[test]) for train, test in splits)

        # use cross validation
        elif self.config.cross_val == "catalog":
            assert self.config.data.db == "openml", "catalog splits only available for openml datasets"

            # for openml load split points from catalog
            secrets = load_secrets()
            client = MongoClient(secrets["MONGO_URI"])
            catalog = client[self.config.data.db].catalog

            splits = catalog.find_one({"_id": self.config.data.coll})["task"]["cross_validation"]

            # turn into list of tuples
            splits = [(split["train_ixs"], split["test_ixs"]) for split in splits]
            print(f"openml dataset, using {len(splits)} pre-defined splits for cross-validation.")

        elif self.config.cross_val == "5-fold":
            # otherwise use 5 random splits
            kfold = KFold(n_splits=5, shuffle=True, random_state=self.config.seed)
            splits = list(kfold.split(df))
            splits = [(train.tolist(), test.tolist()) for train, test in splits]
            print(f"using {len(splits)} random splits for cross-validation.")

        else:
            raise ValueError("invalid `cross_val` argument. Must be 'none', '5-fold', 'catalog'. ")

        self.save_splits(splits)
        return tuple((df.iloc[train], df.iloc[test]) for train, test in splits)

    def _verify_skip_identical(self):
        """if a completed run with the exact same flags already exists, skip (unless `repeat` flag is set)"""
        if self.config.get("repeat", False):
            return

        model_op = os.environ.get("GUILD_OP", "debug")
        runs = get_run_objects(
            operations=model_op, labels="temp", filter_expr=f"dataset={self.config.dataset}", completed=True
        )
        config = OmegaConf.to_container(self.config, enum_to_str=True)
        del config["data"]
        config = flatten_config(config)

        for run in runs:
            run_flags = run["flags"]
            diff = set(run_flags.items()) - set(config.items())
            if diff == set():
                raise Exception(f"run has the same config as completed run {run.short_id} -- skipping.")

    def verify(self):
        """any verification checks before starting the run. If this method returns raises an
        exception, the run is skipped."""
        self._verify_skip_identical()

    def save_data(self, data: dict, fold: int) -> None:
        """saves the data after pipeline execution for the given fold."""
        print(f"\n>>> saving dataset for fold {fold}\n")

        # pickle dataset to disk
        with open(f"dataset_fold_{fold}.pickle.gz", "wb") as f:
            pickle.dump(data, f)

    def load_data(self, fold: int) -> dict:
        """loads the processed data for the given fold."""
        print(f"\n>>> loading dataset for fold {fold}\n")
        with open(f"dataset_fold_{fold}.pickle.gz", "rb") as f:
            data = pickle.load(f)
        return data

    def save_splits(self, splits) -> None:
        """saves the splits as a json file."""
        with open("splits.json", "w") as f:
            json.dump(splits, f)

    def load_splits(self) -> tuple:
        """loads the splits from json file."""
        with open("splits.json", "r") as f:
            return json.load(f)

    #### Below methods are runner-specific and need to be implemented.

    def fetch_df(self) -> pd.DataFrame:
        """loads data from MongoDB and returns a data frame"""
        raise NotImplementedError()

    def pipeline(self, split: tuple[pd.DataFrame, pd.DataFrame], fold: int) -> dict:
        """builds and executes the pipeline for the given train/test split."""
        raise NotImplementedError()

    def train(self, model: Any, data: dict, fold: int) -> None:
        """trains a model. Needs to be implemented by runner class."""
        raise NotImplementedError()

    def train_all(self):
        """trains all models. This is left to the runner class as it may want to run the models as separate
        guild runs (docformer) or inside a single run (scikit-learn baselines)."""
        raise NotImplementedError()

    def eval(self, model: Any, data: dict, fold: int) -> dict:
        """evaluates a model. Needs to be implemented by runner class."""
        raise NotImplementedError()

    def embed(self) -> dict:
        raise NotImplementedError()

    def create_model(self, data: dict) -> Any:
        """Creates a new (untrained) model. Needs to be implemented by runner class."""
        raise NotImplementedError()

    def save_model(self, model: Any, fold: int) -> None:
        """Saves a model to disk for a given fold. Needs to be implemented by runner class."""
        raise NotImplementedError()

    def load_model(self, data: dict, fold: int) -> Any:
        """Loads a model from disk. Needs to be implemented by runner class."""
        raise NotImplementedError()
