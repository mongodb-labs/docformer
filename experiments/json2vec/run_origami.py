import os
import shutil
import time
from pathlib import Path
from typing import Any

import guild._api as gapi
import pandas as pd
from omegaconf import OmegaConf
from runner import BaseRunner

from origami.inference import Predictor
from origami.model import ORIGAMI
from origami.model.vpda import ObjectVPDA
from origami.preprocessing import DFDataset, build_prediction_pipelines, load_df_from_mongodb
from origami.utils import make_progress_callback, walk_all_leaf_kvs
from origami.utils.config import GuardrailsMethod, SequenceOrderMethod
from origami.utils.guild import get_run_path, load_secrets, print_guild_scalars


class ORIGAMIRunner(BaseRunner):
    def _verify_skip_ordered_upscale(self) -> None:
        if self.config.pipeline.sequence_order == SequenceOrderMethod.ORDERED and self.config.pipeline.upscale > 1:
            raise Exception("pipeline.sequence_order=ORDERED with pipeline.upscale > 1 not supported -- skipping.")

    def verify(self) -> None:
        self._verify_skip_ordered_upscale()
        self._verify_skip_identical()

    def fetch_df(self):
        print("\n>>> fetching data\n")
        secrets = load_secrets()

        # load data into dataframe and split into train/test
        df = load_df_from_mongodb(
            secrets["MONGO_URI"],
            self.config.data.db,
            self.config.data.coll,
            limit=self.config.data.limit,
            projection=self.config.data.projection,
        )

        return df

    def pipeline(self, split: tuple[pd.DataFrame, pd.DataFrame], fold: int) -> dict:
        """builds the pipeline for the given train/test split."""
        print(f"\n>>> executing pipeline for fold {fold}\n")

        train_df, test_df = split

        # create pipelines according to config
        pipelines = build_prediction_pipelines(
            self.config.pipeline, target_field=self.config.data.target_field, verbose=True
        )

        # we fit first, because train_df can get modified in transform
        # pipelines['train'].fit(train_df)

        # the train_eval dataset is the train dataset but run through the test pipeline, i.e. not shuffled/upscaled
        # to get a proper "train" accuracy, we use train_eval instead of train
        train_proc_df = pipelines["train"].fit_transform(train_df)
        test_proc_df = pipelines["test"].transform(test_df)
        train_eval_proc_df = pipelines["test"].transform(train_df)

        # this is needed to allow transitions in vpda to work for test data during evaluation
        pipelines["train"]["schema"].fit(test_proc_df)

        # datasets
        train_dataset = DFDataset(train_proc_df)
        train_eval_dataset = DFDataset(train_eval_proc_df)
        test_dataset = DFDataset(test_proc_df)

        # get stateful objects
        schema = pipelines["train"]["schema"].schema
        encoder = pipelines["train"]["encoder"].encoder
        block_size = pipelines["train"]["padding"].length

        # print data stats
        print()
        print(f"train size: {len(train_proc_df)}, eval size: {len(train_eval_proc_df)}, test size: {len(test_proc_df)}")
        print(f"vocab size {encoder.vocab_size}")
        print(f"block size {block_size}")

        print("field cardinalities:")
        for field in schema.leaf_fields:
            print(f"  {field}: {len(schema.get_prim_values(field))}")

        return {
            "train": train_dataset,
            "train_eval": train_eval_dataset,
            "test": test_dataset,
            "schema": schema,
            "encoder": encoder,
            "block_size": block_size,
        }

    def create_model(self, data: dict) -> Any:
        print("\n>>> creating model\n")

        # unpack data objects
        schema = data["schema"]
        encoder = data["encoder"]
        block_size = data["block_size"]

        # fill model config
        self.config.model.vocab_size = encoder.vocab_size
        self.config.model.block_size = block_size

        # schema can only be used in VPDA if the full path is encoded in field tokens

        match self.config.model.guardrails:
            case GuardrailsMethod.STRUCTURE_AND_VALUES:
                if not self.config.pipeline.path_in_field_tokens:
                    raise Exception("GuardrailsMethod.STRUCTURE_AND_VALUES requires path_in_field_tokens=True")
                vpda = ObjectVPDA(encoder, schema)
            case GuardrailsMethod.STRUCTURE_ONLY | GuardrailsMethod.NONE:
                vpda = ObjectVPDA(encoder)

        return ORIGAMI(self.config.model, self.config.train, vpda=vpda)

    def train(self, model, data: dict, fold: int) -> None:
        print(f"\n>>> training model for fold {fold}\n")

        # reduce evaluation sample sizes if dataset is too small
        if self.config.train.sample_test > len(data["test"]):
            self.config.train.sample_test = len(data["test"])
            print(f"reducing config.train.sample_test to {len(data['test'])}")
        if self.config.train.sample_train > len(data["train_eval"]):
            self.config.train.sample_train = len(data["train_eval"])
            print(f"reducing config.train.sample_train to {len(data['train_eval'])}")

        # create a predictor
        predictor = Predictor(model, data["encoder"], self.config.data.target_field)

        # make process callback
        progress_callback = make_progress_callback(self.config.train, data["train_eval"], data["test"], predictor)

        model.set_callback("on_batch_end", progress_callback)
        model.train_model(data["train"], batches=self.config.train.n_batches)

    def train_all(self):
        print("\n>>> training models on all folds\n")

        df = self.fetch_df()
        splits = self.get_splits(df)
        # load flags and flatten fields (needed for gapi.run())
        flags = OmegaConf.load("flags.yml")
        flags = {
            item["path"]: item["value"] for item in walk_all_leaf_kvs(OmegaConf.to_container(flags, enum_to_str=True))
        }
        run_id = os.environ["RUN_ID"]
        # train a model for each split as separate guild run
        for k, split in enumerate(splits):
            data = self.pipeline(split, fold=k)
            self.save_data(data, fold=k)
            model = os.environ["GUILD_OP"].split(":")[0]
            # set the current fold, and data dependency to this "all" run id
            flags["fold"] = k
            flags[f"{model}:data"] = run_id
            label = f"train-for-run-{run_id[:8]}-fold-{k}"
            gapi.run(
                f"{model}:train",
                flags=flags,
                label=label,
                cwd=os.environ["PROJECT_DIR"],
            )
            model_path = get_run_path(labels=[label])
            model_path = Path.joinpath(model_path, f"origami_checkpoint_fold_{k}.pt")

            # wait until file was created
            while not model_path.exists():
                print(f"model file {model_path} does not exist yet, waiting 5 seconds...")
                time.sleep(5)
            print(f"copying model file {model_path} to parent guild directory.")
            shutil.copy(model_path, Path("."))

    def eval(self, model, data: dict, fold: int) -> None:
        print(f"\n>>> evaluating model for fold {fold}\n")
        # create a predictor
        predictor = Predictor(model, data["encoder"], self.config.data.target_field)
        train_acc = predictor.accuracy(data["train_eval"])
        test_acc = predictor.accuracy(data["test"], print_predictions=True)
        print_guild_scalars(
            fold=fold,
            train_acc=f"{train_acc:.4f}",
            test_acc=f"{test_acc:.4f}",
        )
        return {"train_acc": train_acc, "test_acc": test_acc}

    def save_model(self, model: Any, fold: int) -> None:
        print(f"\n>>> saving model for fold {fold}\n")
        model.save(f"./origami_checkpoint_fold_{fold}.pt")

    def load_model(self, data: dict, fold: int) -> Any:
        print(f"\n>>> loading model for fold {fold}\n")
        model = self.create_model(data)
        model.load(f"./origami_checkpoint_fold_{fold}.pt")
        return model


if __name__ == "__main__":
    args, config = ORIGAMIRunner.load_args_and_config()

    runner = ORIGAMIRunner(config, args.operation)
    runner.run()
