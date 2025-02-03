from types import SimpleNamespace

import numpy as np
from pymongo import MongoClient
from sklearn.pipeline import Pipeline

from origami.inference import AutoCompleter, Metrics
from origami.model import ORIGAMI
from origami.model.vpda import ObjectVPDA
from origami.preprocessing import (
    DFDataset,
    DocPermuterPipe,
    DocTokenizerPipe,
    KBinsDiscretizerPipe,
    PadTruncTokensPipe,
    TargetFieldPipe,
    TokenEncoderPipe,
    UpscalerPipe,
    load_df_from_mongodb,
)
from origami.utils.common import set_seed
from origami.utils.config import ModelConfig, PositionEncodingMethod, TrainConfig
from origami.utils.guild import load_secrets, print_guild_scalars

flags = SimpleNamespace()

secrets = load_secrets()

set_seed(flags.seed)

# load PATHOLOGY fields for test data
client = MongoClient(secrets["MONGO_URI"])
collection_test = client.ddxplus[f"{flags.eval_data}-noprob"]

pathologies_test = [
    d["PATHOLOGY"] for d in collection_test.find({}, projection={"PATHOLOGY": 1}, limit=flags.limit, sort=[("_id", 1)])
]

# now load data for training and evaluation (test or validate), same sort order
PROJECTION = {"_id": 0, "PATHOLOGY": 0, "DIFFERENTIAL_DIAGNOSIS": 0}
TARGET_FIELD = "DIFFERENTIAL_DIAGNOSIS_NOPROB"

train_docs_df = load_df_from_mongodb(
    secrets["MONGO_URI"], "ddxplus", "train-noprob", projection=PROJECTION, limit=flags.limit, sort=[("_id", 1)]
)

test_docs_df = load_df_from_mongodb(
    secrets["MONGO_URI"],
    "ddxplus",
    f"{flags.eval_data}-noprob",
    projection=PROJECTION,
    limit=flags.limit,
    sort=[("_id", 1)],
)

# create train and test pipelines
pipes = {
    # --- train only ---
    "upscaler": UpscalerPipe(n=2),
    "permuter": DocPermuterPipe(),
    # --- test only ---
    "target": TargetFieldPipe(TARGET_FIELD),
    # --- train and test ---
    "discretizer": KBinsDiscretizerPipe(bins=128, threshold=128, strategy="kmeans"),
    "tokenizer": DocTokenizerPipe(),
    "padding": PadTruncTokensPipe(length="max"),
    "encoder": TokenEncoderPipe(),
}

train_pipeline = Pipeline(
    [(name, pipes[name]) for name in ("discretizer", "upscaler", "permuter", "tokenizer", "padding", "encoder")]
)
test_pipeline = Pipeline([(name, pipes[name]) for name in ("discretizer", "target", "tokenizer", "padding", "encoder")])

# process train and test/validation data
train_pipeline.fit(train_docs_df)
test_pipeline.fit(test_docs_df)
train_df = train_pipeline.transform(train_docs_df)
test_df = test_pipeline.transform(test_docs_df)

# get stateful objects
encoder = pipes["encoder"].encoder
block_size = pipes["padding"].length

# print data stats
print(f"len train: {len(train_df)}, len val: {len(test_df)}")
print(f"vocab size {encoder.vocab_size}")
print(f"block size {block_size}")

# wrap in datasets
train_dataset = DFDataset(train_df)
test_dataset = DFDataset(test_df)

# model and train configs
model_config = ModelConfig.from_preset(flags.model_size)
model_config.position_encoding = PositionEncodingMethod.KEY_VALUE
model_config.vocab_size = encoder.vocab_size
model_config.block_size = block_size
model_config.mask_field_token_losses = True

train_config = TrainConfig()

vpda = ObjectVPDA(encoder)  # build VPDA without schema (only doc structure enforced)
model = ORIGAMI(model_config, train_config, vpda=vpda)

metrics = Metrics(model)


def progress_callback(model):
    global test_dataset
    if model.batch_num % train_config.eval_every == 0:
        print_guild_scalars(
            step=f"{int(model.batch_num / train_config.eval_every)}",
            epoch=model.epoch_num,
            batch_num=model.batch_num,
            batch_dt=f"{model.batch_dt * 1000:.2f}",
            batch_loss=f"{model.loss:.4f}",
            lr=f"{model.learning_rate:.2e}",
        )


model.set_callback("on_batch_end", progress_callback)
model.train_model(train_dataset, batches=flags.n_batches)
model.save("gpt_checkpoint.pt")

# --- evaluation ---

# generation is faster on cpu
model.device = "cpu"

# optionally evaluate on a smaller subset of the test data
# test_dataset = test_dataset.sample(n=10000)
autocompleter = AutoCompleter(model, encoder, target_field=TARGET_FIELD, max_batch_size=5000, show_progress=False)
completions = autocompleter.autocomplete(test_dataset, decode=True)

df = test_dataset.df
df["generated"] = completions
df["pathology"] = np.array(pathologies_test)
df["predicted"] = [c[TARGET_FIELD] for c in completions]


def get_ddx_arr(ddx_arr):
    if not isinstance(ddx_arr, list):
        # if model doesn't predict an array, this can happen
        # we return an empty list, which will lead to prec = rec = 0
        return []

    if TARGET_FIELD.endswith("_NOPROB"):
        return ddx_arr

    # only return the diagnosis name, not the probability
    return [a[0] for a in ddx_arr]


ddr = []
ddp = []
gtpa_at_1 = []
gtpa = []

for i, row in df.iterrows():
    y_true = get_ddx_arr(row["target"])
    y_pred = get_ddx_arr(row["predicted"])

    intersection = set(y_true).intersection(set(y_pred))
    ddr.append(len(intersection) / len(y_true))
    ddp.append(len(intersection) / len(y_pred) if len(y_pred) > 0 else 0)

    # is pathology the top diagnosis?
    gtpa_at_1.append(int(len(y_pred) > 0 and row["pathology"] == y_pred[0]))

    # is pathology one of the predicted diagnoses?
    gtpa.append(int(row["pathology"] in y_pred))

ddr = np.mean(ddr)
ddp = np.mean(ddp)
f1 = 2 * ddr * ddp / (ddr + ddp)
gtpa_at_1 = np.mean(gtpa_at_1)
gtpa = np.mean(gtpa)

print(f"\n Evaluation result for {flags.eval_data} dataset")
print_guild_scalars(ddr=ddr, ddp=ddp, f1=f1, gtpa_at_1=gtpa_at_1, gtpa=gtpa)
