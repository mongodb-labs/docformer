from pathlib import Path
from types import SimpleNamespace

from pymongo import MongoClient
from sklearn.pipeline import Pipeline

from origami.inference import Predictor
from origami.model import ORIGAMI
from origami.model.vpda import ObjectVPDA
from origami.preprocessing import (
    DFDataset,
    DocPermuterPipe,
    DocTokenizerPipe,
    PadTruncTokensPipe,
    TargetFieldPipe,
    TokenEncoderPipe,
    UpscalerPipe,
    load_df_from_mongodb,
)
from origami.utils.common import set_seed
from origami.utils.config import GuardrailsMethod, ModelConfig, PositionEncodingMethod, TrainConfig
from origami.utils.guild import load_secrets, print_guild_scalars

# populated by guild
flags = SimpleNamespace()
secrets = load_secrets()

# for reproducibility
set_seed(1234)

TARGET_FIELD = "problem"
UPSCALE = 2

client = MongoClient(secrets["MONGO_URI"])
collection = client["codenet_java"].train

target_problems = collection.distinct(TARGET_FIELD)
num_problems = len(target_problems)

target_problems = target_problems[: flags.n_problems]
print(f"training on {flags.n_problems} problems (out of {num_problems})")

# load data into dataframe for train/test

train_docs_df = load_df_from_mongodb(
    "mongodb://localhost:27017",
    "codenet_java",
    "train",
    filter={"problem": {"$in": target_problems}},
    projection={"_id": 0, "filePath": 0},
)

test_docs_df = load_df_from_mongodb(
    "mongodb://localhost:27017",
    "codenet_java",
    "test",
    filter={"problem": {"$in": target_problems}},
    projection={"_id": 0, "filePath": 0},
)

num_train_inst = len(train_docs_df)
num_test_inst = len(test_docs_df)

# create train and test pipelines
pipes = {
    # --- train only ---
    "upscaler": UpscalerPipe(n=UPSCALE),
    "permuter": DocPermuterPipe(shuffle_arrays=True),
    # --- test only ---
    "target": TargetFieldPipe(TARGET_FIELD),
    # --- train and test ---
    "tokenizer": DocTokenizerPipe(path_in_field_tokens=False),
    "padding": PadTruncTokensPipe(length=flags.max_length),
    "encoder": TokenEncoderPipe(max_tokens=flags.max_tokens),
}

train_pipeline = Pipeline(
    [(name, pipes[name]) for name in ("target", "upscaler", "permuter", "tokenizer", "padding", "encoder")],
    verbose=True,
)
test_pipeline = Pipeline([(name, pipes[name]) for name in ("target", "tokenizer", "padding", "encoder")], verbose=True)

# process train, eval and test data (first fit both, then transform)
train_pipeline.fit(train_docs_df)
test_pipeline.fit(test_docs_df)

train_df = train_pipeline.transform(train_docs_df)
test_df = test_pipeline.transform(test_docs_df)

# drop ordered_docs columns to save space
train_df.drop(columns=["docs"], inplace=True)
test_df.drop(columns=["docs"], inplace=True)

# drop all rows where the tokens array doesn't end in 0 (longer than max_length)
train_df = train_df[train_df["tokens"].apply(lambda x: x[-1] == 0)]
test_df = test_df[test_df["tokens"].apply(lambda x: x[-1] == 0)]

# get stateful objects
encoder = pipes["encoder"].encoder
block_size = pipes["padding"].length

# print data stats
print(
    f"dropped {(1 - (len(train_df) / (UPSCALE * num_train_inst))) * 100:.2f}% training instances, and "
    f"{(1 - (len(test_df) / num_test_inst)) * 100:.2f}% test instances."
)
print(f"vocab size {encoder.vocab_size}")
print(f"block size {block_size}")

# confirm that all targets are in the vocabulary
for target in train_df["target"].unique():
    enc = encoder.encode(target)
    assert target == encoder.decode(enc), f"token not {target} represented in vocab."

for target in test_df["target"].unique():
    enc = encoder.encode(target)
    assert target == encoder.decode(enc), f"token not {target} represented in vocab."

# create datasets, VPDA and model

# model and train configs
model_config = ModelConfig.from_preset("small")
model_config.position_encoding = PositionEncodingMethod.KEY_VALUE
model_config.vocab_size = encoder.vocab_size
model_config.block_size = block_size
model_config.n_embd = flags.n_embd
model_config.mask_field_token_losses = False
model_config.tie_weights = False
model_config.guardrails = GuardrailsMethod.STRUCTURE_ONLY
model_config.fuse_pos_with_mlp = True

train_config = TrainConfig()
train_config.learning_rate = flags.learning_rate
train_config.batch_size = flags.batch_size
train_config.n_warmup_batches = 100
train_config.eval_every = flags.eval_every

# datasets
train_dataset = DFDataset(train_df)
test_dataset = DFDataset(test_df)

vpda = ObjectVPDA(encoder)
model = ORIGAMI(model_config, train_config, vpda=vpda)

# load model checkpoint if it exists
checkpoint_file = Path("./gpt-codenet-snapshot.pt")
if checkpoint_file.is_file():
    model.load("gpt-codenet-snapshot.pt")
    print(f"loading existing checkpoint at batch_num {model.batch_num}...")


# create a predictor
predictor = Predictor(model, encoder, TARGET_FIELD)


def progress_callback(model):
    print_guild_scalars(
        step=f"{int(model.batch_num)}",
        epoch=model.epoch_num,
        batch_num=model.batch_num,
        batch_dt=f"{model.batch_dt * 1000:.2f}",
        batch_loss=f"{model.loss:.4f}",
        lr=f"{model.learning_rate:.2e}",
    )
    if model.batch_num % train_config.eval_every == 0:
        try:
            # train_acc = predictor.accuracy(train_dataset.sample(n=100))
            test_acc = predictor.accuracy(test_dataset.sample(n=100), show_progress=True)
            print_guild_scalars(
                step=f"{int(model.batch_num)}",
                # train_acc=f"{train_acc:.4f}",
                test_acc=f"{test_acc:.4f}",
            )
            # print(f"Train accuracy @ 100: {train_acc:.4f}, Test accuracy @ 100: {test_acc:.4f}")
        except AssertionError as e:
            print(e)
            print("continuing...")

        model.save("gpt-codenet-snapshot.pt")
        print("model saved to gpt-codenet-snapshot.pt")


model.set_callback("on_batch_end", progress_callback)

try:
    model.train_model(train_dataset, batches=flags.n_batches)
except KeyboardInterrupt:
    pass

# final save
model.save("gpt-codenet-snapshot.pt")
print("model saved to gpt-codenet-snapshot.pt")

test_acc = predictor.accuracy(test_dataset, show_progress=True)
print_guild_scalars(
    step=f"{int(model.batch_num / train_config.eval_every)}",
    test_acc=f"{test_acc:.4f}",
)

dropped_ratio = 1 - (len(test_df) / num_test_inst)
print(f"Final test accuracy when taking into account the dropped instances: {(1 - dropped_ratio) * test_acc:.4f}%")
