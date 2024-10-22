import pathlib

import click
from click_option_group import optgroup
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from origami.inference import Metrics, Predictor
from origami.model import ORIGAMI
from origami.model.vpda import ObjectVPDA
from origami.preprocessing import (
    DFDataset,
    build_prediction_pipelines,
)
from origami.utils import TopLevelConfig, count_parameters, save_origami_model, make_progress_callback, set_seed

from .utils import create_projection, load_data


@click.command()
@click.argument("source", type=str)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(dir_okay=False, path_type=pathlib.Path, resolve_path=True),
    default="./model.origami",
    show_default=True,
    help="path to write trained model",
)
@click.option("--seed", type=int, default=1234, show_default=True, help="random seed")
@click.option("--verbose", "-v", is_flag=True, default=False)
@optgroup.group("Source Options")
@optgroup.option("--source-db", "-d", type=str, help="database name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--source-coll", "-c", type=str, help="collection name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--include-fields", "-i", type=str, help="comma-separated list of field names to include")
@optgroup.option("--exclude-fields", "-e", type=str, help="comma-separated list of field names to exclude")
@optgroup.option("--skip", "-s", type=int, default=0, help="number of documents to skip")
@optgroup.option("--limit", "-l", type=int, default=0, help="limit the number of documents to load")
@optgroup.group("Config Options")
@optgroup.option("--config-file", "-C", type=click.File("r"), help="path to config file")
@optgroup.option(
    "--max-vocab-size",
    "-V",
    type=int,
    default=0,
    show_default=True,
    help="maximum number of tokens in the vocabulary",
)
@optgroup.option(
    "--num-layers",
    "-L",
    type=int,
    default=4,
    show_default=True,
    help="number of transformer layers",
)
@optgroup.option(
    "--num-attn-heads",
    "-A",
    type=int,
    default=4,
    show_default=True,
    help="number of attention heads",
)
@optgroup.option(
    "--hidden-dim",
    "-H",
    type=int,
    default=128,
    show_default=True,
    help="hidden dimensionality of transformer layers",
)
@optgroup.option(
    "--num-batches", "-N", type=int, default=10000, show_default=True, help="number of batches to train on"
)
@optgroup.option("--batch-size", "-B", type=int, default=100, show_default=True, help="batch size")
@optgroup.option(
    "--pos-encoding",
    "-P",
    type=click.Choice(["NONE", "INTEGER", "KEY_VALUE"]),
    help="type of position encoding",
    default="KEY_VALUE",
    show_default=True,
)
@optgroup.option(
    "--upscaling",
    "-U",
    type=int,
    help="upscaling factor, when `--shuffled` mode is used",
    default=5,
    show_default=True,
)
@optgroup.option(
    "--shuffled/--ordered",
    "shuffled",
    is_flag=True,
    default=True,
    help="shuffle key/value pairs in each object",
    show_default=True,
)
@optgroup.group("Evaluation Options")
@optgroup.option(
    "--val-split-ratio",
    "-r",
    type=float,
    default=0.0,
    show_default=True,
    help="ratio for validation dataset, a value of 0.0 disables validation",
)
@optgroup.option("--target-field", "-t", type=str, help="target field name to predict")
def train(source: str, **kwargs):
    """
    Train an ORIGAMI model.
    """
    set_seed(kwargs["seed"])

    config = TopLevelConfig()

    # data configs
    config.data.source = source
    config.data.db = kwargs["source_db"]
    config.data.coll = kwargs["source_coll"]
    config.data.projection = create_projection(kwargs["include_fields"], kwargs["exclude_fields"])
    config.data.skip = kwargs["skip"]
    config.data.limit = kwargs["limit"]
    config.data.target_field = kwargs["target_field"]

    # model configs
    config.model.n_layer = kwargs["num_layers"]
    config.model.n_head = kwargs["num_attn_heads"]
    config.model.n_embd = kwargs["hidden_dim"]
    config.model.position_encoding = kwargs["pos_encoding"]

    # train configs
    config.train.n_batches = kwargs["num_batches"]
    config.train.batch_size = kwargs["batch_size"]
    config.train.learning_rate = 1e-3
    config.train.n_warmup_batches = 1000
    config.train.print_every = 10
    config.train.eval_every = 100
    config.train.test_split = kwargs["val_split_ratio"]

    # pipeline configs
    config.pipeline.max_vocab_size = kwargs["max_vocab_size"]
    config.pipeline.sequence_order = "SHUFFLED" if kwargs["shuffled"] else "ORDERED"
    config.pipeline.upscale = kwargs["upscaling"]

    # load data
    df = load_data(source, config.data)

    if config.train.test_split > 0:
        train_df, test_df = train_test_split(df, test_size=config.train.test_split, shuffle=config.train.shuffle_split)
    else:
        train_df = df
        test_df = None

    # build pipelines
    pipelines = build_prediction_pipelines(config.pipeline, config.data.target_field)

    # process train and test data
    train_proc_df = pipelines["train"].fit_transform(train_df)
    train_dataset = DFDataset(train_proc_df)

    if config.train.test_split > 0:
        test_proc_df = pipelines["test"].transform(test_df)
        test_dataset = DFDataset(test_proc_df)
    else:
        test_dataset = None

    # get stateful objects and set model parameters
    schema = pipelines["train"]["schema"].schema
    encoder = pipelines["train"]["encoder"].encoder
    config.model.block_size = pipelines["train"]["padding"].length
    config.model.vocab_size = encoder.vocab_size

    # create model with PDA
    vpda = ObjectVPDA(encoder, schema)
    model = ORIGAMI(config.model, config.train, vpda=vpda)

    if kwargs["verbose"]:
        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = count_parameters(model)
        click.echo(f"running on device: {model.device}")
        click.echo(f"number of parameters: {n_params/1e6:.2f}M")
        click.echo(f"config:\n {OmegaConf.to_yaml(config)}")

    # model callback during training, prints training and test metrics
    if config.data.target_field:
        predictor = Predictor(model, encoder, config.data.target_field, max_batch_size=config.train.batch_size)
    else:
        predictor = Metrics(model, batch_size=config.train.batch_size)
    progress_callback = make_progress_callback(
        config.train, train_dataset=train_dataset, test_dataset=test_dataset, predictor=predictor
    )

    # train model
    model.set_callback("on_batch_end", progress_callback)
    model.train_model(train_dataset, batches=config.train.n_batches)

    # save model with config
    save_origami_model(model, pipelines=pipelines, config=config, path=kwargs.get("model_path"))
