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
from origami.utils import TopLevelConfig, count_parameters, make_progress_callback, save_origami_model, set_seed
from origami.utils.config import GuardrailsMethod, SequenceOrderMethod

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
@click.option(
    "--set-parameter",
    "-p",
    type=str,
    multiple=True,
    help="set additional config parameters, format: key.subkey=value. Multiple parameters can be set.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
@optgroup.group("Source Options")
@optgroup.option("--source-db", "-d", type=str, help="database name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--source-coll", "-c", type=str, help="collection name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--include-fields", "-i", type=str, help="comma-separated list of field names to include")
@optgroup.option("--exclude-fields", "-e", type=str, help="comma-separated list of field names to exclude")
@optgroup.option("--skip", "-s", type=int, default=0, help="number of documents to skip")
@optgroup.option("--limit", "-l", type=int, default=0, help="limit the number of documents to load")
@optgroup.group("Config Options")
# @optgroup.option("--config-file", "-C", type=click.File("r"), help="path to config file")
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
    "-T",
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
    "--learning-rate",
    "-L",
    type=float,
    default=1e-3,
    show_default=True,
    help="max. learning rate of the model",
)
@optgroup.option(
    "--num-batches", "-N", type=int, default=10000, show_default=True, help="number of batches to train on"
)
@optgroup.option("--batch-size", "-B", type=int, default=100, show_default=True, help="batch size")
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
@optgroup.option(
    "--guardrails",
    "-G",
    type=click.Choice(["NONE", "STRUCTURE_ONLY", "STRUCTURE_AND_VALUES"]),
    default="STRUCTURE_ONLY",
    help="guardrails settings",
)
@optgroup.option(
    "--ignore-field-path",
    is_flag=True,
    default=False,
    help="By default, the full field path (i.e. `foo.bar.baz`) is used for field tokens. This option will ignore the full path and only use the last field name (`bar`). Use this for deeply nested object.",
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
@optgroup.option("--target-field", "-t", type=str, help="target field to predict")
def train(source: str, **kwargs):
    """
    Train an ORIGAMI model.
    """
    set_seed(kwargs["seed"])

    config = TopLevelConfig()

    # data configs
    config.data.source = source

    if source.startswith("mongodb://"):
        if kwargs["source_db"] is None or kwargs["source_coll"] is None:
            raise click.BadParameter("--source-db and --source-coll are required for MongoDB URI")

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
    config.model.guardrails = kwargs["guardrails"]

    # train configs
    config.train.n_batches = kwargs["num_batches"]
    config.train.batch_size = kwargs["batch_size"]
    config.train.learning_rate = kwargs["learning_rate"]
    config.train.print_every = 10
    config.train.eval_every = 100
    config.train.test_split = kwargs["val_split_ratio"]

    # as guideline, use 5 x batch size for evaluation of train/test metrics
    config.train.sample_train = 5 * kwargs["batch_size"]
    config.train.sample_test = 5 * kwargs["batch_size"]

    # pipeline configs
    config.pipeline.max_vocab_size = kwargs["max_vocab_size"]
    config.pipeline.sequence_order = "SHUFFLED" if kwargs["shuffled"] else "ORDERED"
    config.pipeline.upscale = kwargs["upscaling"]
    config.pipeline.path_in_field_tokens = not kwargs["ignore_field_path"]
    if not config.pipeline.path_in_field_tokens:
        if config.model.guardrails == GuardrailsMethod.STRUCTURE_AND_VALUES:
            click.echo(
                click.style(
                    f"info: `Can't use guardrails={config.model.guardrails}` when `pipeline.path_in_field_tokens` is `False`. Setting to `STRUCTURE_ONLY`.",
                    fg="green",
                )
            )
            config.model.guardrails = "STRUCTURE_ONLY"

    if config.pipeline.sequence_order == SequenceOrderMethod.ORDERED and config.pipeline.upscale > 1:
        click.echo(
            click.style(
                f"info: `pipeline.upscale={config.pipeline.upscale}` is ignored when `pipeline.sequence_order` is `ORDERED`.",
                fg="green",
            )
        )

    # set custom parameters (overrides other settings)
    for p in kwargs["set_parameter"]:
        key, value = p.split("=")
        key, subkey = key.split(".")
        config[key][subkey] = value

    # load data
    df = load_data(source, config.data)

    if config.train.test_split > 0:
        train_df, test_df = train_test_split(df, test_size=config.train.test_split, shuffle=config.train.shuffle_split)
    else:
        train_df = df
        test_df = None

    # build pipelines
    pipelines = build_prediction_pipelines(config.pipeline, config.data.target_field, verbose=kwargs["verbose"])

    # process train and test data
    train_proc_df = pipelines["train"].fit_transform(train_df)
    train_dataset = DFDataset(train_proc_df)

    # limit the number of training examples if dataset is too small
    config.train.sample_train = min(len(train_dataset), config.train.sample_train)

    if config.train.test_split > 0:
        test_proc_df = pipelines["test"].transform(test_df)
        test_dataset = DFDataset(test_proc_df)
        # limit the number of test examples if dataset is too small
        config.train.sample_test = min(len(test_dataset), config.train.sample_test)

        # for guardrails mode STRUCTURE_AND_VALUES, this is needed to allow transitions
        # in vpda to work for test data
        pipelines["train"]["schema"].fit(test_df)

    else:
        test_dataset = None

    # get stateful objects and set model parameters
    schema = pipelines["train"]["schema"].schema
    encoder = pipelines["train"]["encoder"].encoder
    config.model.block_size = pipelines["train"]["padding"].length
    config.model.vocab_size = encoder.vocab_size

    # warn about unique and high-cardinality keys in schema
    total_count = schema.count
    for path, field in schema.leaf_fields.items():
        prim_values = schema.get_prim_values(path)
        ratio = len(prim_values) / total_count
        if ratio == 1:
            click.echo(
                click.style(
                    f"warning: field `{path}` has only unique values and should be excluded with --exclude-fields",
                    fg="red",
                )
            )
        elif ratio > 0.5:
            click.echo(
                click.style(
                    f"warning: field `{path}` is high cardinality with {int(ratio * 100)}% unique values. Consider excluding it with --exclude-fields",
                    fg="yellow",
                )
            )

    # create model with PDA
    if kwargs["verbose"]:
        click.echo(">>> creating PDA rules\n")

    match config.model.guardrails:
        case GuardrailsMethod.STRUCTURE_AND_VALUES:
            if not config.pipeline.path_in_field_tokens:
                raise Exception("GuardrailsMethod.STRUCTURE_AND_VALUES requires path_in_field_tokens=True")
            vpda = ObjectVPDA(encoder, schema)
        case GuardrailsMethod.STRUCTURE_ONLY | GuardrailsMethod.NONE:
            vpda = ObjectVPDA(encoder)

    if kwargs["verbose"]:
        click.echo(">>> creating ORiGAMi model\n")
    model = ORIGAMI(config.model, config.train, vpda=vpda)

    if kwargs["verbose"]:
        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = count_parameters(model)
        click.echo(f"running on device: {model.device}")
        click.echo(f"number of parameters: {n_params / 1e6:.2f}M")
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
