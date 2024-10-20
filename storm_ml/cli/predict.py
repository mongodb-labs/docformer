import json
import pathlib

import click
from click_option_group import optgroup

from storm_ml.inference import Predictor
from storm_ml.model import STORM
from storm_ml.model.vpda import DocumentVPDA
from storm_ml.preprocessing import DFDataset, TargetFieldPipe
from storm_ml.utils import Symbol, load_storm_model

from .utils import create_projection, load_data


@click.command()
@click.argument("source", type=str)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path, resolve_path=True),
    default="./model.storm",
    show_default=True,
    help="path to trained model",
)
@click.option("--target-field", "-t", type=str, required=True, help="target field name to predict")
@optgroup.group("Source Options")
@optgroup.option("--source-db", "-d", type=str, help="database name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--source-coll", "-c", type=str, help="collection name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--include-fields", "-i", type=str, help="comma-separated list of field names to include")
@optgroup.option("--exclude-fields", "-e", type=str, help="comma-separated list of field names to exclude")
@optgroup.option("--skip", "-s", type=int, default=0, help="number of documents to skip")
@optgroup.option("--limit", "-l", type=int, default=0, help="limit the number of documents to load")
@optgroup.group("Output Options")
@optgroup.option("--json", "-j", is_flag=True, default=False, help="output full JSON objects including target field")
@click.option("--verbose", "-v", is_flag=True, default=True)
def predict(source, **kwargs):
    """Predict target fields with a trained STORM model."""

    # load model, config and pipelines
    model_dict = load_storm_model(kwargs.get("model_path"))
    state_dict = model_dict["state_dict"]
    config = model_dict["config"]
    pipelines = model_dict["pipelines"]
    encoder = pipelines["train"]["encoder"].encoder
    schema = pipelines["train"]["schema"].schema

    # create model
    vpda = DocumentVPDA(encoder, schema)
    model = STORM(config.model, config.train, vpda=vpda)
    model.load_state_dict(state_dict)

    # data configs
    config.data.source = source
    config.data.db = kwargs["source_db"]
    config.data.coll = kwargs["source_coll"]
    config.data.projection = create_projection(kwargs["include_fields"], kwargs["exclude_fields"])
    config.data.limit = kwargs["limit"]
    config.data.target_field = kwargs["target_field"]

    # load data
    df = load_data(source, config.data)

    # update or create new target pipe with new target_field
    test_pipeline = pipelines["test"]

    if "target" in test_pipeline:
        test_pipeline["target"].target_field = config.data.target_field
    else:
        test_pipeline.steps.insert(0, ["target", TargetFieldPipe(config.data.target_field)])
    test_df = test_pipeline.transform(df)

    # datasets
    test_dataset = DFDataset(test_df)

    # predict target field
    predictor = Predictor(model, encoder, config.data.target_field)

    predictions = predictor.predict(test_dataset)

    for i, (pred, doc, target) in enumerate(zip(predictions, test_dataset.df["docs"], test_dataset.df["target"])):
        if kwargs["json"]:
            doc[config.data.target_field] = pred
            line_str = json.dumps(doc)
        else:
            line_str = pred
        if target == Symbol.UNKNOWN:
            click.echo(line_str)
        else:
            if pred == target:
                # replace with green version
                click.echo(f"\033[32m{line_str}\033[0m")
            else:
                # replace with red version
                click.echo(f"\033[31m{line_str}\033[0m")

    # if any target field is not None, print accuracy
    if any(test_dataset.df["target"] != Symbol.UNKNOWN):
        acc = predictor.accuracy(test_dataset, show_progress=False)
        click.echo(f"\nTest accuracy: {acc:.4f}", err=True)
