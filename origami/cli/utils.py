import json
import pathlib
from typing import Optional

import click
import pandas as pd
from omegaconf import OmegaConf

from origami.preprocessing import docs_to_df, load_df_from_mongodb
from origami.utils import DataConfig


def create_projection(include_fields: Optional[str] = None, exclude_fields: Optional[str] = None) -> dict:
    """
    Create a MongoDB projection dict from CLI options.

    If `include_fields` is given, it should be a comma-separated list of field names to include in the projection.
    If `exclude_fields` is given, it should be a comma-separated list of field names to exclude from the projection.

    If both `include_fields` and `exclude_fields` are given, a `ValueError` is raised.

    The returned dict can be used as the `projection` argument to `pymongo.collection.Collection.find`.
    """
    projection = {}

    if include_fields and exclude_fields:
        raise ValueError("Cannot specify both --include-fields and --exclude-fields")

    if include_fields:
        projection.update({field.strip(): 1 for field in include_fields.split(",")})

    if exclude_fields:
        projection.update({field.strip(): 0 for field in exclude_fields.split(",")})

    return projection


def filter_data(data: list[dict], projection: dict) -> list[dict]:
    """
    Filter the given data by the given MongoDB projection.

    If the projection is empty, the original data is returned unchanged.

    If the projection contains fields with value 1, only these fields are kept in the filtered data.
    If the projection contains fields with value 0, all fields except these are kept in the filtered data.

    Args:
        data (list[dict]): The data to filter.
        projection (dict): The MongoDB projection to apply to the data.

    Returns:
        list[dict]: The filtered data.
    """
    if not projection:
        return data

    projection_values = set(projection.values())
    if len(projection_values) != 1 or list(projection_values)[0] not in {0, 1}:
        raise ValueError("Invalid projection: all values must be either 0 or 1")

    keep_fields = list(projection_values)[0] == 1
    filtered_data = []

    for document in data:
        if keep_fields:
            filtered_doc = {k: v for k, v in document.items() if k in projection}
        else:
            filtered_doc = {k: v for k, v in document.items() if k not in projection}
        filtered_data.append(filtered_doc)

    return filtered_data


def load_data(source: str, data_config: DataConfig) -> pd.DataFrame:
    """
    Load data from various sources such as MongoDB, JSON, CSV, or a directory containing supported files.

    Parameters:
    - source (str): The data source, which can be a MongoDB URI, JSON file, JSONL file, CSV file, or a directory.
    - **kwargs: Additional keyword arguments for customization.

    Returns:
    - pandas.DataFrame: The loaded data as a DataFrame.
    """
    if source.startswith("mongodb://"):
        # load data from MongoDB, project out _id field by default
        projection = {"_id": 0} | OmegaConf.to_object(data_config.projection)
        df = load_df_from_mongodb(
            source,
            data_config.db,
            data_config.coll,
            limit=data_config.limit,
            skip=data_config.skip,
            projection=projection,
        )

    elif source.endswith(".json") or source.endswith(".jsonl"):
        try:
            with open(source, "r") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            with open(source, "r") as f:
                # try loading jsonl-style line by line
                data = [json.loads(line) for line in f]

        # filter data based on projection
        data = filter_data(data, data_config.projection)
        df = docs_to_df(data)

    elif source.endswith(".csv"):
        df = pd.read_csv(source)
        data = df.to_dict(orient="records")

        # filter data based on projection
        data = filter_data(data, data_config.projection)

        df = docs_to_df(data)

    elif pathlib.Path(source).is_dir():
        # recursively call load_data for each supported file in the directory
        dfs = []
        for path in pathlib.Path(source).glob("*.*"):
            if path.is_file() and path.suffix in [".json", ".jsonl", ".csv"]:
                click.echo(f"reading {path}")
                dfs.append(load_data(str(path), data_config))
        df = pd.concat(dfs)
    else:
        raise ValueError(f"unsupported source type for source {source}")

    # apply limit
    if data_config.limit > 0:
        df = df.head(data_config.limit)

    return df
