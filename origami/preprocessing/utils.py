from collections import OrderedDict
from copy import deepcopy
from typing import Any

import pandas as pd
import torch
from pymongo import MongoClient
from torch.utils.data import default_collate

from origami.utils.common import ArrayStart, FieldToken, Symbol

CAT_THRESHOLD = 1000


def docs_to_df(docs: list[dict]) -> pd.DataFrame:
    """converts a list of documents into a DataFrame with column named `docs`."""
    df = pd.DataFrame({"docs": docs})
    df["id"] = df.index
    return df


def load_df_from_mongodb(uri, db, coll, **kwargs):
    """loads the documents from a MongoDB collection into a DataFrame with column named `docs`."""
    client = MongoClient(uri)
    collection = client[db][coll]
    docs = list(collection.find(**kwargs))
    return docs_to_df(docs)


def deepcopy_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=df.columns, data=deepcopy(df.values))


def tokenize(doc: dict, path_in_field_tokens: bool = True) -> dict:
    """parses a document into sequence of tokens."""

    def parse_rec(thing: Any, prefix: str = ""):
        """inner parse function, recursively parses dicts/OrderedDicts, lists and
        primitive values."""
        if isinstance(thing, (dict, OrderedDict)):
            yield Symbol.START if prefix == "" else Symbol.SUBDOC_START
            for key, val in thing.items():
                if path_in_field_tokens:
                    # the full field path is encoded in the FieldToken
                    yield FieldToken(prefix + key)
                else:
                    # only the field name is encoded in the FieldToken
                    yield FieldToken(key)
                for tok in parse_rec(val, prefix=f"{prefix + key}."):
                    yield tok
            yield Symbol.END if prefix == "" else Symbol.SUBDOC_END
        elif isinstance(thing, list):
            yield ArrayStart(len(thing))
            for val in thing:
                for tok in parse_rec(val, prefix=prefix):
                    yield tok
        else:
            yield thing

    return list(parse_rec(doc))


def detokenize(tokens: list) -> dict:
    """reverses the tokenization process, turning a sequence of tokens into a dict."""

    gen = iter(tokens)
    assert next(gen) == Symbol.START

    def consume_value(gen):
        value = next(gen)
        if isinstance(value, ArrayStart):
            # if the value is of type list:
            value = consume_list(gen, value.size)
        elif value == Symbol.SUBDOC_START:
            value = consume_doc(gen)
        elif isinstance(value, Symbol):
            value = value.name
        return value

    def consume_list(gen, size: int):
        lst = []
        for _ in range(size):
            lst.append(consume_value(gen))
        return lst

    def consume_doc(gen):
        inner_doc = {}
        key = next(gen)

        while key not in [Symbol.END, Symbol.SUBDOC_END]:
            # discard full path information "a.b.c", only return child key name "c"
            if isinstance(key, FieldToken):
                key = key.name.split(".")[-1]
            inner_doc[key] = consume_value(gen)
            key = next(gen)
        return inner_doc

    return consume_doc(gen)


def target_collate_fn(target_token_id: int):
    def collate_fn(tokens: list[torch.tensor]) -> torch.tensor:
        """collate function that only returns sequences up to a target token (incl.). Assumes
        the target token is at the same position in each sequence. (use with TargetTokenBatchSampler)"""
        tokens = default_collate(tokens)

        # find target position
        pos = torch.where(tokens[0] == target_token_id)[-1].item()

        # return tokens up to that position + 1 (including the target)
        return tokens[:, : pos + 1]

    return collate_fn
