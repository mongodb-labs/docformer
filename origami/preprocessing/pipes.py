import pickle
import random
from collections import OrderedDict, defaultdict
from copy import copy
from typing import Optional

import numpy as np
import pandas as pd
from mdbrtools.schema import Schema
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_is_fitted

from origami.utils.common import ArrayStart, Symbol, pad_trunc, reorder_with_target_last, walk_all_leaf_kvs

from .encoder import StreamEncoder
from .utils import CAT_THRESHOLD, deepcopy_df, tokenize


class ColumnMissingException(Exception):
    pass


class BasePipe(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class ShuffleRowsPipe(BasePipe):
    """A Pipe that shuffles the rows in a DataFrame."""

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.sample(frac=1, replace=False, ignore_index=True)


class UpscalerPipe(BasePipe):
    """A Pipe that upscales the data by duplicating each rown `n` times. This performs a deepcopy on the data"""

    def __init__(self, n: int):
        self.n = n

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return pd.concat([deepcopy_df(X) for _ in range(self.n)], ignore_index=True)


class DocPermuterPipe(BasePipe):
    """A Pipe that permutes the field of the documents in the `docs` column. The original column
    is renamed to `ordered_docs`.

    Expects a DataFrame with `docs` column.

    Modifies the `docs` column in-place, and adds a `ordered_docs` column.
    """

    def __init__(self, shuffle_arrays: bool = False):
        self.shuffle_arrays = shuffle_arrays

    def _shuffle_keys(self, doc: dict | OrderedDict) -> OrderedDict:
        items = list(doc.items())
        for i, (k, v) in enumerate(items):
            if isinstance(v, (dict, OrderedDict)):
                v = self._shuffle_keys(v)
                items[i] = (k, v)
            if self.shuffle_arrays and isinstance(v, list):
                v = copy(v)
                random.shuffle(v)
                items[i] = (k, v)
        random.shuffle(items)
        return OrderedDict(items)

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if "docs" not in X.columns:
            raise ColumnMissingException("DocPermuterPipe requires column 'docs' in the DataFrame.")
        X = X.copy()
        X["ordered_docs"] = X["docs"]
        X["docs"] = X["docs"].astype("object").apply(self._shuffle_keys)

        return X


class SchemaParserPipe(BasePipe):
    """A Pipe that parses documents into a Schema object.

    Expects a DataFrame with `docs` column.
    """

    def __init__(self):
        self.schema = Schema()
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        self._is_fitted = True
        if "docs" not in X.columns:
            raise ColumnMissingException("SchemaParserPipe requires column 'docs' in the DataFrame.")

        self.schema.parse(X["docs"])
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self)
        return X


class TargetFieldPipe(BasePipe):
    """A Pipe that extracts the target field from a DataFrame and stores it in the `target` column. It also moves
    the target field last in the `docs` column.

    Expects a DataFrame with `docs` column.

    Modifies the `docs` column in-place, and adds a `target` column.
    """

    def __init__(self, target_field: str):
        self.target_field = target_field

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if "docs" not in X.columns:
            raise ColumnMissingException("TargetFieldPipe requires column 'docs' in the DataFrame.")

        X = deepcopy_df(X)
        docs, targets = zip(*X["docs"].map(lambda doc: reorder_with_target_last(doc, self.target_field)))
        X["docs"] = docs
        X["target"] = targets
        return X


class DocTokenizerPipe(BasePipe):
    """A Pipe that tokenizes the documents in a DataFrame."""

    def __init__(self, path_in_field_tokens: bool = True):
        self.path_in_field_tokens = path_in_field_tokens

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        X["tokens"] = X["docs"].apply(lambda doc: tokenize(doc, path_in_field_tokens=self.path_in_field_tokens))
        return X


class TokenEncoderPipe(BasePipe):
    """A Pipe that encodes the tokens to integer IDs"""

    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens
        self.encoder = StreamEncoder(predefined=Symbol)
        self.encoder.freeze()
        self._is_fitted = False

    def _encode_missing_array_starts(self):
        # find max array length
        array_tokens = [a for a in self.encoder.get_values() if isinstance(a, ArrayStart)]

        if len(array_tokens) > 0:
            max_array_length = max([a.size for a in array_tokens])

            # encode missing array starts
            for i in range(max_array_length):
                self.encoder.encode(ArrayStart(i))

    def fit(self, X: pd.DataFrame, y=None):
        self._is_fitted = True
        self.encoder.unfreeze()
        for seq in X["tokens"]:
            self.encoder.encode(seq)
        self._encode_missing_array_starts()

        if self.max_tokens:
            self.encoder.truncate(self.max_tokens)

        self.encoder.freeze()
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """explicitly implemented here for performance reasons, so we only need to encode once."""

        # if max_tokens is specified, we need to truncate before encoding and can't
        # fit_transform, because we don't know how many tokens we'll end up with
        if self.max_tokens:
            return self.fit(X).transform(X)

        if "tokens" not in X.columns:
            raise ColumnMissingException("TokenEncoderPipe requires column 'tokens' in the DataFrame.")

        X = X.copy()
        self.encoder.unfreeze()
        tokens = [self.encoder.encode(seq) for seq in X["tokens"]]
        self._encode_missing_array_starts()

        if self.max_tokens:
            self.encoder.truncate(self.max_tokens)

        self.encoder.freeze()

        X["tokens"] = tokens
        return X

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if "tokens" not in X.columns:
            raise ColumnMissingException("TokenEncoderPipe requires column 'tokens' in the DataFrame.")

        X = X.copy()
        X["tokens"] = X["tokens"].apply(self.encoder.encode)
        return X


class PadTruncTokensPipe(BasePipe):
    """A Pipe that pads the sequences to a fixed length."""

    def __init__(self, length: int | str = "max"):
        self.length = length
        self._is_fitted = False

    def _check_requirements(self, X: pd.DataFrame) -> None:
        check_is_fitted(self)
        if "tokens" not in X.columns:
            raise ColumnMissingException("PaddSequencePipe requires column 'tokens' in the DataFrame.")

    def fit(self, X: pd.DataFrame, y=None):
        self._is_fitted = True
        self._check_requirements(X)

        if self.length == "max":
            # add 1 to always have a `pad` token at the end (and an `end` token in the inputs)
            self.length = max(len(seq) for seq in X["tokens"]) + 1

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self._check_requirements(X)

        X = X.copy()
        X["tokens"] = X["tokens"].apply(lambda seq: pad_trunc(seq, self.length))
        return X


class ExistsTrackerPipe(BasePipe):
    """Observe the existance of field tokens, and track how often fields are missing.
    It makes no changes to the docs themselves. The information is saved in the self.fields dictionary.
    """

    def __init__(self):
        self.fields = defaultdict(int)

    def _walk_all_keys(self, doc: dict, prefix="") -> None:
        for key, value in doc.items():
            path = f"{prefix}.{key}".lstrip(".")
            self.fields[path] += 1

            if isinstance(value, dict):
                self._walk_all_keys(value, prefix=path)

    def fit(self, X: pd.DataFrame, y=None) -> "ExistsTrackerPipe":
        if "docs" not in X.columns:
            raise ColumnMissingException("ExistsTrackerPipe requires column 'docs' in the DataFrame.")

        X["docs"].apply(self._walk_all_keys)
        total_docs = len(X)
        self.fields = {k: v / total_docs for k, v in self.fields.items()}

        return self


class IdSetterPipe(BasePipe):
    """Modifies the _id field if one exists to be a constant"""

    def _replace_id_with_constant(self, doc: dict) -> dict:
        if "_id" in doc:
            doc["_id"] = "_id_"
        return doc

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if "docs" not in X.columns:
            raise ColumnMissingException("IdSetterPipe requires column 'docs' in the DataFrame.")

        # performs a deep copy to not change the original docs
        docs_series = pickle.loads(pickle.dumps(X["docs"]))
        X = X.copy()
        X["docs"] = docs_series.apply(self._replace_id_with_constant)
        return X


class KBinsDiscretizerPipe(BasePipe):
    """
    Bins all numeric values in a DataFrame using scikit-learn's KBinsDiscretizer.
    Numerical values are converted to the midpoint of its closest bin
    in order to reduce the cardinality of tokens that GPT has to learn.

    The `strategy` parameter is passed on to scikit-learn's KBinsDiscretizer
    instance and can be one of "uniform", "kmeans", or "quantile".
    """

    def __init__(
        self,
        bins: int = 100,
        threshold: int = CAT_THRESHOLD,
        strategy: str = "uniform",
    ):
        self.bins = bins
        self.threshold = threshold
        self.strategy = strategy

        self._is_fitted = False

        self.discretizers = {}

    def fit(self, X: pd.DataFrame, y=None) -> "KBinsDiscretizerPipe":
        """Creates a discretizer for each numerical field in the DataFrame."""

        self._is_fitted = True
        assert self.threshold >= self.bins, (
            f"`{self.threshold}` threshold is lower than {self.bins} bins. Use fewer bins to reduce cardinality."
        )

        docs = X["docs"]
        numerical_fields = defaultdict(list)

        for doc in docs:
            for item in walk_all_leaf_kvs(doc):
                if isinstance(item["value"], (float, int)) and not isinstance(item["value"], bool):
                    numerical_fields[item["path"]].append(item["value"])

        for path, values in numerical_fields.items():
            cardinality = len(set(values))
            # use binning if there are more unique values than the given threshold
            # otherwise leave low cardinality values as it is without binning
            if cardinality > self.threshold:
                self.discretizers[path] = KBinsDiscretizer(
                    n_bins=self.bins, encode="ordinal", strategy=self.strategy, subsample=None
                )
                values = np.array(values).reshape(-1, 1)
                self.discretizers[path].fit(values)

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self)

        docs_series = pickle.loads(pickle.dumps(X["docs"]))
        for doc in docs_series:
            for item in walk_all_leaf_kvs(doc):
                if (
                    item["path"] in self.discretizers
                    and isinstance(item["value"], (int, float))
                    and not isinstance(item["value"], bool)
                ):
                    # First use the discretizer to bucket the number
                    transformed_data = self.discretizers[item["path"]].transform(np.array(item["value"]).reshape(1, -1))
                    # Then assign the value as the mid point of the bucket
                    transformed_data = self.discretizers[item["path"]].inverse_transform(transformed_data)
                    item["parent"][item["idx"]] = transformed_data[0][0]

        X = X.copy()
        X["docs"] = docs_series
        return X
