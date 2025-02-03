from sklearn.base import BaseEstimator, TransformerMixin

from origami.utils import walk_all_leaf_kvs


def flatten_config(config: dict) -> dict:
    flat = {item["path"]: item["value"] for item in walk_all_leaf_kvs(config, include_pos_in_path=True)}
    return flat


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X
