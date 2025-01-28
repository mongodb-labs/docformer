import unittest

from origami.utils.common import flatten_docs, walk_all_leaf_kvs


class TestWalkAllLeafKVs(unittest.TestCase):
    def test_walk_all_leaf_kvs_nested(self):
        d = {"foo": 1, "bar": {"baz": 2, "buz": 3}}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "foo", "key": "foo", "pos": None, "path": "foo", "value": 1},
                {"idx": "baz", "key": "baz", "pos": None, "path": "bar.baz", "value": 2},
                {"idx": "buz", "key": "buz", "pos": None, "path": "bar.buz", "value": 3},
            ],
        )

    def test_walk_all_leaf_kvs_list(self):
        d = {"foo": 1, "bar": [2, 3, 4]}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "foo", "key": "foo", "pos": None, "path": "foo", "value": 1},
                {"idx": 0, "key": "bar", "pos": 0, "path": "bar.[]", "value": 2},
                {"idx": 1, "key": "bar", "pos": 1, "path": "bar.[]", "value": 3},
                {"idx": 2, "key": "bar", "pos": 2, "path": "bar.[]", "value": 4},
            ],
        )

    def test_walk_all_leaf_kvs_list_with_pos(self):
        d = {"foo": 1, "bar": [2, 3, 4]}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d, include_pos_in_path=True),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "foo", "key": "foo", "pos": None, "path": "foo", "value": 1},
                {"idx": 0, "key": "bar", "pos": 0, "path": "bar.[0]", "value": 2},
                {"idx": 1, "key": "bar", "pos": 1, "path": "bar.[1]", "value": 3},
                {"idx": 2, "key": "bar", "pos": 2, "path": "bar.[2]", "value": 4},
            ],
        )

    def test_walk_all_leaf_kvs_list_of_dicts(self):
        d = {"foo": [{"bar": 1}, {"bar": 2}]}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "bar", "key": "bar", "pos": 0, "path": "foo.[].bar", "value": 1},
                {"idx": "bar", "key": "bar", "pos": 1, "path": "foo.[].bar", "value": 2},
            ],
        )

    def test_walk_all_leaf_kvs_list_of_dicts_with_pos(self):
        d = {"foo": [{"bar": 1}, {"bar": 2}]}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d, include_pos_in_path=True),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "bar", "key": "bar", "pos": 0, "path": "foo.[0].bar", "value": 1},
                {"idx": "bar", "key": "bar", "pos": 1, "path": "foo.[1].bar", "value": 2},
            ],
        )


class TestFlattenDocs(unittest.TestCase):
    def test_flatten_docs(self):
        docs = [
            {"foo": 1, "bar": {"baz": 2, "buz": 3}},
            {"foo": 4, "bar": ["test", {"a": "b"}]},
        ]

        flat = flatten_docs(docs)

        self.assertEqual(
            flat, [{"foo": 1, "bar.baz": 2, "bar.buz": 3}, {"foo": 4, "bar.[0]": "test", "bar.[1].a": "b"}]
        )
