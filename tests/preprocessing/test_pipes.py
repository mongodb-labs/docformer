import unittest
from collections import OrderedDict

import pandas as pd
from mdbrtools.schema import Schema
from sklearn.exceptions import NotFittedError

from origami.preprocessing import (
    DocPermuterPipe,
    DocTokenizerPipe,
    ExistsTrackerPipe,
    IdSetterPipe,
    KBinsDiscretizerPipe,
    PadTruncTokensPipe,
    SchemaParserPipe,
    ShuffleRowsPipe,
    TargetFieldPipe,
    TokenEncoderPipe,
    UpscalerPipe,
)
from origami.preprocessing.pipes import ColumnMissingException
from origami.utils.common import ArrayStart, FieldToken, Symbol


class TestShuffleRowsPipe(unittest.TestCase):
    def test_shuffle_rows_pipe(self):
        df = pd.DataFrame({"foo": range(100)})
        self.assertTrue(df["foo"].tolist() == list(range(100)))

        pipe = ShuffleRowsPipe()
        df = pipe.transform(df)

        self.assertEqual(len(df), 100)
        self.assertFalse(df["foo"].tolist() == list(range(100)))

        # test that index was reset
        self.assertEqual(list(range(100)), list(df.index))


class TestUpscalerPipe(unittest.TestCase):
    def test_upscaler_pipe(self):
        df = pd.DataFrame(
            {
                "foo": [
                    {"a": 1, "b": 2},
                    {"a": 1, "b": 2},
                    {"a": 1, "b": 2},
                    {"a": 1, "b": 2},
                    {"a": 1, "b": 2},
                ]
            }
        )

        pipe = UpscalerPipe(n=3)
        df = pipe.transform(df)

        self.assertEqual(len(df), 15)

        # test that items are deep-copied
        df["foo"].iloc[0]["a"] = 9

        # assert that 9 only occurs once in the column
        self.assertEqual([d["a"] for d in df["foo"]].count(9), 1)

        # test that index was reset
        self.assertEqual(list(range(len(df))), list(df.index))


class TestDocPermuterPipe(unittest.TestCase):
    def test_document_permuter(self):
        doc_with_100_fields = {f"field_{i}": i for i in range(100)}

        df = pd.DataFrame({"docs": [doc_with_100_fields]})

        self.assertTrue(list(df.loc[0, "docs"].values()) == list(doc_with_100_fields.values()))

        pipe = DocPermuterPipe()
        df = pipe.transform(df)

        self.assertIsInstance(df.loc[0, "docs"], OrderedDict)
        self.assertFalse(list(df.loc[0, "docs"].values()) == list(doc_with_100_fields.values()))

        self.assertIn("ordered_docs", df.columns)
        self.assertTrue(list(df.loc[0, "ordered_docs"].values()) == list(doc_with_100_fields.values()))

        # test that index is range
        self.assertEqual(list(range(len(df))), list(df.index))

    def test_doc_permuter_shuffle_arrays_true(self):
        doc_with_list = {"a": list(range(100))}

        df = pd.DataFrame({"docs": [doc_with_list]})

        self.assertListEqual(df.loc[0, "docs"]["a"], list(range(100)))

        pipe = DocPermuterPipe(shuffle_arrays=True)
        df = pipe.transform(df)

        self.assertIsInstance(df.loc[0, "docs"]["a"], list)
        self.assertFalse(df.loc[0, "docs"]["a"] == list(range(100)))

        self.assertIn("ordered_docs", df.columns)
        self.assertTrue(df.loc[0, "ordered_docs"]["a"] == list(range(100)))

    def test_doc_permuter_shuffle_arrays_false(self):
        doc_with_list = {"a": list(range(100))}

        df = pd.DataFrame({"docs": [doc_with_list]})

        self.assertListEqual(df.loc[0, "docs"]["a"], list(range(100)))

        pipe = DocPermuterPipe(shuffle_arrays=False)
        df = pipe.transform(df)

        self.assertIsInstance(df.loc[0, "docs"]["a"], list)
        self.assertTrue(df.loc[0, "docs"]["a"] == list(range(100)))

        self.assertIn("ordered_docs", df.columns)
        self.assertTrue(df.loc[0, "ordered_docs"]["a"] == list(range(100)))


class TestSchemaParserPipe(unittest.TestCase):
    def test_schema_parser_pipe(self):
        df = pd.DataFrame({"docs": [{"a": 1}, {"a": 2, "b": 3}]})

        pipe = SchemaParserPipe()
        pipe.fit(df)

        schema = pipe.schema
        self.assertIsInstance(schema, Schema)
        self.assertEqual(schema.count, 2)

        self.assertEqual(schema["a"].types["int"].values, [1, 2])
        self.assertEqual(schema["b"].types["int"].values, [3])
        self.assertEqual(schema["a"].types["int"].count, 2)
        self.assertEqual(schema["b"].types["int"].count, 1)


class TestTargetFieldPipe(unittest.TestCase):
    def test_supervised_target_pipe(self):
        df = pd.DataFrame(
            {
                "docs": [
                    {"a": 1, "b": 2, "c": 3},
                    {"b": 1, "a": 3, "c": 2},
                    {"c": 2, "a": 1},  # Missing 'b'
                ]
            }
        )

        pipe = TargetFieldPipe("b")
        df = pipe.transform(df)

        self.assertIn("target", df.columns)

        for doc, target in zip(df["docs"], df["target"]):
            self.assertIn("b", doc)
            self.assertEqual(doc["b"], target)


class TestDocTokenizerPipe(unittest.TestCase):
    def test_doc_tokenizer_pipe(self):
        df = pd.DataFrame({"docs": [{"foo": 1, "bar": [1, 2, 3], "baz": {"baz_inner": True}}]})

        pipe = DocTokenizerPipe()
        df = pipe.transform(df)

        self.assertTrue("tokens" in df.columns)

        self.assertEqual(
            df.loc[0, "tokens"],
            [
                Symbol.START,
                FieldToken("foo"),
                1,
                FieldToken("bar"),
                ArrayStart(3),
                1,
                2,
                3,
                FieldToken("baz"),
                Symbol.SUBDOC_START,
                FieldToken("baz.baz_inner"),
                True,
                Symbol.SUBDOC_END,
                Symbol.END,
            ],
        )

        # test that index is range
        self.assertEqual(list(range(len(df))), list(df.index))


class TestTokenEncoderPipe(unittest.TestCase):
    def test_token_encoder_init(self):
        pipe = TokenEncoderPipe()

        self.assertTrue(pipe.encoder.frozen)

    def test_token_encoder_fit_transform(self):
        df = pd.DataFrame({"tokens": ["foo", "bar", 1, 2]})

        pipe = TokenEncoderPipe()
        df = pipe.fit_transform(df)

        # 0-9 are the predefined Symbols
        self.assertEqual(df["tokens"].tolist(), [10, 11, 12, 13])

        self.assertTrue(pipe.encoder.frozen)

        # test that index is range
        self.assertEqual(list(range(len(df))), list(df.index))

    def test_token_encoder_transform(self):
        df_original = pd.DataFrame({"tokens": ["foo", "bar", 1, 2]})
        df_transform1 = pd.DataFrame({"tokens": ["foo", "bar", 1, 2]})
        df_transform2 = pd.DataFrame({"tokens": ["foo", "bar", 3, 4]})

        pipe = TokenEncoderPipe()
        df_transform1 = pipe.transform(df_transform1)

        # before fitting, all tokens are unknown
        unknown = pipe.encoder.encode(Symbol.UNKNOWN)
        self.assertEqual(df_transform1["tokens"].tolist(), [unknown] * 4)

        self.assertTrue(pipe.encoder.frozen)
        pipe.fit(df_original)
        self.assertTrue(pipe.encoder.frozen)

        # after fitting, some tokens are encoded, some remain unknown
        df_transform2 = pipe.transform(df_transform2)
        self.assertEqual(df_transform2["tokens"].tolist(), [10, 11, unknown, unknown])

        # test that index is range
        self.assertEqual(list(range(len(df_transform1))), list(df_transform1.index))
        self.assertEqual(list(range(len(df_transform2))), list(df_transform2.index))

    def test_token_encoder_fill_missing_arrays(self):
        df = pd.DataFrame({"tokens": [[ArrayStart(2), "b", "c"], [ArrayStart(5), "b", "c", "d", "e"]]})
        print(df)
        pipe = TokenEncoderPipe()
        pipe.fit(df)

        self.assertIn(ArrayStart(0), pipe.encoder.get_values())
        self.assertIn(ArrayStart(1), pipe.encoder.get_values())
        self.assertIn(ArrayStart(2), pipe.encoder.get_values())
        self.assertIn(ArrayStart(3), pipe.encoder.get_values())
        self.assertIn(ArrayStart(4), pipe.encoder.get_values())
        self.assertIn(ArrayStart(5), pipe.encoder.get_values())


class TestPadTruncTokensPipe(unittest.TestCase):
    def test_pad_trunc_tokens_pipe_pad(self):
        df = pd.DataFrame({"tokens": [[1, 2, 3], [4, 5]]})

        pipe = PadTruncTokensPipe(length=5)
        df = pipe.fit_transform(df)

        tokens = df["tokens"].tolist()
        self.assertEqual(tokens, [[1, 2, 3, Symbol.PAD, Symbol.PAD], [4, 5, Symbol.PAD, Symbol.PAD, Symbol.PAD]])

        # test that index is range
        self.assertEqual(list(range(len(df))), list(df.index))

    def test_pad_trunc_tokens_pipe_trunc(self):
        df = pd.DataFrame({"tokens": [[1, 2, 3], [4, 5, 6, 7]]})

        pipe = PadTruncTokensPipe(length=2)
        df = pipe.fit_transform(df)

        tokens = df["tokens"].tolist()
        self.assertEqual(tokens, [[1, 2], [4, 5]])

    def test_pad_trunc_tokens_pipe_max(self):
        df = pd.DataFrame({"tokens": [[1, 2], [4, 5, 6, 7]]})

        pipe = PadTruncTokensPipe(length="max")
        df = pipe.fit_transform(df)

        tokens = df["tokens"].tolist()
        self.assertEqual(tokens, [[1, 2, Symbol.PAD, Symbol.PAD, Symbol.PAD], [4, 5, 6, 7, Symbol.PAD]])

    def test_pad_trunc_tokens_pipe_max_default(self):
        df = pd.DataFrame({"tokens": [[1, 2], [4, 5, 6, 7]]})

        pipe = PadTruncTokensPipe()
        df = pipe.fit_transform(df)

        tokens = df["tokens"].tolist()
        self.assertEqual(tokens, [[1, 2, Symbol.PAD, Symbol.PAD, Symbol.PAD], [4, 5, 6, 7, Symbol.PAD]])

    def test_pad_trunc_tokens_pipe_tokens_missing(self):
        df = pd.DataFrame({"docs": [{"a": 1, "b": 2}]})

        pipe = PadTruncTokensPipe(length="max")
        self.assertRaises(ColumnMissingException, lambda: pipe.fit_transform(df))

    def test_pad_trunc_tokens_pipe_not_fitted(self):
        df = pd.DataFrame({"docs": [{"a": 1, "b": 2}]})

        pipe = PadTruncTokensPipe(length="max")
        self.assertRaises(NotFittedError, lambda: pipe.transform(df))


class TestExistsTrackerPipe(unittest.TestCase):
    def test_exists_tracker_pipe(self):
        df = pd.DataFrame(
            {
                "docs": [
                    {"a": 1, "c": "str", "d": 3},
                    {"a": 1, "b": None, "c": "str"},
                    {"a": 2, "b": 2.1},
                    {"a": 2, "b": 2.0},
                ]
            }
        )

        pipe = ExistsTrackerPipe()
        df = pipe.fit_transform(df)

        exists_fields = pipe.fields

        self.assertEqual(1.0, exists_fields["a"])
        self.assertEqual(0.75, exists_fields["b"])
        self.assertEqual(0.5, exists_fields["c"])
        self.assertEqual(0.25, exists_fields["d"])

    def test_exists_tracker_pipe_nested_subdocuments(self):
        df = pd.DataFrame(
            {
                "docs": [
                    {"a": 1, "c": "str", "d": 3},
                    {"a": 1, "b": {"b1": 1, "b2": {"b3": 0}}, "c": "str"},
                    {"a": 1, "b": {"b1": 1, "b2": 1}, "c": "str"},
                    {"a": 1, "b": {"b2": 1}, "c": "str"},
                ]
            }
        )

        pipe = ExistsTrackerPipe()
        df = pipe.fit_transform(df)

        exists_fields = pipe.fields

        self.assertEqual(1.0, exists_fields["a"])
        self.assertEqual(0.75, exists_fields["b"])
        self.assertEqual(1.0, exists_fields["c"])
        self.assertEqual(0.25, exists_fields["d"])
        self.assertEqual(0.5, exists_fields["b.b1"])
        self.assertEqual(0.75, exists_fields["b.b2"])
        self.assertEqual(0.25, exists_fields["b.b2.b3"])


class TestIdSetterPipe(unittest.TestCase):
    def test_id_setter_pipe_no_id(self):
        df = pd.DataFrame({"docs": [{"bar": "foo"}]})

        pipe = IdSetterPipe()
        df = pipe.transform(df)

        self.assertEqual(df.loc[0, "docs"], {"bar": "foo"})

        # test that index is range
        self.assertEqual(list(range(len(df))), list(df.index))

    def test_id_setter_pipe(self):
        df = pd.DataFrame(
            {
                "docs": [
                    {"_id": "some-id", "bar": "foo"},
                    {"_id": "_id_", "bar": "foo"},
                    {"_id": "other-id", "bar": "foo"},
                ]
            }
        )
        pipe = IdSetterPipe()
        df = pipe.transform(df)

        for doc in df["docs"]:
            self.assertEqual(doc["_id"], "_id_")


class TestKBinsDiscretizerPipe(unittest.TestCase):
    def test_kbinzdiscretizer(self):
        df = pd.DataFrame(
            {
                "docs": [
                    {"a": 1, "b": None, "c": "str", "d": 3},
                    {"a": 1, "b": 6.7, "c": "str", "d": 4},
                    {"a": 2, "b": 2.1, "c": None, "d": 5},
                    {"a": 2, "b": 2.0, "c": None, "d": 6},
                ]
            }
        )

        pipe = KBinsDiscretizerPipe(bins=2, threshold=2)
        df = pipe.fit_transform(df)

        discretizers = pipe.discretizers
        self.assertEqual(len(discretizers), 2)

        # a is too low cardinality to use binning
        self.assertFalse("a" in discretizers)
        self.assertTrue("b" in discretizers)
        self.assertFalse("c" in discretizers)
        self.assertTrue("d" in discretizers)

        # test that index is range
        self.assertEqual(list(range(len(df))), list(df.index))

    def test_kbinzdiscretizer_nested(self):
        df = pd.DataFrame(
            {
                "docs": [
                    {"a": 1, "c": "str", "d": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                    {"a": 1, "b": {"a": 1, "b": 0}, "c": "str", "d": [4, 5, 6]},
                    {"a": 2, "b": {"a": 2, "b": 1}, "c": None, "d": 5},
                    {"a": 2, "b": {"a": 3, "b": 1}, "c": None, "d": 6},
                ]
            }
        )

        pipe = KBinsDiscretizerPipe(bins=2, threshold=2)
        df = pipe.fit_transform(df)

        discretizers = pipe.discretizers
        self.assertEqual(len(discretizers), 2)

        self.assertFalse("a" in discretizers)
        self.assertFalse("b" in discretizers)
        self.assertTrue("b.a" in discretizers)
        self.assertFalse("b.b" in discretizers)
        self.assertFalse("c" in discretizers)
        self.assertTrue("d.[]" in discretizers)

        self.assertEqual(len(df.loc[0, "docs"]["d"]), 9)
