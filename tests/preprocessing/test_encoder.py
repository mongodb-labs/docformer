import unittest
from random import shuffle

import numpy as np
import pandas as pd
import torch

from origami.preprocessing.encoder import StreamEncoder
from origami.utils.common import ArrayStart, Symbol


class TestEncoder(unittest.TestCase):
    def test_encode_single_val(self):
        encoder = StreamEncoder()
        result = encoder.encode_val("foo")
        self.assertEqual(result, 0)
        decoded = encoder.decode_val(result)
        self.assertEqual(decoded, "foo")

    def test_encode_multiple_vals(self):
        encoder = StreamEncoder()
        for i in range(4):
            val = f"_{i}_"
            r = encoder.encode_val(val)
            self.assertEqual(r, i)
            d = encoder.decode_val(r)
            self.assertEqual(d, val)

    def test_encode_single_val_predefined(self):
        encoder = StreamEncoder(predefined={"SPECIAL": 0})
        result = encoder.encode_val("foo")
        self.assertEqual(result, 1)
        decoded = encoder.decode_val(result)
        self.assertEqual(decoded, "foo")
        self.assertEqual(encoder.decode_val(0), "SPECIAL")

    def test_encode_iter(self):
        encoder = StreamEncoder()
        values = [f"_{i}_" for i in range(10)]
        result = encoder.encode_iter(values)
        self.assertEqual(list(result), list(range(10)))
        decoded = list(encoder.decode_iter(encoder.encode_iter(values)))
        self.assertEqual(list(decoded), values)

    def test_encode_list(self):
        encoder = StreamEncoder()
        values = [f"_{i}_" for i in range(10)]
        result = encoder.encode_list(values)
        self.assertEqual(result, list(range(10)))
        decoded = encoder.decode_list(result)
        self.assertEqual(decoded, values)

    def test_encode_dict_values(self):
        encoder = StreamEncoder()
        values = [f"_{i}_" for i in range(10)]
        d = dict(zip(values, values))

        result = encoder.encode_dict(d)
        self.assertEqual(list(result.keys()), values)
        self.assertEqual(list(result.values()), list(range(10)))

        decoded = encoder.decode_dict(result)
        self.assertEqual(decoded, d)

    def test_encode_dict_values_and_keys(self):
        encoder = StreamEncoder()
        values = [f"_{i}_" for i in range(10)]
        d = dict(zip(values, values))

        result = encoder.encode_dict(d, include_keys=True)
        self.assertEqual(list(result.keys()), list(range(10)))
        self.assertEqual(list(result.values()), list(range(10)))

        decoded = encoder.decode_dict(result, include_keys=True)
        self.assertEqual(decoded, d)

    def test_encode_recursive(self):
        encoder = StreamEncoder()

        item = {"key1": [{"key2": "val1", "key3": "val2"}, "val3", "val4"]}
        result = encoder.encode(item)
        self.assertEqual(result, {"key1": [{"key2": 0, "key3": 1}, 2, 3]})

        decoded = encoder.decode(result)
        self.assertEqual(decoded, item)

    def test_encode_types(self):
        # test encoding list -> list
        encoder = StreamEncoder()
        item = ["foo", "bar", "baz"]
        result = encoder.encode(item)
        self.assertEqual(result, [0, 1, 2])
        self.assertIsInstance(result, list)

        # test encoding set -> set
        encoder = StreamEncoder()
        item = {"foo", "bar", "baz"}
        result = encoder.encode(item)
        self.assertEqual(result, {0, 1, 2})
        self.assertIsInstance(result, set)

        # test encoding tuple -> tuple
        encoder = StreamEncoder()
        item = ("foo", "bar", "baz")
        result = encoder.encode(item)
        self.assertEqual(result, (0, 1, 2))
        self.assertIsInstance(result, tuple)

        # test encoding pd.Series -> list
        encoder = StreamEncoder()
        item = pd.Series(["foo", "bar", "baz"])
        result = encoder.encode(item)
        self.assertEqual(result, [0, 1, 2])
        self.assertIsInstance(result, list)

        # test encoding torch.Tensor -> list
        encoder = StreamEncoder()
        item = torch.tensor([16.2, 21.5, 13.9])
        result = encoder.encode(item)
        self.assertEqual(result, [0, 1, 2])
        self.assertIsInstance(result, list)

        # test encoding np.ndarray -> list
        encoder = StreamEncoder()
        item = np.array(["foo", "bar", "baz"])
        result = encoder.encode(item)
        self.assertEqual(result, [0, 1, 2])
        self.assertIsInstance(result, list)

    def test_decode_types(self):
        # test decoding list -> list
        encoder = StreamEncoder()
        encoder.encode(["foo", "bar", "baz"])
        item = [0, 1, 2]
        result = encoder.decode(item)
        self.assertEqual(result, ["foo", "bar", "baz"])
        self.assertIsInstance(result, list)

        # test decoding set -> set
        encoder = StreamEncoder()
        encoder.encode(["foo", "bar", "baz"])
        item = {0, 1, 2}
        result = encoder.decode(item)
        self.assertEqual(result, {"foo", "bar", "baz"})
        self.assertIsInstance(result, set)

        # test decoding tuple -> tuple
        encoder = StreamEncoder()
        encoder.encode(["foo", "bar", "baz"])
        item = (0, 1, 2)
        result = encoder.decode(item)
        self.assertEqual(result, ("foo", "bar", "baz"))
        self.assertIsInstance(result, tuple)

        # test decoding Series -> list
        encoder = StreamEncoder()
        encoder.encode(["foo", "bar", "baz"])
        item = pd.Series([0, 1, 2])
        result = encoder.decode(item)
        self.assertEqual(result, ["foo", "bar", "baz"])
        self.assertIsInstance(result, list)

        # test decoding Tensor -> list
        encoder = StreamEncoder()
        encoder.encode(["foo", "bar", "baz"])
        item = torch.tensor([0, 1, 2])
        result = encoder.decode(item)
        self.assertEqual(result, ["foo", "bar", "baz"])
        self.assertIsInstance(result, list)

        # test decoding array -> list
        encoder = StreamEncoder()
        encoder.encode(["foo", "bar", "baz"])
        item = np.array([0, 1, 2])
        result = encoder.decode(item)
        self.assertEqual(result, ["foo", "bar", "baz"])
        self.assertIsInstance(result, list)

    def test_encode_recursive_with_keys(self):
        encoder = StreamEncoder()

        item = {"key1": [{"key2": "val1", "key3": "val2"}, "val3", "val4"]}
        result = encoder.encode(item, include_dict_keys=True)
        self.assertEqual(result, {0: [{1: 2, 3: 4}, 5, 6]})

        decoded = encoder.decode(result, include_dict_keys=True)
        self.assertEqual(decoded, item)

    def test_encode_list_of_tuples(self):
        encoder = StreamEncoder()
        seq = [("A", "1"), ("B", "2"), ("A", "3"), ("B", "4")]
        result = encoder.encode(seq)

        self.assertEqual(result, [(0, 1), (2, 3), (0, 4), (2, 5)])

    def test_freeze_without_predefined(self):
        encoder = StreamEncoder()
        result = encoder.encode_val("foo")
        self.assertEqual(result, 0)

        encoder.freeze()

        self.assertEqual(encoder.encode_val("foo"), 0)
        self.assertRaises(KeyError, lambda: encoder.encode_val("bar"))

        encoder.unfreeze()

        self.assertEqual(encoder.encode_val("foo"), 0)
        self.assertEqual(encoder.encode_val("bar"), 1)

    def test_freeze_with_predefined(self):
        encoder = StreamEncoder(predefined=Symbol)
        result = encoder.encode_val("foo")
        self.assertEqual(result, len(Symbol))

        encoder.freeze()

        self.assertEqual(encoder.encode_val("foo"), len(Symbol))
        self.assertEqual(encoder.encode_val("bar"), encoder.encode(Symbol.UNKNOWN))

        encoder.unfreeze()

        self.assertEqual(encoder.encode_val("bar"), len(Symbol) + 1)

    def test_token_freq(self):
        values = ["foo"] * 7 + ["bar"] * 3 + ["baz"]
        shuffle(values)

        encoder = StreamEncoder()
        encoder.encode(values)

        self.assertEqual(encoder.token_freq["foo"], 7)
        self.assertEqual(encoder.token_freq["bar"], 3)
        self.assertEqual(encoder.token_freq["baz"], 1)

    def test_truncate_noop(self):
        encoder = StreamEncoder()
        encoder.encode(["foo", "bar", "baz"])
        encoder.truncate(3)
        self.assertEqual(encoder.get_values(), ["foo", "bar", "baz"])
        self.assertEqual(list(encoder.ids_to_tokens.keys()), list(range(3)))

    def test_truncate_above_limit(self):
        encoder = StreamEncoder()
        encoder.encode(["foo", "bar", "foo", "baz", "foo", "baz"])
        encoder.truncate(2)
        self.assertEqual(encoder.get_values(), ["foo", "baz"])
        self.assertEqual(encoder.token_freq["foo"], 3)
        self.assertEqual(encoder.token_freq["baz"], 2)
        self.assertEqual(len(encoder.token_freq), 2)
        self.assertEqual(list(encoder.ids_to_tokens.keys()), list(range(2)))

    def test_truncate_with_symbols(self):
        encoder = StreamEncoder(predefined=Symbol)
        encoder.encode(["foo", "foo", "baz", "foo", "bar", "baz"])

        self.assertEqual(len(encoder), 13)

        encoder.truncate(11)
        self.assertEqual(len(encoder), 11)

        values = encoder.get_values()
        self.assertListEqual(values, list(Symbol) + ["foo"])
        self.assertEqual(list(encoder.ids_to_tokens.keys()), list(range(11)))
        self.assertEqual(encoder.encode(Symbol.PAD), 0)

    def test_truncate_with_array_starts(self):
        encoder = StreamEncoder()
        encoder.encode(["foo", "foo", ArrayStart(2), "baz", "foo", "baz"])
        encoder.truncate(2)
        self.assertEqual(encoder.get_values(), [ArrayStart(2), "foo"])
        self.assertEqual(list(encoder.ids_to_tokens.keys()), list(range(2)))

    def test_truncate_raise_value_error(self):
        encoder = StreamEncoder(predefined=Symbol)
        encoder.encode(["foo", "foo", "baz", "foo", "bar", "baz"])
        self.assertEqual(len(encoder), 13)
        with self.assertRaises(ValueError):
            encoder.truncate(5)

    def test_truncate_encode_with_unknown_symbol(self):
        encoder = StreamEncoder(predefined=Symbol)
        encoder.encode(["foo", "foo", "baz", "foo", "bar", "baz"])

        # before truncating
        foo_enc = encoder.encode("foo")
        bar_enc = encoder.encode("bar")

        self.assertEqual(encoder.decode(foo_enc), "foo")
        self.assertEqual(encoder.decode(bar_enc), "bar")

        encoder.truncate(11)
        encoder.freeze()

        # after truncating
        foo_enc = encoder.encode("foo")
        bar_enc = encoder.encode("bar")

        self.assertEqual(encoder.decode(foo_enc), "foo")
        self.assertEqual(encoder.decode(bar_enc), Symbol.UNKNOWN)
        self.assertEqual(len(encoder.token_freq), 11)

        self.assertEqual(list(encoder.ids_to_tokens.keys()), list(range(11)))
        self.assertEqual(encoder.encode(Symbol.PAD), 0)
