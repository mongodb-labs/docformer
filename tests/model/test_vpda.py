import unittest

import numpy as np
import pandas as pd
import torch
from mdbrtools.schema import parse_schema
from sklearn.pipeline import Pipeline

from origami.model.vpda import INVALID, NOOP, VPDA, ObjectVPDA
from origami.preprocessing import DocTokenizerPipe, PadTruncTokensPipe, TokenEncoderPipe
from origami.utils.common import ArrayStart, FieldToken, Symbol

START = 1
END = 2
A = 3
B = 4
PAD = 5


class TestVPDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_alph = torch.tensor([4, 1, 6, 0])
        cls.symbol_alph = torch.tensor([3, 4, 5, 6, 8, 9])
        cls.state_alph = torch.tensor([0, 1, 2])
        cls.vpda = VPDA(
            TestVPDA.input_alph,
            TestVPDA.symbol_alph,
            TestVPDA.state_alph,
            pad_input=6,
            start_state=0,
            end_state=2,
            start_symbol=3,
            end_symbol=5,
        )

    @classmethod
    def make_anbn_pda(cls):
        vpda = VPDA(
            input_alph=torch.tensor([START, END, A, B, PAD]),
            symbol_alph=torch.tensor([START, A, END]),
            state_alph=torch.tensor([START, A, B, END]),
            pad_input=PAD,
            start_symbol=START,
            end_symbol=END,
            start_state=START,
            end_state=END,
        )

        # implement rules

        # START -> END: END --> START, END
        vpda.insert_transition(
            source_state=START,
            target_state=END,
            input=END,
            pop_symbol=START,
            push_symbol=END,
        )

        # START -> A: a --> epsilon, a
        vpda.insert_transition(source_state=START, target_state=A, input=A, pop_symbol=None, push_symbol=A)

        # A -> A: a --> epsilon, a
        vpda.insert_transition(source_state=A, target_state=A, input=A, pop_symbol=None, push_symbol=A)

        # A -> B: b --> a, epsilon
        vpda.insert_transition(source_state=A, target_state=B, input=B, pop_symbol=A, push_symbol=None)

        # B -> B: b --> a, epsilon
        vpda.insert_transition(source_state=B, target_state=B, input=B, pop_symbol=A, push_symbol=None)

        # B -> END: END --> START, END
        vpda.insert_transition(
            source_state=B,
            target_state=END,
            input=END,
            pop_symbol=START,
            push_symbol=END,
        )

        return vpda

    def test_init(self):
        vpda = TestVPDA.vpda

        self.assertTrue(np.array_equal(vpda.inputs_to_idx, np.array([3, 1, -1, -1, 0, -1, 2])))

        self.assertTrue(
            np.array_equal(
                vpda.symbols_to_idx,
                np.array(
                    [
                        0,
                        -1,
                        -1,
                        1,
                        2,
                        3,
                        4,
                        -1,
                        5,
                        6,
                    ]
                ),
            )
        )

        self.assertTrue(np.array_equal(vpda.states_to_idx, np.array([0, 1, 2])))

        shape = (
            TestVPDA.state_alph.size(0),
            TestVPDA.symbol_alph.size(0) + 1,
            TestVPDA.input_alph.size(0),
        )

        self.assertEqual(vpda.pop_t.shape, shape)
        self.assertEqual(vpda.push_t.shape, shape)
        self.assertEqual(vpda.state_t.shape, shape)

        invalid_mask = np.zeros_like(vpda.pop_t, dtype="bool")
        invalid_mask[vpda.states_to_idx[vpda.end_state], :, vpda.inputs_to_idx[vpda.pad_input]] = True

        self.assertTrue((vpda.pop_t[invalid_mask] == NOOP).all())
        self.assertTrue((vpda.push_t[invalid_mask] == NOOP).all())
        self.assertTrue((vpda.state_t[invalid_mask] == vpda.end_state).all())
        self.assertTrue((vpda.pop_t[~invalid_mask] == INVALID).all())
        self.assertTrue((vpda.push_t[~invalid_mask] == INVALID).all())
        self.assertTrue((vpda.state_t[~invalid_mask] == INVALID).all())

        # all but the last state dimension is INVALID (due to default transition)
        # self.assertTrue((vpda.pop_t[:, :, :-1] == INVALID).all())
        # self.assertTrue((vpda.push_t[:, :, :-1] == INVALID).all())
        # self.assertTrue((vpda.state_t[:, :, :-1] == INVALID).all())

    def test_inverse_index(self):
        vpda = TestVPDA.vpda

        # test for a single value
        x = np.array([3])
        self.assertTrue(np.array_equal(vpda._create_inverse_index(x), np.array([-1, -1, -1, 0])))

        # test for empty input
        x = np.array([])
        self.assertTrue(np.array_equal(vpda._create_inverse_index(x), np.array([])))

        # test for negative numbers (should raise IndexError with out of bounds message)
        x = np.array([-1, -2, -3])
        self.assertRaises(IndexError, lambda: vpda._create_inverse_index(x))

        # test for large numbers
        x = np.array([1000, 2000, 3000])
        self.assertEqual(vpda._create_inverse_index(x).size, 3001)

        # test that indexing into the inverse index with x returns torch.arange(x.size(0))
        x = np.array(np.random.choice(np.arange(1, 100), size=10, replace=False))
        ix = vpda._create_inverse_index(x)
        self.assertTrue(np.array_equal(ix[x], torch.arange(x.shape[0])))

    def test_top_of_stack(self):
        vpda = TestVPDA.vpda

        vpda.stacks = np.array(
            [
                [2, 3, -3, 5],  # 5 on top
                [4, -4, 5, -5],  # nothing on top (-1)
                [2, -2, 3, 0],  # 3 on top
                [2, 4, -4, -2],  # nothing on top (-1)
            ],  # nothing on top
            dtype="int32",
        )

        top_of_stack = vpda.top_of_stack()
        self.assertTrue(np.array_equal(top_of_stack, np.array([5, -1, 3, -1])))

        # ---

        vpda.stacks = np.array(
            [
                [2, 1, 4, 2, -2, -4],  # 1 on top
                [0, 0, 0, 0, 0, 0],  # nothing on top (-1)
            ],  # nothing on top
            dtype="int32",
        )

        top_of_stack = vpda.top_of_stack()
        self.assertTrue(np.array_equal(top_of_stack, np.array([1, -1])))

    def test_top_of_empty_stack(self):
        vpda = TestVPDA.vpda
        vpda.stacks = np.zeros((4, 0), dtype="int32")

        top_of_stack = vpda.top_of_stack()
        self.assertTrue(np.array_equal(top_of_stack, np.array([-1, -1, -1, -1])))

    def test_get_input_mask(self):
        vpda = TestVPDA.make_anbn_pda()
        vpda.initialize(1)

        #                           0     START  END   A      B     PAD
        expected = np.array([[False, False, True, True, False, False]])
        allowed = vpda.get_input_mask()
        self.assertTrue(np.array_equal(allowed, expected))

        vpda.next(np.array([A]))

        #                           0     START  END   A      B     PAD
        expected = np.array([[False, False, False, True, True, False]])
        allowed = vpda.get_input_mask()
        self.assertTrue(np.array_equal(allowed, expected))

        vpda.next(np.array([B]))

        #                           0     START  END   A      B     PAD
        expected = np.array([[False, False, True, False, False, False]])
        allowed = vpda.get_input_mask()
        self.assertTrue(np.array_equal(allowed, expected))

        vpda.next(np.array([END]))

        #                           0    START   END    A      B      PAD
        expected = np.array([[False, False, False, False, False, True]])
        allowed = vpda.get_input_mask()
        self.assertTrue(np.array_equal(allowed, expected))

    def test_a_n_b_n_accepts(self):
        vpda = TestVPDA.make_anbn_pda()

        accepts = vpda.accepts(
            torch.tensor(
                [
                    [END, PAD, PAD, PAD, PAD, PAD, PAD],  # True
                    [A, A, A, B, B, B, END],  # True
                    [A, A, B, END, PAD, PAD, PAD],  # False
                    [B, A, B, B, END, PAD, PAD],  # False
                    [A, A, B, B, END, PAD, PAD],  # True
                    [A, B, END, PAD, PAD, PAD, PAD],  # True
                    [A, B, B, END, PAD, PAD, PAD],  # False
                    [START, END, PAD, PAD, PAD, PAD, PAD],  # False
                ]
            )
        )

        self.assertTrue(
            np.array_equal(
                accepts,
                torch.tensor([True, True, False, False, True, True, False, False]),
            )
        )

    def test_display_transitions(self):
        input_labels = ["START", "A", "B", "END", "PAD"]
        symbol_labels = ["START", "A", "END"]
        state_labels = ["START", "A", "B", "END"]

        vpda = TestVPDA.make_anbn_pda()
        vpda.initialize(1)

        vpda.display_transitions(
            input_labels=input_labels,
            state_labels=state_labels,
            symbol_labels=symbol_labels,
        )


class TestObjectVPDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = [
            {"animal": "Burro", "gender": "Female", "color": "white"},
            {"animal": "Hamsters", "gender": "Male", "color": "white"},
            {"animal": "Peafowl", "gender": "Male", "color": "white"},
            {"animal": "Turkeys", "gender": "Male", "color": "brown"},
            {"animal": "Sheep", "gender": "Male", "color": "black"},
            {"animal": "Dogs", "gender": "Male", "color": "mixed"},
            {"animal": "Cats", "gender": "Male", "color": "white"},
            {"animal": "Chameleons", "gender": "Male", "color": "mixed"},
            {"animal": "Mice", "gender": "Male", "color": ["white", "black", "mixed"]},
            {"animal": "Hamsters", "gender": "Male", "color": "mixed"},
            {"animal": "Turkeys", "gender": "Male", "color": "brown"},
            {"animal": "Birds", "gender": "Female", "color": "black"},
            {"animal": "Rats", "gender": "Male", "color": "white"},
            {"animal": "Snakes", "gender": "Female", "color": "white"},
            {"animal": "Turkeys", "gender": "Male", "color": "mixed"},
            {"animal": "Rabbits", "gender": "Female", "color": "white"},
            {"animal": "Chameleons", "gender": "Female", "color": "black"},
            {"animal": "Turkeys", "gender": "Male", "color": "brown"},
            {"animal": "Hamsters", "gender": "Female", "color": "white"},
            {"animal": "Gerbils", "gender": "Female", "color": "white"},
        ]

        pipeline = Pipeline(
            [
                ("tokenizer", DocTokenizerPipe()),
                ("padder", PadTruncTokensPipe()),
                ("encoder", TokenEncoderPipe()),
            ]
        )

        df = pd.DataFrame({"docs": data})
        df = pipeline.fit(df)

        # attach to class for use in tests
        cls.data = data
        cls.encoder = pipeline["encoder"].encoder
        cls.vpda = ObjectVPDA(cls.encoder)

    def test_accepts_top_level_documents(self):
        vpda = TestObjectVPDA.vpda
        encoder = TestObjectVPDA.encoder

        accepts = vpda.accepts(
            torch.tensor(
                [
                    [
                        encoder.encode(Symbol.START),
                        encoder.encode(FieldToken("animal")),
                        encoder.encode("Birds"),
                        encoder.encode(FieldToken("color")),
                        encoder.encode("mixed"),
                        encoder.encode(Symbol.END),
                    ],  # True
                    [
                        encoder.encode(Symbol.START),
                        encoder.encode(FieldToken("animal")),
                        encoder.encode("white"),
                        encoder.encode(FieldToken("color")),
                        encoder.encode("Turkeys"),
                        encoder.encode(Symbol.END),
                    ],  # True (TODO: This needs to be False)
                    [
                        encoder.encode(Symbol.START),
                        encoder.encode("white"),
                        encoder.encode(FieldToken("color")),
                        encoder.encode(Symbol.END),
                        encoder.encode(Symbol.PAD),
                        encoder.encode(Symbol.PAD),
                    ],
                    [  # False
                        encoder.encode(Symbol.START),
                        encoder.encode(FieldToken("color")),
                        encoder.encode("black"),
                        encoder.encode(Symbol.END),
                        encoder.encode(Symbol.PAD),
                        encoder.encode(Symbol.PAD),
                    ],  # True
                ]
            )
        )

        self.assertTrue(
            np.array_equal(
                accepts,
                torch.tensor([True, True, False, True]),
            )
        )

    def test_accepts_subdocuments(self):
        vpda = TestObjectVPDA.vpda
        encoder = TestObjectVPDA.encoder

        accepts = vpda.accepts(
            torch.tensor(
                [
                    [
                        encoder.encode(Symbol.START),
                        encoder.encode(FieldToken("animal")),
                        encoder.encode(Symbol.SUBDOC_START),
                        encoder.encode(FieldToken("gender")),
                        encoder.encode("Female"),
                        encoder.encode(Symbol.SUBDOC_END),
                        encoder.encode(FieldToken("color")),
                        encoder.encode("mixed"),
                        encoder.encode(Symbol.END),
                    ],  # True
                    [
                        encoder.encode(Symbol.START),
                        encoder.encode(FieldToken("animal")),
                        encoder.encode(FieldToken("color")),
                        encoder.encode("white"),
                        encoder.encode(FieldToken("gender")),
                        encoder.encode("Male"),
                        encoder.encode(Symbol.END),
                        encoder.encode(Symbol.PAD),
                        encoder.encode(Symbol.PAD),
                    ],  # False
                    [
                        encoder.encode(Symbol.START),
                        encoder.encode(FieldToken("color")),
                        encoder.encode(Symbol.SUBDOC_START),
                        encoder.encode(Symbol.SUBDOC_END),
                        encoder.encode(Symbol.END),
                        encoder.encode(Symbol.PAD),
                        encoder.encode(Symbol.PAD),
                        encoder.encode(Symbol.PAD),
                        encoder.encode(Symbol.PAD),
                    ],  # True
                    [
                        encoder.encode(Symbol.START),
                        encoder.encode(FieldToken("animal")),
                        encoder.encode("Mice"),
                        encoder.encode(Symbol.SUBDOC_START),
                        encoder.encode(FieldToken("color")),
                        encoder.encode("mixed"),
                        encoder.encode(Symbol.SUBDOC_END),
                        encoder.encode(Symbol.END),
                        encoder.encode(Symbol.PAD),
                    ],  # False
                ]
            )
        )

        self.assertTrue(
            np.array_equal(
                accepts,
                torch.tensor([True, False, True, False]),
            )
        )

    def test_accepts_animals_from_pipeline(self):
        pipeline = Pipeline(
            [("tokenizer", DocTokenizerPipe()), ("padder", PadTruncTokensPipe()), ("encoder", TokenEncoderPipe())]
        )

        df = pd.DataFrame({"docs": TestObjectVPDA.data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        encoder = pipeline["encoder"].encoder
        vpda = ObjectVPDA(encoder)

        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == True).all())

        print(vpda.stacks)

    def test_accepts_unknown_prim_value(self):
        data = [
            {"foo": 1, "bar": Symbol.UNKNOWN},
            # {"foo": 1, "bar": {"baz": Symbol.UNKNOWN}},
            # {"foo": 1, "bar": [1, Symbol.UNKNOWN, 3]},
        ]

        pipeline = Pipeline(
            [("tokenizer", DocTokenizerPipe()), ("padder", PadTruncTokensPipe()), ("encoder", TokenEncoderPipe())]
        )

        df = pd.DataFrame({"docs": data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        encoder = pipeline["encoder"].encoder
        vpda = ObjectVPDA(encoder)
        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == True).all())

    def test_accepts_unknown_prim_value_with_schema(self):
        data = TestObjectVPDA.data

        pipeline = Pipeline(
            [("tokenizer", DocTokenizerPipe()), ("padder", PadTruncTokensPipe()), ("encoder", TokenEncoderPipe())]
        )

        df = pd.DataFrame({"docs": TestObjectVPDA.data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        encoder = pipeline["encoder"].encoder
        schema = parse_schema(data)

        # accept original documents
        vpda = ObjectVPDA(encoder, schema)
        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == True).all())

        # accept documents with UNKNOWN values
        unknown_docs = [
            {"animal": Symbol.UNKNOWN, "gender": "Female", "color": "white"},
            {"animal": "Hamsters", "gender": "Male", "color": Symbol.UNKNOWN},
        ]
        df = pd.DataFrame({"docs": unknown_docs})
        df = pipeline.fit_transform(df)
        unknown_tokens = torch.tensor(df["tokens"])
        accepts = vpda.accepts(unknown_tokens)
        self.assertTrue((accepts == True).all())

    def test_accepts_unknown_field_token_with_schema(self):
        data = TestObjectVPDA.data

        pipeline = Pipeline(
            [("tokenizer", DocTokenizerPipe()), ("padder", PadTruncTokensPipe()), ("encoder", TokenEncoderPipe())]
        )

        df = pd.DataFrame({"docs": TestObjectVPDA.data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        encoder = pipeline["encoder"].encoder
        schema = parse_schema(data)

        # accept original documents
        vpda = ObjectVPDA(encoder, schema)
        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == True).all())

        # accept documents with UNKNOWN fields and any primitive values, arrays, sub_docs
        unknown_docs = [
            {"unknown": "Mice", "gender": "Female", "color": "white"},
            {"animal": "Hamsters", "ANOTHER_UNKNOWN_FIELD": "Birds", "color": "black"},
            {"UNKNOWN_FIELD": "UNKNOWN_VALUE", "gender": "Male", "color": "black"},
            {"UNKNOWN_FIELD": {"foo": "bar"}, "gender": "Male", "color": "black"},
            {"UNKNOWN_FIELD": ["Male", "Hamster", "black"], "gender": "Male", "color": "black"},
        ]
        df = pd.DataFrame({"docs": unknown_docs})
        df = pipeline.transform(df)
        unknown_tokens = torch.tensor(df["tokens"])
        accepts = vpda.accepts(unknown_tokens)
        self.assertTrue((accepts == True).all())

    def test_accepts_complex_from_pipeline(self):
        data = [
            {"foo": 1, "bar": 2},
            {"foo": 1, "bar": {"baz": 2}},
            {"foo": [], "bar": [1, 2, 3]},
            {"foo": [[]], "bar": [{"baz": 1}, {"baz": 2}]},
            {"foo": [[1], [2], [[3]]]},
            {"foo": {"bar": [{}, {}, {"baz": 1}, 1, []]}},
        ]

        pipeline = Pipeline(
            [("tokenizer", DocTokenizerPipe()), ("padder", PadTruncTokensPipe()), ("encoder", TokenEncoderPipe())]
        )

        df = pd.DataFrame({"docs": TestObjectVPDA.data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        encoder = pipeline["encoder"].encoder
        vpda = ObjectVPDA(encoder)
        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == True).all())

    def test_rejects_complex_from_pipeline(self):
        data = [
            {"foo": 1, "bar": 2},
            {"foo": 1, "bar": {"baz": 2}},
            {"foo": [], "bar": [1, 2, 3]},
            {"foo": [[]], "bar": [{"baz": 1}, {"baz": 2}]},
            {"foo": [[1], [2], [[3]]]},
            {"foo": {"bar": [{}, {}, {"baz": 1}, 1, []]}},
        ]

        pipeline = Pipeline(
            [("tokenizer", DocTokenizerPipe()), ("padder", PadTruncTokensPipe()), ("encoder", TokenEncoderPipe())]
        )

        df = pd.DataFrame({"docs": data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        correct_tokens = tokens.clone()

        encoder = pipeline["encoder"].encoder

        # for debugging: print out tokens as a DataFrame
        # pd.set_option("display.expand_frame_repr", False)
        # df = pd.DataFrame(pipeline.context["encoder"].decode(tokens))
        # print(df.transpose())

        # forget START token
        tokens[0, :-1] = correct_tokens[0, 1:]
        tokens[0, -1] = encoder.encode(Symbol.PAD)

        # forget SUBDOC_END token
        tokens[1, 7] = encoder.encode(Symbol.END)
        tokens[1, 8] = encoder.encode(Symbol.PAD)

        # incorrect array length
        tokens[2, 4] = encoder.encode(ArrayStart(2))

        # produced value instead of FieldToken
        tokens[3, 7] = encoder.encode(1)

        # produced END instead of PAD while in END state
        tokens[4, 13] = encoder.encode(Symbol.END)

        # forgot to close one subdoc with SUBDOC_END
        tokens[5, 15] = encoder.encode(Symbol.END)
        tokens[5, 16] = encoder.encode(Symbol.PAD)

        vpda = ObjectVPDA(encoder)
        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == False).all())

    def test_rejects_non_pad_tokens_after_end(self):
        data = [
            {"foo": 1, "bar": 2},
            {"foo": 1, "bar": 2},
            {"foo": 1, "bar": 2},
            {"foo": 1, "bar": 2},
        ]

        pipeline = Pipeline(
            [
                ("tokenizer", DocTokenizerPipe()),
                ("padder", PadTruncTokensPipe(length=8)),
                ("encoder", TokenEncoderPipe()),
            ]
        )

        df = pd.DataFrame({"docs": data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        encoder = pipeline["encoder"].encoder

        # for debugging: print out tokens as a DataFrame
        # pd.set_option("display.expand_frame_repr", False)
        # df = pd.DataFrame(encoder.decode(tokens))
        # print(df.transpose())

        # SUBDOC_START token after END
        tokens[0, 6] = encoder.encode(Symbol.SUBDOC_START)

        # SUBDOC_START token after PAD
        tokens[1, 7] = encoder.encode(Symbol.SUBDOC_START)

        # END token after END
        tokens[2, 6] = encoder.encode(Symbol.END)

        # value after PAD
        tokens[3, 7] = encoder.encode(1)

        vpda = ObjectVPDA(encoder)
        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == False).all())

    def test_accepts_values_with_schema(self):
        data = TestObjectVPDA.data

        pipeline = Pipeline(
            [("tokenizer", DocTokenizerPipe()), ("padder", PadTruncTokensPipe()), ("encoder", TokenEncoderPipe())]
        )

        df = pd.DataFrame({"docs": data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        encoder = pipeline["encoder"].encoder
        schema = parse_schema(data)

        # accept original documents
        vpda = ObjectVPDA(encoder, schema)
        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == True).all())

        # reject documents that have values from other fields
        bad_docs = [
            {"animal": "Female", "gender": "Female", "color": "white"},
            {"animal": "Hamsters", "gender": "Male", "color": "Female"},
            {"animal": "white", "gender": "Male", "color": "white"},
            {"animal": "Turkeys", "gender": "mixed", "color": "brown"},
        ]

        df = pd.DataFrame({"docs": bad_docs})
        df = pipeline.fit_transform(df)
        bad_tokens = torch.tensor(df["tokens"])

        rejects = vpda.accepts(bad_tokens)
        self.assertTrue((rejects == False).all())

    def test_accepts_arrays_with_schema(self):
        data = TestObjectVPDA.data

        pipeline = Pipeline(
            [("tokenizer", DocTokenizerPipe()), ("padder", PadTruncTokensPipe()), ("encoder", TokenEncoderPipe())]
        )

        df = pd.DataFrame({"docs": data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        encoder = pipeline["encoder"].encoder
        schema = parse_schema(data)

        # accept original documents
        vpda = ObjectVPDA(encoder, schema)
        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == True).all())

        # accept documents with arrays of primitive values from the same field
        array_docs = [
            {"animal": "Mice", "gender": "Female", "color": ["white", "black"]},
            {"animal": [], "gender": "Female", "color": ["white", "black"]},
            {
                "animal": "Mice",
                "gender": ["Female", "Male"],
                "color": ["white", "black"],
            },
        ]

        df = pd.DataFrame({"docs": array_docs})
        df = pipeline.fit_transform(df)
        array_tokens = torch.tensor(df["tokens"])

        accepts = vpda.accepts(array_tokens)
        self.assertTrue((accepts == True).all())

    def test_get_input_mask_with_schema(self):
        data = TestObjectVPDA.data

        pipeline = Pipeline(
            [("tokenizer", DocTokenizerPipe()), ("padder", PadTruncTokensPipe()), ("encoder", TokenEncoderPipe())]
        )

        df = pd.DataFrame({"docs": data})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])
        encoder = pipeline["encoder"].encoder
        schema = parse_schema(data)

        # accept original documents
        vpda = ObjectVPDA(encoder, schema)
        accepts = vpda.accepts(tokens)
        self.assertTrue((accepts == True).all())

        # does not accept documents with primitive values from different fields
        # including primitive values inside of arrays
        array_docs = [
            # example of correct document
            {"animal": "Mice", "gender": "Female", "color": ["white", "black"]},
            # example where primitive value does not match the field
            {"animal": "Female", "gender": "Mice", "color": ["white", "black"]},
            # example where primitive values inside arrays don't all match the field
            {
                "animal": "Mice",
                "gender": ["Female", "white"],
                "color": ["Male", "black"],
            },
        ]
        df = pd.DataFrame({"docs": array_docs})
        df = pipeline.fit_transform(df)
        array_tokens = torch.tensor(df["tokens"])

        accepts = vpda.accepts(array_tokens)
        expected = torch.tensor([True, False, False])
        self.assertTrue(np.array_equal(accepts, expected))
