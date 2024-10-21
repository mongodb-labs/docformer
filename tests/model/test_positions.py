import unittest

import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from origami.preprocessing import (
    DocTokenizerPipe,
    PadTruncTokensPipe,
    SchemaParserPipe,
    TokenEncoderPipe,
)
from origami.model.positions import (
    BasePositionEncoding,
    KeyValuePositionEncoding,
    IntegerPositionEncoding,
)
from origami.utils.common import ArrayStart, FieldToken, Symbol
from origami.model.vpda import ObjectVPDA



class TestBasePositionEncoding(unittest.TestCase):
    def test_init_no_fuse_with_mlp(self):
        base_pos_enc = BasePositionEncoding(16, fuse_with_mlp=False)
        self.assertEqual(base_pos_enc.fuse_with_mlp, False)
        self.assertIsNone(getattr(base_pos_enc, "fuse_mlp", None))

    def test_init_fuse_with_mlp(self):
        base_pos_enc = BasePositionEncoding(16, fuse_with_mlp=True)
        self.assertEqual(base_pos_enc.fuse_with_mlp, True)
        self.assertIsInstance(base_pos_enc.fuse_mlp, torch.nn.Module)

        # confirm layer sizes of fuse_mlp
        fuse_mlp_sizes = [
            (m.in_features, m.out_features) for m in base_pos_enc.fuse_mlp.children() if isinstance(m, torch.nn.Linear)
        ]
        expected_sizes = [(2 * 16, 2 * 16), (2 * 16, 4 * 16), (4 * 16, 16)]
        self.assertEqual(fuse_mlp_sizes, expected_sizes)

    def test_init_fuse_with_mlp_custom_layers(self):
        base_pos_enc = BasePositionEncoding(16, fuse_with_mlp=True, mlp_layer_factors=[3, 6, 6, 1])
        self.assertEqual(base_pos_enc.fuse_with_mlp, True)
        self.assertIsInstance(base_pos_enc.fuse_mlp, torch.nn.Module)

        # confirm layer sizes of fuse_mlp
        fuse_mlp_sizes = [
            (m.in_features, m.out_features) for m in base_pos_enc.fuse_mlp.children() if isinstance(m, torch.nn.Linear)
        ]
        expected_sizes = [
            (2 * 16, 3 * 16),
            (3 * 16, 6 * 16),
            (6 * 16, 6 * 16),
            (6 * 16, 16),
        ]
        self.assertEqual(fuse_mlp_sizes, expected_sizes)


class TestIntegerPositionEncoding(unittest.TestCase):
    def test_forward_sum(self):
        tok_emb = torch.rand((4, 8, 16))

        pos_enc = IntegerPositionEncoding(8, 16, fuse_with_mlp=False)
        x = pos_enc(tok_emb)

        for i in range(8):
            pos_emb = pos_enc.embedding(torch.tensor([i], dtype=torch.long))
            self.assertTrue(torch.equal(tok_emb[:, i, :] + pos_emb, x[:, i, :]))

    def test_forward_fused(self):
        tok_emb = torch.rand((4, 8, 16))

        pos_enc = IntegerPositionEncoding(8, 16, fuse_with_mlp=True)
        x = pos_enc(tok_emb)

        # here we can only test that the output shape matches the input
        self.assertEqual(x.shape, tok_emb.shape)


class TestKeyValuePositionEncoding(unittest.TestCase):
    def test_forward_sum_subdoc(self):
        docs = [
            {"foo": {"bar": 1}, "baz": 2},
        ]

        pipeline = Pipeline(
            [
                ("schema", SchemaParserPipe()),
                ("tokenizer", DocTokenizerPipe()),
                ("padder", PadTruncTokensPipe(length=10)),
                ("encoder", TokenEncoderPipe()),
            ]
        )

        df = pd.DataFrame({"docs": docs})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])

        schema = pipeline["schema"].schema
        encoder = pipeline["encoder"].encoder

        pos_enc = KeyValuePositionEncoding(encoder.vocab_size, 16, fuse_with_mlp=False)

        # encode the document

        vpda = ObjectVPDA(encoder, schema)
        vpda.accepts(tokens)
        stacks = torch.tensor(vpda.stacks)

        # we pass in a zero tensor for the token embeddings so we can compare the
        # position embeddings in isolation
        pos_emb = pos_enc(torch.zeros((tokens.size(0), tokens.size(1), 16)), stacks)

        # The stack for the document should look like this (FT = FieldToken)
        #
        #                                  FT(foo.bar)
        #                FT(foo)   FT(foo)   FT(foo)    FT(foo)        FT(baz)
        #  START   DOC     DOC       DOC       DOC        DOC     DOC    DOC     DOC   END

        stack_symbols = [
            [Symbol.START],
            [Symbol.DOC],
            [Symbol.DOC, FieldToken("foo")],
            [Symbol.DOC, FieldToken("foo")],
            [Symbol.DOC, FieldToken("foo"), FieldToken("foo.bar")],
            [Symbol.DOC, FieldToken("foo")],
            [Symbol.DOC],
            [Symbol.DOC, FieldToken("baz")],
            [Symbol.DOC],
            [Symbol.END],
        ]

        # construct target embeddings manually by adding up embedded stack symbols
        emb_matrix = pos_enc.embedding.weight
        target_embeddings = torch.zeros_like(pos_emb)
        for i, stack in enumerate(stack_symbols):
            for symbol in stack:
                token = encoder.encode(symbol)
                embedding = emb_matrix[token]
                target_embeddings[0, i] += embedding

        self.assertTrue(torch.isclose(pos_emb, target_embeddings).all())

    def test_forward_sum_array_of_subdocs(self):
        docs = [
            {"foo": [{"bar": 1}, {"bar": 2}]},
        ]

        pipeline = Pipeline(
            [
                ("schema", SchemaParserPipe()),
                ("tokenizer", DocTokenizerPipe()),
                ("padder", PadTruncTokensPipe(length=13)),
                ("encoder", TokenEncoderPipe()),
            ]
        )

        df = pd.DataFrame({"docs": docs})
        df = pipeline.fit_transform(df)
        tokens = torch.tensor(df["tokens"])

        schema = pipeline["schema"].schema
        encoder = pipeline["encoder"].encoder

        pos_enc = KeyValuePositionEncoding(encoder.vocab_size, 16, fuse_with_mlp=False)

        vpda = ObjectVPDA(encoder, schema)
        vpda.accepts(tokens)
        stacks = torch.tensor(vpda.stacks)

        # we pass in a zero tensor for the token embeddings so we can compare the
        # position embeddings in isolation
        pos_emb = pos_enc(torch.zeros((tokens.size(0), tokens.size(1), 16)), stacks)

        stack_symbols = [
            [Symbol.START],
            [Symbol.DOC],
            [Symbol.DOC, FieldToken("foo")],
            [Symbol.DOC, FieldToken("foo"), ArrayStart(2)],
            [Symbol.DOC, FieldToken("foo"), ArrayStart(2)],  # <- noop (subdoc start)
            [Symbol.DOC, FieldToken("foo"), ArrayStart(2), FieldToken("foo.bar")],
            [Symbol.DOC, FieldToken("foo"), ArrayStart(2)],
            [Symbol.DOC, FieldToken("foo")],
            [Symbol.DOC, FieldToken("foo")],  # <- noop (subdoc start)
            [Symbol.DOC, FieldToken("foo"), FieldToken("foo.bar")],
            [Symbol.DOC, FieldToken("foo")],
            [Symbol.DOC],
            [Symbol.END],
        ]

        # construct target embeddings manually by adding up embedded stack symbols
        emb_matrix = pos_enc.embedding.weight
        target_embeddings = torch.zeros_like(pos_emb)
        for i, stack in enumerate(stack_symbols):
            for symbol in stack:
                token = encoder.encode(symbol)
                embedding = emb_matrix[token]
                target_embeddings[0, i] += embedding

        self.assertTrue(torch.isclose(pos_emb, target_embeddings).all())
