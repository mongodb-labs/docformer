import unittest

import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from origami.model.positions import (
    BasePositionEncoding,
    IntegerPositionEncoding,
    KeyValuePositionEncoding,
    SineCosinePositionEncoding,
)
from origami.model.vpda import ObjectVPDA
from origami.preprocessing import (
    DocTokenizerPipe,
    PadTruncTokensPipe,
    SchemaParserPipe,
    TokenEncoderPipe,
)
from origami.utils.common import ArrayStart, FieldToken, Symbol


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


class TestSineCosinePositionEncoding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.seq_len = 16
        self.embedding_dim = 32
        self.block_size = 64

        self.encoder = SineCosinePositionEncoding(
            block_size=self.block_size, embedding_dim=self.embedding_dim, fuse_with_mlp=False
        )

        self.encoder_with_mlp = SineCosinePositionEncoding(
            block_size=self.block_size, embedding_dim=self.embedding_dim, fuse_with_mlp=True
        )

    def test_output_shape(self):
        """Test if the output shape matches the input shape"""
        tok_emb = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        output = self.encoder(tok_emb)

        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.embedding_dim), "Output shape should match input shape"
        )

    def test_positional_encoding_pattern(self):
        """Test if the positional encoding follows the expected sine/cosine pattern"""
        # Get the raw positional encoding matrix
        pe = self.encoder.pe[0]  # Remove batch dimension

        # Test first position (pos = 0)
        self.assertAlmostEqual(
            pe[0, 0].item(),  # sin(0) = 0
            0.0,
            places=6,
            msg="First position, first dimension should be sin(0) = 0",
        )

        self.assertAlmostEqual(
            pe[0, 1].item(),  # cos(0) = 1
            1.0,
            places=6,
            msg="First position, second dimension should be cos(0) = 1",
        )

    def test_different_sequence_lengths(self):
        """Test if the encoder handles different sequence lengths correctly"""
        # Test with shorter sequence
        short_tok_emb = torch.randn(self.batch_size, 5, self.embedding_dim)
        short_output = self.encoder(short_tok_emb)
        self.assertEqual(short_output.shape, (self.batch_size, 5, self.embedding_dim))

        # Test with longer sequence
        long_tok_emb = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        long_output = self.encoder(long_tok_emb)
        self.assertEqual(long_output.shape, (self.batch_size, self.seq_len, self.embedding_dim))

    def test_mlp_fusion(self):
        """Test if MLP fusion works correctly"""
        tok_emb = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        output = self.encoder_with_mlp(tok_emb)

        # Check output shape (should match input shape due to final MLP layer)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.embedding_dim),
            "MLP fusion output shape should match input shape",
        )

        # Check that output is different from simple addition
        simple_output = self.encoder(tok_emb)
        self.assertFalse(
            torch.allclose(output, simple_output), "MLP fusion should produce different results from simple addition"
        )

    def test_periodicity(self):
        """Test if the encoding has the expected periodicity properties"""
        pe = self.encoder.pe[0]  # Remove batch dimension

        # For dimension d, the wavelength should be 10000^(2d/embedding_dim)
        d = 0  # First dimension
        wavelength = 10000 ** (2 * d / self.embedding_dim)

        # Check if values repeat with the expected period
        pos1 = 0
        pos2 = int(wavelength / 2)  # Half wavelength for sine should give opposite values

        self.assertAlmostEqual(
            pe[pos1, d].item(), -pe[pos2, d].item(), places=4, msg="Sine values should be opposite at half wavelength"
        )

    def test_output_range(self):
        """Test if the output values are in a reasonable range"""
        tok_emb = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        output = self.encoder(tok_emb)

        # Check if output values are not exploding
        self.assertTrue(torch.all(torch.isfinite(output)), "Output should not contain inf or nan values")

        # Check if positional encoding values are bounded
        pe = self.encoder.pe
        self.assertTrue(
            torch.all(pe >= -1) and torch.all(pe <= 1), "Positional encoding values should be bounded between -1 and 1"
        )

    def test_device_compatibility(self):
        """Test if the encoder works on different devices"""
        if torch.cuda.is_available():
            encoder_cuda = SineCosinePositionEncoding(
                block_size=self.block_size, embedding_dim=self.embedding_dim
            ).cuda()

            tok_emb = torch.randn(self.batch_size, self.seq_len, self.embedding_dim, device="cuda")

            output = encoder_cuda(tok_emb)
            self.assertEqual(output.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()


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
