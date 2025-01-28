import unittest

from origami.preprocessing.pipelines import build_prediction_pipelines
from origami.utils.config import SequenceOrderMethod


class TestBuildPipeline(unittest.TestCase):
    def test_build_prediction_pipelines(self):
        train_pipe, test_pipe = build_prediction_pipelines(
            target_field="foo", n_bins=10, sequence_order=SequenceOrderMethod.ORDERED, upscale=2
        )

        self.assertEqual(
            [list(p)[0] for p in list(train_pipe.get_params()["steps"])],
            ["binning", "schema", "tokenizer", "padding", "encoder"],
        )

        self.assertEqual(
            [list(p)[0] for p in list(test_pipe.get_params()["steps"])],
            ["binning", "target", "tokenizer", "padding", "encoder"],
        )

    def test_build_prediction_pipelines_with_permute(self):
        train_pipe, test_pipe = build_prediction_pipelines(
            target_field="foo", n_bins=10, sequence_order=SequenceOrderMethod.SHUFFLED, upscale=2
        )

        self.assertEqual(
            [list(p)[0] for p in list(train_pipe.get_params()["steps"])],
            ["binning", "schema", "upscaler", "permuter", "tokenizer", "padding", "encoder"],
        )

        self.assertEqual(
            [list(p)[0] for p in list(test_pipe.get_params()["steps"])],
            ["binning", "target", "tokenizer", "padding", "encoder"],
        )
