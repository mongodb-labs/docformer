import unittest

from sklearn.pipeline import Pipeline

from origami.preprocessing.pipelines import build_prediction_pipelines
from origami.utils.config import (
    NumericMethod,
    PipelineConfig,
    SequenceOrderMethod,
)


class TestPredictionPipelines(unittest.TestCase):
    def setUp(self):
        """Set up base configuration for tests"""
        self.base_config = PipelineConfig(
            max_vocab_size=1000,
            numeric_method=NumericMethod.NONE,
            n_bins=100,
            sequence_order=SequenceOrderMethod.ORDERED,
            upscale=1,
            path_in_field_tokens=True,
        )

    def test_basic_pipeline_no_target(self):
        """Test basic pipeline creation without target field"""
        pipelines = build_prediction_pipelines(self.base_config)

        # Check pipeline types
        self.assertIsInstance(pipelines["train"], Pipeline)
        self.assertIsInstance(pipelines["test"], Pipeline)

        # Check pipeline steps (train always includes schema)
        expected_train_steps = ["schema", "tokenizer", "padding", "encoder"]
        expected_test_steps = ["tokenizer", "padding", "encoder"]

        self.assertEqual([step[0] for step in pipelines["test"].steps], expected_test_steps)
        self.assertEqual([step[0] for step in pipelines["train"].steps], expected_train_steps)

    def test_pipeline_with_target(self):
        """Test pipeline creation with target field"""
        pipelines = build_prediction_pipelines(self.base_config, target_field="target_column")

        expected_train_steps = ["target", "schema", "tokenizer", "padding", "encoder"]
        expected_test_steps = ["target", "tokenizer", "padding", "encoder"]

        self.assertEqual([step[0] for step in pipelines["test"].steps], expected_test_steps)
        self.assertEqual([step[0] for step in pipelines["train"].steps], expected_train_steps)

    def test_pipeline_with_binning(self):
        """Test pipeline with numeric binning enabled"""
        config = self.base_config.copy()
        config.numeric_method = NumericMethod.BINNING

        pipelines = build_prediction_pipelines(config, target_field="target_column")

        expected_train_steps = ["binning", "target", "schema", "tokenizer", "padding", "encoder"]
        expected_test_steps = ["binning", "target", "tokenizer", "padding", "encoder"]

        self.assertEqual([step[0] for step in pipelines["test"].steps], expected_test_steps)
        self.assertEqual([step[0] for step in pipelines["train"].steps], expected_train_steps)

    def test_pipeline_with_shuffling(self):
        """Test pipeline with shuffled sequence order"""
        config = self.base_config.copy()
        config.sequence_order = SequenceOrderMethod.SHUFFLED
        config.upscale = 2

        pipelines = build_prediction_pipelines(config)

        expected_train_steps = ["schema", "upscaler", "permuter", "tokenizer", "padding", "encoder"]
        expected_test_steps = ["tokenizer", "padding", "encoder"]

        self.assertEqual([step[0] for step in pipelines["train"].steps], expected_train_steps)
        self.assertEqual([step[0] for step in pipelines["test"].steps], expected_test_steps)

    def test_pipeline_all_features(self):
        """Test pipeline with all features enabled"""
        config = self.base_config.copy()
        config.sequence_order = SequenceOrderMethod.SHUFFLED
        config.numeric_method = NumericMethod.BINNING
        config.upscale = 2

        pipelines = build_prediction_pipelines(config, target_field="target_column")

        expected_train_steps = [
            "binning",
            "target",
            "schema",
            "upscaler",
            "permuter",
            "tokenizer",
            "padding",
            "encoder",
        ]
        expected_test_steps = ["binning", "target", "tokenizer", "padding", "encoder"]

        self.assertEqual([step[0] for step in pipelines["train"].steps], expected_train_steps)
        self.assertEqual([step[0] for step in pipelines["test"].steps], expected_test_steps)

    def test_pipeline_components_configuration(self):
        """Test that pipeline components are configured correctly"""
        config = self.base_config.copy()
        config.max_vocab_size = 2000
        config.n_bins = 50

        pipelines = build_prediction_pipelines(config)

        # Check encoder configuration
        encoder = dict(pipelines["train"].named_steps)["encoder"]
        self.assertEqual(encoder.max_tokens, 2000)

        # Check binning configuration when enabled
        config.numeric_method = NumericMethod.BINNING
        pipelines = build_prediction_pipelines(config)
        binning = dict(pipelines["train"].named_steps)["binning"]
        self.assertEqual(binning.bins, 50)
        self.assertEqual(binning.strategy, "kmeans")


if __name__ == "__main__":
    unittest.main()
