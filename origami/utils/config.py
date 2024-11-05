from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from omegaconf import MISSING, OmegaConf


class SequenceOrderMethod(Enum):
    ORDERED = 1
    SHUFFLED = 2


class NumericMethod(Enum):
    NONE = 1
    BINNING = 2


class PositionEncodingMethod(Enum):
    NONE = 1
    INTEGER = 2
    SINE_COSINE = 3
    KEY_VALUE = 4


class GuardrailsMethod(Enum):
    NONE = 1
    STRUCTURE_ONLY = 2
    STRUCTURE_AND_VALUES = 3


class BaseConfig:
    """Wrap all configs deriving from BaseConfig in OmegaConf objects."""

    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        cls.__init__(instance, **kwargs)
        return OmegaConf.structured(instance)


@dataclass(kw_only=True)
class ModelConfig(BaseConfig):
    """Model config for creating an ORiGAMi model."""

    # architecture parameters
    n_layer: int = MISSING
    n_head: int = MISSING
    n_embd: int = MISSING
    vocab_size: int = MISSING
    block_size: int = MISSING

    # dropout hyperparameters
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    # weight tieing between embedding matrix and linear head weights (saves parameters)
    tie_weights: bool = False

    # the position encoding method to use (NONE, INTEGER, KEY_VALUE)
    position_encoding: PositionEncodingMethod = PositionEncodingMethod.KEY_VALUE

    # Whether to use an MLP to fuse position and token embeddings or just sum them
    fuse_pos_with_mlp: bool = False

    # whether or not to mask field tokens in loss calculation
    mask_field_token_losses: bool = False

    # whether or not to use guardrails (requires a ObjectVPDA to be passed into model)
    guardrails: GuardrailsMethod = GuardrailsMethod.STRUCTURE_ONLY

    @staticmethod
    def from_preset(size: str, **kwargs) -> "ModelConfig":
        match size:
            case "xs":
                return ModelConfig(n_layer=3, n_head=1, n_embd=48, **kwargs)
            case "small":
                return ModelConfig(n_layer=4, n_head=4, n_embd=128, **kwargs)
            case "medium":
                return ModelConfig(n_layer=6, n_head=6, n_embd=192, **kwargs)
            case "large":
                return ModelConfig(n_layer=8, n_head=8, n_embd=384, **kwargs)
            case "xl":
                return ModelConfig(n_layer=12, n_head=12, n_embd=768, **kwargs)
            case _:
                raise ValueError(f"Unknown model size: {size}")


@dataclass(repr=True, kw_only=True)
class TrainConfig(BaseConfig):
    """Train config for training a GPT model."""

    # device to train on (auto, cpu, cuda, mps)
    device: str = "auto"

    # dataloder parameters (currently not used)
    num_workers: int = 0

    # optimizer parameters
    batch_size: int = 100
    n_batches: int = MISSING
    n_warmup_batches: int = 1000
    learning_rate: float = 1e-3
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1  # only applied on matmul weights
    grad_norm_clip: float = 1.0
    lr_end_factor: float = 0.01

    # validation options
    test_split: float = 0.0
    shuffle_split: bool = True

    # print and eval options
    print_every: int = 100
    eval_every: int = 1000
    sample_train: int = 100
    sample_test: int = 100


@dataclass(repr=True, kw_only=True)
class PipelineConfig(BaseConfig):
    """Config needed to build and execute pipelines."""

    # pipeline options
    max_vocab_size: int = 0
    numeric_method: NumericMethod = NumericMethod.BINNING
    n_bins: int = 100
    sequence_order: SequenceOrderMethod = MISSING
    upscale: int = 1
    path_in_field_tokens: bool = True


@dataclass(repr=True, kw_only=True)
class DataConfig(BaseConfig):
    """Config to define data source and target field."""

    # source string
    source: str = MISSING

    # database parameters
    db: Optional[str] = MISSING
    coll: Optional[str] = MISSING
    projection: dict = field(default_factory=lambda: {"_id": 0})
    skip: int = 0
    limit: int = 0

    # prediction options
    target_field: Optional[str] = MISSING


@dataclass(repr=True, kw_only=True)
class TopLevelConfig(BaseConfig):
    """combined Model/Train/Data/Pipeline config."""

    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    pipeline: PipelineConfig = PipelineConfig()
