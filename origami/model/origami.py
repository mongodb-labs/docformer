import time

# suppress deprecation warning for setting epoch in LR scheduler, likely bug in pytorch
import warnings
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data.dataloader import DataLoader, default_collate

from origami.model.vpda import ObjectVPDA
from origami.preprocessing import DFDataset
from origami.utils import auto_device, torch_isin
from origami.utils.config import GuardrailsMethod, ModelConfig, PositionEncodingMethod, TrainConfig

from .positions import (
    IntegerPositionEncoding,
    SharedKeyValuePositionEncoding,
    SineCosinePositionEncoding,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.optim.lr_scheduler",
    message=r".*The epoch parameter in `scheduler\.step\(\)`.*",
)


class ORIGAMI(nn.Module):
    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = auto_device(device)
        self.to(device)

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig, vpda: Optional[ObjectVPDA] = None):
        super().__init__()

        # store arguments
        self.model_config = model_config
        self.train_config = train_config
        self.vpda = vpda

        self.optimizer = None
        self.scheduler = None
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        self._device = auto_device(train_config.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.batch_num = 0
        self.batch_time = 0.0
        self.batch_dt = 0.0
        self.epoch_num = 0
        self.loss = 0.0

        # creating the embeddings layers and model
        self.token_embed = nn.Embedding(self.model_config.vocab_size, self.model_config.n_embd, padding_idx=0)

        # create position encoding layer
        match self.model_config.position_encoding:
            case PositionEncodingMethod.INTEGER:
                self.pos_encoding = IntegerPositionEncoding(
                    self.model_config.block_size,
                    self.model_config.n_embd,
                    fuse_with_mlp=self.model_config.fuse_pos_with_mlp,
                )
            case PositionEncodingMethod.SINE_COSINE:
                self.pos_encoding = SineCosinePositionEncoding(
                    self.model_config.block_size,
                    self.model_config.n_embd,
                    fuse_with_mlp=self.model_config.fuse_pos_with_mlp,
                )
            case PositionEncodingMethod.KEY_VALUE:
                self.pos_encoding = SharedKeyValuePositionEncoding(
                    self.token_embed,
                    fuse_with_mlp=self.model_config.fuse_pos_with_mlp,
                )
            case PositionEncodingMethod.NONE:
                self.pos_encoding = None
            case _:
                raise ValueError(f"unknown position encoding {self.model_config.position_encoding}")

        encoder_layer = nn.TransformerEncoderLayer(
            self.model_config.n_embd,
            self.model_config.n_head,
            4 * self.model_config.n_embd,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.model_config.n_layer)
        self.head = nn.Linear(self.model_config.n_embd, self.model_config.vocab_size, bias=False)

        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        if self.model_config.tie_weights:
            self.head.weight = self.token_embed.weight

        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def collate_fn(self, tokens: torch.tensor) -> tuple[torch.tensor, torch.tensor, Optional[torch.tensor]]:
        """This function prepares takes a 2D tensor of tokens of shape (batch_size, seq_len) and
        returns inputs, targets and optionally guardrails masks for the model.
        """

        tokens = default_collate(tokens)

        if self.model_config.guardrails == GuardrailsMethod.NONE:
            out_masks = None
        else:
            out_masks = torch.tensor(self.vpda.get_sequence_mask(tokens))[:, 1:].to(self.device)

        # targets are the same as input but shifted by one token
        tokens = tokens.to(self.device)
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()

        return inputs, targets, out_masks

    def train_model(self, dataset: DFDataset, epochs: int = None, batches: int = None):
        self.train()

        assert epochs or batches, "Must specify number of epochs or number of batches"

        loader = DataLoader(
            dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
        )

        # setup the optimizer
        self.optimizer = self.configure_optimizer(self.train_config)

        # setup the learning rate scheduler
        warmup_batches = self.train_config.n_warmup_batches
        total_iters = warmup_batches + (batches or len(loader) * epochs)

        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_batches)

        linear_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.train_config.lr_end_factor,
            total_iters=total_iters,
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, linear_scheduler], milestones=[warmup_batches]
        )

        # if resuming training from a checkpoint, advance scheduler according to num_batches
        for _ in range(self.batch_num):
            self.scheduler.step()

        try:
            # if we are training for a number of epochs
            if epochs:
                for _ in range(epochs):
                    for batch in loader:
                        self.train_batch(batch)

                    # epoch callbacks and stats
                    self.trigger_callbacks("on_epoch_end")
                    self.epoch_num += 1

            # if we are training for a number of batches
            if batches:
                target_batches = self.batch_num + batches
                while True:
                    for batch in loader:
                        self.train_batch(batch)
                        if self.batch_num >= target_batches:
                            return

                    self.trigger_callbacks("on_epoch_end")
                    self.epoch_num += 1

        except KeyboardInterrupt:
            print(f"model training interrupted after {self.epoch_num} epochs ({self.batch_num} batches total).")

    def train_batch(self, batch):
        config = self.train_config

        # forward the model
        _, self.loss = self(*batch)

        # backprop and update the parameters
        self.zero_grad(set_to_none=True)
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.grad_norm_clip)
        self.optimizer.step()
        self.scheduler.step()

        # batch callbacks and stats
        self.trigger_callbacks("on_batch_end")
        self.batch_num += 1
        tnow = time.time()
        self.batch_dt = tnow - self.batch_time
        self.batch_time = tnow

    def save(self, path: str) -> None:
        """saves the trained weights to file, including batch and epoch num."""
        torch.save(self.state_dict() | {"batch_num": self.batch_num, "epoch_num": self.epoch_num}, path)

    def load(self, path: str) -> None:
        """loads trained weights from file. Note: You need to initialise the model
        instance with the exact same configuration and dataset."""
        state_dict = torch.load(path, map_location=self.device)

        # load batch and epoch num if available (backwards compatible with old format)
        self.batch_num = state_dict.pop("batch_num", 0)
        self.epoch_num = state_dict.pop("epoch_num", 0)

        self.load_state_dict(state_dict)

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """A forward pass through the model up to the last hidden layer. Call this to get
        access to the embeddings produced by the final hidden layer."""

        b, t = idx.size()
        assert t <= self.model_config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.model_config.block_size}"
        )

        # token embeddings (b, t, n_embd)
        tok_emb = self.token_embed(idx)

        # position embeddings
        match self.model_config.position_encoding:
            case PositionEncodingMethod.INTEGER:
                x = self.pos_encoding(tok_emb)
            case PositionEncodingMethod.SINE_COSINE:
                x = self.pos_encoding(tok_emb)
            case PositionEncodingMethod.KEY_VALUE:
                # use the VPDA stacks for position encoding
                if self.training and self.model_config.guardrails != GuardrailsMethod.NONE:
                    # for guard rails we calculated the stacks on all tokens, but here we
                    # only need it for the inputs, so slicing 2 elements off the end (pop+push symbol)
                    stacks = torch.tensor(self.vpda.stacks[:, :-2]).to(self.device)
                else:
                    # if guard rails are not used, or if this is for inference calculate stacks
                    self.vpda.get_sequence_mask(idx)
                    stacks = torch.tensor(self.vpda.stacks).to(self.device)

                x = self.pos_encoding(tok_emb, stacks)
            case PositionEncodingMethod.NONE:
                x = tok_emb
            case _:
                raise ValueError(f"unknown position encoding {self.model_config.position_encoding}")

        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
        # for now, we have to move mask to device after creating
        # due to a bug on MPS devices, see https://github.com/pytorch/pytorch/issues/116170
        mask = mask.to(self.device)

        return self.transformer(x, mask)

    def forward(self, idx, targets=None, out_mask=None):
        """forward pass through the model."""

        hidden = self.hidden(idx)
        logits = self.head(hidden)

        # set out-of-context logits to -inf, out_mask shape is (b, s, v)
        if out_mask is not None:
            logits.masked_fill_(~out_mask, -torch.inf)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if self.model_config.mask_field_token_losses:
                # mask out field tokens from loss calculation
                field_mask = torch_isin(targets, self.vpda.field_token_ids)
                targets[field_mask] = 0
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss

    def configure_optimizer(self, train_config):
        """return a standard AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=train_config.learning_rate,
            betas=train_config.betas,
        )
        return optimizer
