from typing import Optional

import torch
from mdbrtools.schema import Schema
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from origami.model import ORIGAMI
from origami.model.vpda import ObjectVPDA
from origami.preprocessing import DFDataset, StreamEncoder, detokenize, target_collate_fn
from origami.utils import FieldToken, Symbol

from .batch_sampler import TargetTokenBatchSampler


class AutoCompleter:
    def __init__(
        self,
        model: ORIGAMI,
        encoder: StreamEncoder,
        target_field: str,
        schema: Optional[Schema] = None,
        max_batch_size: int = 100,
        show_progress: bool = False,
    ):
        self.model = model
        self.vpda = ObjectVPDA(encoder, schema)
        self.encoder = encoder
        self.target_field = target_field
        self.max_batch_size = max_batch_size
        self.show_progress = show_progress

    @torch.no_grad()
    def autocomplete_batch(self, idx: torch.Tensor) -> torch.Tensor:
        """Autocompletes a batch, assuming all sequences in the batch have the same target token position."""
        self.model.eval()

        pad_token_id = self.encoder.encode(Symbol.PAD)

        guardrails = self.model.model_config.guardrails
        block_size = self.model.model_config.block_size
        device = self.model.device

        idx = idx.to(device)
        indexes = torch.arange(idx.size(0), device=device)

        print("\n", idx.size(), end=" ")
        if guardrails:
            self.vpda.initialize(idx.size(0))
            masks = torch.tensor(self.vpda.get_sequence_mask(idx)[:, 1:])
            next_mask = torch.tensor(self.vpda.get_input_mask()).unsqueeze(1)
            masks = torch.cat((masks, next_mask), dim=1)
            masks = masks.to(device)
        else:
            masks = None

        completed_rows = []

        # generation loop, from current position until end of blocksize (or all pad)
        for _ in range(block_size - idx.size(1)):
            print(".", end="")
            # forward the model, get the logits
            logits, _ = self.model(idx, None, masks)

            # pluck the logits at the final step
            logits = logits[:, -1, :]

            # greedily get the most likely value (arg max)
            idx_next = torch.argmax(logits, dim=-1, keepdims=True)

            # append sampled indexes to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # end early for all sequences have reached PAD tokens
            complete = (idx_next == pad_token_id).view(-1)
            if complete.any():
                completed_rows.extend(zip(indexes[complete], idx[complete]))
                if complete.all():
                    break

                # as samples complete, remove them from generation and VPDA
                idx = idx[~complete, :]
                indexes = indexes[~complete]
                idx_next = idx_next[~complete, :]
                masks = masks[~complete, :]
                complete = complete.cpu().numpy()
                self.vpda.stacks = self.vpda.stacks[~complete, :]
                self.vpda.states = self.vpda.states[~complete]
                self.vpda.n_stacks = (~complete).sum().item()

            if guardrails:
                # update VPDA with sampled tokens and get next mask
                self.vpda.next(idx_next.squeeze())
                next_mask = torch.tensor(self.vpda.get_input_mask()).unsqueeze(1).to(device)
                masks = torch.cat((masks, next_mask), dim=1)

        # switch model back into train mode
        self.model.train()

        # pad, sort rows and turn into full tensor
        idx = pad_sequence(
            [d[1] for d in sorted(completed_rows, key=lambda x: x[0].item())],
            batch_first=True,
            padding_value=pad_token_id,
        )

        return idx

    @torch.no_grad()
    def autocomplete(self, data: torch.Tensor | DFDataset, decode: bool = False) -> list | torch.Tensor:
        self.model.eval()

        pad_token_id = self.encoder.encode(Symbol.PAD)

        # get the target token id so we can chop off the input after the target and only
        # use the sequence up to the target as input (at tok_pos)
        target_token_id = self.encoder.encode(FieldToken(self.target_field))

        # create sampler to iterate over batches of rows with same target token position
        sampler = TargetTokenBatchSampler(data, target_token_id, self.max_batch_size)

        # setup the dataloader with batch sampler and target collate fn that only returns
        # tokens up to the target field
        loader = DataLoader(data, batch_sampler=sampler, collate_fn=target_collate_fn(target_token_id), num_workers=0)

        if self.show_progress:
            loader = tqdm(loader)

        completed = []
        for batch in loader:
            completed.extend(c for c in self.autocomplete_batch(batch))

        completed = pad_sequence(completed, batch_first=True, padding_value=pad_token_id)

        # undo sorting operation from batch sampler
        inverse_index = torch.argsort(sampler.sorted_idxs, dim=0, stable=True)
        completed = completed[inverse_index]

        return [detokenize(d) for d in self.encoder.decode(completed)] if decode else completed
