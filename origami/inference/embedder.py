from typing import Optional

import torch
from torch.utils.data.dataloader import DataLoader

from origami.model import ORIGAMI
from origami.preprocessing import DFDataset, StreamEncoder
from origami.utils import FieldToken, Symbol


class Embedder:
    def __init__(
        self, model: ORIGAMI, encoder: StreamEncoder, target_field: Optional[str] = None, batch_size: int = 128
    ):
        self.model = model
        self.encoder = encoder
        self.batch_size = batch_size
        self.target_field = target_field

    @torch.no_grad()
    def embed(
        self,
        dataset: DFDataset,
        position: str = "last",
        reduction: str = "index",
    ) -> torch.Tensor:
        """
        Create embeddings for the given dataset, by passing the data through the hidden layers
        of the model and extract activations at the final hidden layer.

        Args:
            dataset (MongoDBDataset): The dataset to perform prediction on.

            position (str): One of `target`, `last`, `end`
                - target: takes the embedding of the target field token (before its value). For
                          best results, ensure that the target field is the last field in the dataset
                - last: takes the embedding of the last token in the sequence (usually a PAD token).
                        Note: if the embeddings are for prediction and the target field is included
                        in the dataset, this will leak label information into the embeddings.
                - end: takes the embedding of the `END` token (position may vary from document
                       to document). Note: if the embeddings are for prediction and the target
                       field is included in the dataset, this will leak label information into
                       the embeddings.

            reduction (str): One of `index`, `sum`, `mean`
                - index: returns the activation only at the index specified by `position`
                - sum: returns the sum of activations across the sequence dimension up to `position`
                - mean: returns the mean of activations across the sequence dimension up to `position`

            target_field (str): The field to be used as the target for position="target".

        Returns:
            torch.Tensor: The embeddings for each document, in the shape of (n_docs, n_embd)
        """

        assert position in [
            "target",
            "last",
            "end",
        ], "position must be 'target', 'last', or 'end'"

        assert reduction in [
            "index",
            "sum",
            "mean",
        ], "reduction must be 'index', 'sum', or 'mean'"

        # switch model into inference mode
        self.model.eval()

        # initialize the embedding tensor
        embeddings = torch.zeros(
            (0, self.model.model_config.n_embd),
            dtype=torch.float,
            device=self.model.device,
        )

        # setup the dataloader
        test_loader = DataLoader(
            dataset,
            shuffle=False,
            collate_fn=self.model.collate_fn,
            batch_size=self.batch_size,
        )

        if position == "end":
            end_token_id = self.encoder.encode(Symbol.END)
        if position == "target":
            assert self.target_field is not None, "target_field required for position='target'"
            target_token_id = self.encoder.encode(FieldToken(self.target_field))

        # embed each batch and fill embeddings tensor with activations depending on position
        for i, batch in enumerate(test_loader):
            inputs = batch[0]

            # find indices of the token of interest
            match position:
                case "target":
                    indices = torch.where(inputs == target_token_id)[1]
                case "last":
                    indices = torch.full((inputs.size(0),), inputs.size(1) - 1, dtype=torch.long)
                case "end":
                    indices = torch.where(inputs == end_token_id)[1]

            match reduction:
                case "index":
                    emb_batch = self.model.hidden(inputs)[torch.arange(inputs.size(0)), indices, :]
                case "sum":
                    emb_batch = torch.cumsum(self.model.hidden(inputs), dim=1)[torch.arange(inputs.size(0)), indices, :]
                case "mean":
                    emb_batch = torch.cumsum(self.model.hidden(inputs), dim=1)[
                        torch.arange(inputs.size(0)), indices, :
                    ] / (indices.unsqueeze(1) + 1)

            embeddings = torch.cat((embeddings, emb_batch), dim=0)

        # switch model back into training mode
        self.model.train()
        return embeddings
