import torch
import torch.nn as nn


class LearnableEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        position_size: int,
        d_model: int,
        dropout_rate: float,
        padding_id: int,
        token_embedding: nn.Embedding,
    ):
        super().__init__()
        self.padding_id = padding_id
        self.token_embedding = token_embedding
        self.position_embedding = nn.Embedding(position_size, d_model, padding_id)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_embedding = self.embed_tokens(tokens)
        position_embedding = self.embed_positions(tokens)
        hs = self.embedding_norm(token_embedding + position_embedding)
        hs = self.dropout(hs)
        return hs

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens)

    def embed_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, length = tokens.size()
        padding_mask = tokens != self.padding_id
        # tokens: [[11, 12, 1, 1], [11, 12, 13, 14]] (padding_id = 1)
        # positions: [[2, 3, 1, 1], [2, 3, 4, 5]]
        positions = torch.arange(
            self.padding_id,
            length + self.padding_id,
            dtype=torch.long,
            device=tokens.device,
        ).repeat(batch_size, 1)
        positions[padding_mask] = self.padding_id
        return self.position_embedding(positions)
