import torch
import torch.nn as nn


class FactorizedEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        position_size: int,
        embedding_size: int,
        d_model: int,
        dropout_rate: float,
        padding_id: int,
    ):
        super().__init__()
        self.padding_id = padding_id
        self.token_embedding = nn.Embedding(vocab_size, embedding_size, padding_id)
        self.segment_embedding = nn.Embedding(2, embedding_size)
        self.position_embedding = nn.Embedding(position_size, embedding_size)
        self.embedding_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding_projection = nn.Linear(embedding_size, d_model)

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        token_embedding = self.embed_tokens(tokens)
        segment_embedding = self.embed_segments(segments)
        position_embedding = self.embed_positions(tokens)
        hs = token_embedding + segment_embedding + position_embedding
        hs = self.embedding_norm(hs)
        hs = self.dropout(hs)
        return self.embedding_projection(hs)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens)

    def embed_segments(self, segments: torch.Tensor) -> torch.Tensor:
        return self.segment_embedding(segments)

    def embed_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        positions = torch.arange(length, dtype=torch.long, device=tokens.device)
        return self.position_embedding(positions.unsqueeze(0))  # (1, L)
