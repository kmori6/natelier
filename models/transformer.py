from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, padding_id: int, dropout_rate: float
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_id)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hs = self.embedding(tokens)
        pe = self.positional_encoding(hs)
        return self.dropout(hs + pe)

    def positional_encoding(self, hs: torch.Tensor) -> torch.Tensor:
        l, d = hs.size()[1:]
        pe = hs.new_zeros(l, d)  # (L, D)
        pos = torch.arange(0, l).unsqueeze(1)  # (1, L)
        pe[:, 0::2] = torch.sin(pos / torch.pow(10000, torch.arange(0, d, 2) / d))
        pe[:, 1::2] = torch.cos(pos / torch.pow(10000, torch.arange(0, d, 2) / d))
        return pe.unsqueeze(0)  # (1, L, D)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_attention_heads: int, dropout_rate: float):
        super().__init__()
        self.h = num_attention_heads
        self.d_model = d_model
        self.d_k = self.d_v = int(d_model / num_attention_heads)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): query (B, L1, D)
            k (torch.Tensor): key (B, L2, D)
            v (torch.Tensor): value (B, L2, D)
            masks (torch.Tensor): encoder masks (B, L1) or decoder masks (B, L1, L2)
        """
        (b, l1, d), l2 = q.size(), k.size(1)
        q = self.w_q(q).view(b, l1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(b, l2, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(b, l2, self.h, self.d_v).transpose(1, 2)
        matrices = q @ k.transpose(2, 3) / np.sqrt(self.d_k)  # (B, H, L1, L2)
        if len(masks.size()) == 2:
            masks = masks.repeat_interleave(self.h * l2, dim=0).view(b, self.h, l1, l2)
        else:
            masks = masks.repeat_interleave(self.h, dim=0).view(b, self.h, l1, l2)
        matrices[masks == 0] = matrices.new_tensor(torch.finfo(torch.float32).min)
        weights = torch.softmax(matrices, dim=-1)  # (B, H, L1, L2)
        matrices = self.dropout(weights) @ v  # (B, H, L1, D_k)
        concat = matrices.transpose(1, 2).contiguous().view(b, l1, d)
        return self.w_o(concat)  # (B, L1, D)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: nn.Module):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w_1(x)
        x = self.activation(x)
        return self.w_2(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        dropout_rate: float,
        ff_activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
        )
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, activation=ff_activation)
        self.mha_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, hs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # multi-head attention
        shortcut = hs
        hs = self.dropout(self.mha(hs, hs, hs, masks))
        hs = self.mha_norm(shortcut + hs)
        # feed-forward
        shortcut = hs
        hs = self.dropout(self.ff(hs))
        hs = self.ff_norm(shortcut + hs)
        return hs, masks


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        dropout_rate: float,
        ff_activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.masked_mha = MultiHeadAttention(
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
        )
        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
        )
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, activation=ff_activation)
        self.masked_mha_norm = nn.LayerNorm(d_model)
        self.mha_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        decoder_hs: torch.Tensor,
        decoder_masks: torch.Tensor,
        encoder_hs: torch.Tensor,
        encoder_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # masked multi-head attention
        shortcut = decoder_hs
        decoder_hs = self.dropout(
            self.masked_mha(decoder_hs, decoder_hs, decoder_hs, decoder_masks)
        )
        decoder_hs = self.masked_mha_norm(shortcut + decoder_hs)
        # multi-head attention
        shortcut = decoder_hs
        decoder_hs = self.dropout(
            self.mha(decoder_hs, encoder_hs, encoder_hs, encoder_masks)
        )
        decoder_hs = self.mha_norm(shortcut + decoder_hs)
        # feed-forward
        shortcut = decoder_hs
        decoder_hs = self.dropout(self.ff(decoder_hs))
        decoder_hs = self.ff_norm(shortcut + decoder_hs)
        return decoder_hs, decoder_masks, encoder_hs, encoder_masks


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        num_layers: int,
        dropout_rate: float,
        padding_id: int,
        ff_activation: nn.Module,
    ):
        super().__init__()
        self.embedding = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_id=padding_id,
            dropout_rate=dropout_rate,
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_attention_heads=num_attention_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, tokens: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hs = self.embedding(tokens)
        for layer in self.layers:
            hs, masks = layer(hs, masks)
        return hs, masks


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        num_layers: int,
        dropout_rate: float,
        padding_id: int,
        ff_activation: nn.Module,
    ):
        super().__init__()
        self.embedding = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_id=padding_id,
            dropout_rate=dropout_rate,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_attention_heads=num_attention_heads,
                    dropout_rate=dropout_rate,
                    ff_activation=ff_activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        decoder_tokens: torch.Tensor,
        decoder_masks: torch.Tensor,
        encoder_hs: torch.Tensor,
        encoder_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hs = self.embedding(decoder_tokens)
        for layer in self.layers:
            hs, decoder_masks, encoder_hs, encoder_masks = layer(
                hs, decoder_masks, encoder_hs, encoder_masks
            )
        hs = self.classifier(hs)
        return hs
