from typing import Tuple
import torch
import torch.nn as nn
from .modules.absolute_positional_embedding import AbsolutePositionalEmbedding
from .modules.multi_head_attention import MultiHeadAttention
from .modules.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, num_attention_heads: int, dropout_rate: float
    ):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_attention_heads, dropout_rate)
        self.mha = MultiHeadAttention(d_model, num_attention_heads, dropout_rate)
        self.ff = FeedForward(d_model, d_ff)
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


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = AbsolutePositionalEmbedding(dropout_rate)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, d_ff, num_attention_heads, dropout_rate)
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
        hs = self.position_embedding(hs)
        for layer in self.layers:
            hs, decoder_masks, encoder_hs, encoder_masks = layer(
                hs, decoder_masks, encoder_hs, encoder_masks
            )
        hs = self.classifier(hs)
        return hs
