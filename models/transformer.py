from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, dropout_rate: float, embedding: nn.Module):
        super().__init__()
        self.embedding = embedding
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
            masks = masks.repeat_interleave(self.h * l1, dim=0).view(b, self.h, l1, l2)
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
        ff_activation: nn.Module,
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
        ff_activation: nn.Module,
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
        embedding: nn.Module,
    ):
        super().__init__()
        self.embedding = Embedding(dropout_rate=dropout_rate, embedding=embedding)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_attention_heads=num_attention_heads,
                    dropout_rate=dropout_rate,
                    ff_activation=ff_activation,
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
        embedding: nn.Module,
    ):
        super().__init__()
        self.embedding = Embedding(dropout_rate=dropout_rate, embedding=embedding)
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
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)
        self.classifier.weight = embedding.weight

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


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_ff: int = 2048,
        num_attention_heads: int = 8,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 1,
        dropout_rate: float = 0.1,
        padding_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.padding_id = padding_id
        embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_id)
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_encoder_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            ff_activation=nn.ReLU(),
            embedding=embedding,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_decoder_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            ff_activation=nn.ReLU(),
            embedding=embedding,
        )

    def initialize_embeddings(self, vocab_size: int):
        embedding = nn.Embedding(vocab_size, self.d_model, self.padding_id)
        self.encoder.embedding.embedding = embedding
        self.decoder.embedding.embedding = embedding
        self.decoder.classifier = nn.Linear(self.d_model, vocab_size, bias=False)
        self.decoder.classifier.weight = embedding.weight

    def freeze_embeddings(self):
        for p in self.encoder.embedding.embedding.parameters():
            p.requires_grad = False
        for p in self.decoder.embedding.embedding.parameters():
            p.requires_grad = False
        for p in self.decoder.classifier.parameters():
            p.requires_grad = False

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze_decoder(self):
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        encoder_tokens: torch.Tensor,
        encoder_masks: torch.Tensor,
        decoder_tokens: torch.Tensor,
        decoder_masks: torch.Tensor,
    ):
        encoder_hs, encoder_masks = self.encoder(encoder_tokens, encoder_masks)
        logits = self.decoder(decoder_tokens, decoder_masks, encoder_hs, encoder_masks)
        return logits

    def decode(
        self, tokens: torch.Tensor, beam_size: int = 5, max_length: int = None
    ) -> list[Dict[str, Union[float, str]]]:
        assert (
            len(tokens.size()) == 2 and tokens.size(0) == 1
        ), f"batch size should be 1 (tokens shape: {tokens.size()})"
        max_length = max_length if max_length else tokens.size(1) + 50
        # encoder forward
        encoder_hs, encoder_masks = self.encoder(tokens, torch.ones_like(tokens))
        encoder_hs = encoder_hs.repeat_interleave(beam_size, dim=0)
        encoder_masks = encoder_masks.repeat_interleave(beam_size, dim=0)
        # initial stats
        stats = [
            {"score": 0.0 if i == 0 else -1e8, "tokens": [self.bos_id]}
            for i in range(beam_size)
        ]
        results = []
        # beam search
        for i in range(1, max_length + 1):
            # decoder forward
            decoder_tokens = tokens.new_tensor([sample["tokens"] for sample in stats])
            decoder_masks = tokens.new_ones(len(stats), i, i).tril()
            logits = self.decoder(
                decoder_tokens,
                decoder_masks,
                encoder_hs[: len(stats), :, :],
                encoder_masks[: len(stats), :],
            )
            scores = tokens.new_tensor(
                [[sample["score"]] for sample in stats]
            ) + torch.log_softmax(logits[:, -1, :], dim=-1)
            beam_scores, beam_tokens = torch.topk(scores.flatten(), len(stats))
            stats = [
                {
                    "score": beam_score.item(),
                    "tokens": stats[beam_token.item() // self.vocab_size]["tokens"]
                    + [beam_token.item() % self.vocab_size],
                }
                for beam_score, beam_token in zip(beam_scores, beam_tokens)
            ]
            # add eos_token_id
            if i == max_length:
                for sample in stats:
                    sample["tokens"].append(self.eos_id)
            # sort stats
            keep_stats = []
            for sample in stats:
                if sample["tokens"][-1] == self.eos_id:
                    results.append(sample)
                else:
                    keep_stats.append(sample)
            stats = keep_stats
            # stop search
            if len(stats) < 1 or len(results) >= beam_size:
                break
        return sorted(results, key=lambda x: x["score"], reverse=True)
