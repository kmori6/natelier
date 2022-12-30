from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import HubertModel

from .transformer import Embedding, Encoder, EncoderLayer, FeedForward

KEY_DICT = {
    "feature_projection": {
        "feature_projection.projection.weight": "embedding.projection.linear.weight",
        "feature_projection.projection.bias": "embedding.projection.linear.bias",
    },
    "encoder": {
        "pos_conv_embed.conv.bias": "position_embedding.conv.bias",
        "pos_conv_embed.conv.weight_g": "position_embedding.conv.weight_g",
        "pos_conv_embed.conv.weight_v": "position_embedding.conv.weight_v",
        "attention.q_proj.weight": "mha.w_q.weight",
        "attention.q_proj.bias": "mha.w_q.bias",
        "attention.k_proj.weight": "mha.w_k.weight",
        "attention.k_proj.bias": "mha.w_k.bias",
        "attention.v_proj.weight": "mha.w_v.weight",
        "attention.v_proj.bias": "mha.w_v.bias",
        "attention.out_proj.weight": "mha.w_o.weight",
        "attention.out_proj.bias": "mha.w_o.bias",
        "feed_forward.intermediate_dense.weight": "ff.w_1.weight",
        "feed_forward.intermediate_dense.bias": "ff.w_1.bias",
        "feed_forward.output_dense.weight": "ff.w_2.weight",
        "feed_forward.output_dense.bias": "ff.w_2.bias",
        "layer_norm.weight": "mha_norm.weight",
        "layer_norm.bias": "mha_norm.bias",
        "final_layer_norm.weight": "ff_norm.weight",
        "final_layer_norm.bias": "ff_norm.bias",
    },
}


class TemporalConvolution(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        group_norm: bool,
    ):
        super().__init__()
        self.group_norm = group_norm
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        if group_norm:
            self.norm = nn.GroupNorm(output_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        hs = self.conv(hs)
        if self.group_norm:
            hs = self.norm(hs)
        hs = self.activation(hs)
        return hs


class LinearProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        hs = self.linear(hs)
        hs = self.dropout(hs)
        return hs


class PositionEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, groups: int):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=groups,
            ),
            dim=2,
        )
        self.activation = nn.GELU()

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        hs = self.conv(hs)[:, :, :-1]  # (B, D, L)
        hs = self.activation(hs)
        return hs


class ConvSpeechEmbedding(Embedding):
    def __init__(
        self,
        d_proj: int,
        d_model: int,
        dropout_rate: float,
        kernel_sizes: List[int],
        strides: List[int],
        pos_kernel_size: int,
        pos_conv_groups: int,
    ):
        super().__init__(dropout_rate=dropout_rate, embedding=None)
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.embedding = nn.ModuleList(
            [
                TemporalConvolution(
                    input_dim=1 if i == 0 else d_proj,
                    output_dim=d_proj,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    group_norm=True if i == 0 else False,
                )
                for i in range(len(kernel_sizes))
            ]
        )
        self.projection = LinearProjection(d_proj, d_model, dropout_rate)
        self.position_embedding = PositionEmbedding(
            d_model, d_model, pos_kernel_size, pos_conv_groups
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, waves: torch.Tensor, wave_lengths: torch.Tensor) -> torch.Tensor:
        hs = self.embed_speech(waves)
        hs = self.embed_position(hs)
        hs = self.dropout(self.norm(hs))

        for kernel_size, stride in zip(self.kernel_sizes, self.strides):
            wave_lengths = (
                torch.div(wave_lengths - kernel_size, stride, rounding_mode="floor") + 1
            )

        return hs, wave_lengths

    def embed_speech(self, waves: torch.Tensor) -> torch.Tensor:
        hs = waves.unsqueeze(1)  # (B, 1, L)
        for module in self.embedding:
            hs = module(hs)  # (B, P, L)
        hs = self.projection(hs.transpose(1, 2))  # (B, L, D)
        return hs

    def embed_position(self, hs: torch.Tensor) -> torch.Tensor:
        pos = self.position_embedding(hs.transpose(1, 2)).transpose(1, 2)
        return hs + pos


class HubertFeedForward(FeedForward):
    def __init__(
        self, d_model: int, d_ff: int, dropout_rate: float, activation: nn.Module
    ):
        super().__init__(d_model=d_model, d_ff=d_ff, activation=activation)
        self.w_1_dropout = nn.Dropout(dropout_rate)
        self.w_2_dropout = nn.Dropout(dropout_rate)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        hs = self.w_1(hs)
        hs = self.activation(hs)
        hs = self.w_1_dropout(hs)
        hs = self.w_2(hs)
        hs = self.w_2_dropout(hs)
        return hs


class HubertEncoderLayer(EncoderLayer):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        dropout_rate: float,
        ff_activation: nn.Module,
    ):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            ff_activation=ff_activation,
        )
        self.ff = HubertFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            activation=ff_activation,
        )


class DistilHubert(Encoder):
    def __init__(
        self,
        d_model: int = 768,
        d_proj: int = 512,
        d_ff: int = 3072,
        num_attention_heads: int = 12,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        kernel_sizes: List[int] = [10, 3, 3, 3, 3, 2, 2],
        strides: List[int] = [5, 2, 2, 2, 2, 2, 2],
        pos_embed_kernel_size: int = 128,
        pos_embed_groups: int = 16,
        ff_activation: nn.Module = nn.GELU(),
        load_pretrained_weight: bool = False,
    ):
        super().__init__(
            vocab_size=None,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=None,
            ff_activation=ff_activation,
            embedding=None,
        )
        self.embedding = ConvSpeechEmbedding(
            d_proj=d_proj,
            d_model=d_model,
            dropout_rate=dropout_rate,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pos_kernel_size=pos_embed_kernel_size,
            pos_conv_groups=pos_embed_groups,
        )
        self.layers = nn.ModuleList(
            [
                HubertEncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_attention_heads=num_attention_heads,
                    dropout_rate=dropout_rate,
                    ff_activation=ff_activation,
                )
                for _ in range(num_layers)
            ]
        )
        if load_pretrained_weight:
            self.load_pretrained_weight()

    def freeze_convolution(self):
        for p in self.embedding.embedding.parameters():
            p.requires_grad = False

    def forward(self, waves: torch.Tensor, wave_lengths: torch.Tensor) -> torch.Tensor:
        hs, hs_lens = self.embedding(waves, wave_lengths)
        masks = pad_sequence(
            [torch.ones(length) for length in hs_lens],
            batch_first=True,
            padding_value=0,
        )
        for layer in self.layers:
            hs, masks = layer(hs, masks)
        return hs, hs_lens, masks

    def load_pretrained_weight(self):
        pretrained_model = HubertModel.from_pretrained("ntu-spml/distilhubert")
        state_dict = pretrained_model.state_dict()
        del state_dict["masked_spec_embed"]
        tgt_dict = {}
        for k in tqdm(state_dict.keys()):
            modules = k.split(".")
            main_module = modules[0]
            if main_module == "feature_extractor":
                i, m, w = modules[-3:]
                if m == "layer_norm":
                    m = "norm"
                tgt_key = f"embedding.embedding.{i}.{m}.{w}"
            elif main_module == "feature_projection":
                tgt_key = f"{KEY_DICT[main_module][k]}"
            else:
                sub_module = modules[1]
                if sub_module == "pos_conv_embed":
                    tgt_key = (
                        f"embedding.{KEY_DICT[main_module]['.'.join(modules[1:])]}"
                    )
                elif sub_module == "layer_norm":
                    w = modules[-1]
                    tgt_key = f"embedding.norm.{w}"
                else:
                    i = modules[2]
                    tgt_key = (
                        f"layers.{i}.{KEY_DICT[main_module]['.'.join(modules[3:])]}"
                    )
            tgt_dict[tgt_key] = state_dict[k]
        self.load_state_dict(tgt_dict)
