from typing import List
import torch
import torch.nn as nn


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


class ConvSpeechEmbedding(nn.Module):
    def __init__(
        self,
        proj_dim: int,
        d_model: int,
        dropout_rate: float,
        kernel_sizes: List[int],
        strides: List[int],
        pos_kernel_size: int,
        pos_conv_groups: int,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.speech_embedding = nn.ModuleList(
            [
                TemporalConvolution(
                    input_dim=1 if i == 0 else proj_dim,
                    output_dim=proj_dim,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    group_norm=True if i == 0 else False,
                )
                for i in range(len(kernel_sizes))
            ]
        )
        self.projection = LinearProjection(proj_dim, d_model, dropout_rate)
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
        for module in self.speech_embedding:
            hs = module(hs)  # (B, P, L)
        hs = self.projection(hs.transpose(1, 2))  # (B, L, D)
        return hs

    def embed_position(self, hs: torch.Tensor) -> torch.Tensor:
        pos = self.position_embedding(hs.transpose(1, 2)).transpose(1, 2)
        return hs + pos
