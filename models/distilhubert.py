from typing import List, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from .modules.convolution_speech_embedding import ConvSpeechEmbedding
from .modules.feed_forward import FeedForward as BaseFeedForward
from .modules.multi_head_attention import MultiHeadAttention
from transformers import HubertModel as PretrainedModel

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


class FeedForward(BaseFeedForward):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__(d_model, d_ff)
        self.w_1_dropout = nn.Dropout(dropout_rate)
        self.w_2_dropout = nn.Dropout(dropout_rate)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        hs = self.w_1(hs)
        hs = self.activation(hs)
        hs = self.w_1_dropout(hs)
        hs = self.w_2(hs)
        hs = self.w_2_dropout(hs)
        return hs


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, num_attention_heads: int, dropout_rate: float
    ):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_attention_heads, dropout_rate)
        self.ff = FeedForward(d_model, d_ff, dropout_rate)
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
        hs = self.ff(hs)
        hs = self.ff_norm(shortcut + hs)
        return hs, masks


class DistilhubertModel(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        proj_dim: int = 512,
        d_ff: int = 3072,
        num_attention_heads: int = 12,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        kernel_sizes: List[int] = [10, 3, 3, 3, 3, 2, 2],
        strides: List[int] = [5, 2, 2, 2, 2, 2, 2],
        pos_embed_kernel_size: int = 128,
        pos_embed_groups: int = 16,
    ):
        super().__init__()
        self.embedding = ConvSpeechEmbedding(
            proj_dim=proj_dim,
            d_model=d_model,
            dropout_rate=dropout_rate,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pos_kernel_size=pos_embed_kernel_size,
            pos_conv_groups=pos_embed_groups,
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, d_ff, num_attention_heads, dropout_rate)
                for _ in range(num_layers)
            ]
        )

    def freeze_convolution(self):
        for p in self.embedding.speech_embedding.parameters():
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

    @classmethod
    def from_pretrained(cls):
        pretrained_model = PretrainedModel.from_pretrained("ntu-spml/distilhubert")
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
                tgt_key = f"embedding.speech_embedding.{i}.{m}.{w}"
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
        model = cls()
        model.load_state_dict(tgt_dict)
        return model
