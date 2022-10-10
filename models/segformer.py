from typing import List
from tqdm import tqdm
import torch
import torch.nn as nn
from .modules.overlapped_patch_embedding import OverlappedPatchMerging
from .modules.multi_head_attention import MultiHeadAttention
from .modules.feed_forward import FeedForward
from transformers import SegformerModel as PretrainedModel

KEY_DICT = {
    "patch_embeddings": {
        "proj.weight": "feats_embedding.weight",
        "proj.bias": "feats_embedding.bias",
        "layer_norm.weight": "norm.weight",
        "layer_norm.bias": "norm.bias",
    },
    "block": {
        "layer_norm_1.weight": "esa_norm.weight",
        "layer_norm_1.bias": "esa_norm.bias",
        "attention.self.query.weight": "esa.w_q.weight",
        "attention.self.query.bias": "esa.w_q.bias",
        "attention.self.key.weight": "esa.w_k.weight",
        "attention.self.key.bias": "esa.w_k.bias",
        "attention.self.value.weight": "esa.w_v.weight",
        "attention.self.value.bias": "esa.w_v.bias",
        "attention.self.sr.weight": "esa.w_r.weight",
        "attention.self.sr.bias": "esa.w_r.bias",
        "attention.self.layer_norm.weight": "esa.norm.weight",
        "attention.self.layer_norm.bias": "esa.norm.bias",
        "attention.output.dense.weight": "esa.w_o.weight",
        "attention.output.dense.bias": "esa.w_o.bias",
        "layer_norm_2.weight": "mf_norm.weight",
        "layer_norm_2.bias": "mf_norm.bias",
        "mlp.dense1.weight": "mf.w_1.weight",
        "mlp.dense1.bias": "mf.w_1.bias",
        "mlp.dwconv.dwconv.weight": "mf.conv.weight",
        "mlp.dwconv.dwconv.bias": "mf.conv.bias",
        "mlp.dense2.weight": "mf.w_2.weight",
        "mlp.dense2.bias": "mf.w_2.bias",
    },
}


class EfficientSelfAttention(MultiHeadAttention):
    def __init__(
        self,
        d_model: int,
        num_attention_heads: int,
        reduction_ratio: int,
        dropout_rate: float,
    ):
        super().__init__(d_model, num_attention_heads, dropout_rate)
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio > 1:
            self.w_r = nn.Conv2d(
                d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio
            )
            self.norm = nn.LayerNorm(d_model)

    def forward(self, hs: torch.Tensor, hight: int, width: int) -> torch.Tensor:
        b, _, c = hs.size()  # (B, L, C)
        q = self.w_q(hs).view(b, -1, self.h, self.d_k).transpose(1, 2)
        if self.reduction_ratio > 1:
            hs = hs.transpose(1, 2).view(b, c, hight, width)
            hs = self.w_r(hs)  # (B, C, H', W')
            hs = hs.view(b, c, -1).transpose(1, 2)  # (B, L', C)
            hs = self.norm(hs)
        k = self.w_k(hs).view(b, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(hs).view(b, -1, self.h, self.d_v).transpose(1, 2)
        matmul = torch.matmul(q, k.transpose(2, 3))  # (B, C, L, L or L')
        scale = matmul / torch.sqrt(torch.tensor(self.d_k))
        softmax = torch.softmax(scale, dim=-1)
        matmul = torch.matmul(self.dropout(softmax), v)  # (B, C, L, D_k)
        concat = matmul.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.w_o(concat)  # (B, L, C)


class MixFfn(FeedForward):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_model, d_ff)
        self.conv = nn.Conv2d(
            d_ff, d_ff, kernel_size=3, stride=1, padding=1, groups=d_ff
        )

    def forward(self, hs: torch.Tensor, hight: int, width: int) -> torch.Tensor:
        hs = self.w_1(hs)  # (B, L, C)
        b, _, c = hs.size()
        hs = hs.transpose(1, 2).view(b, c, hight, width)
        hs = self.conv(hs)  # (B, C, L)
        hs = hs.view(b, c, -1).transpose(1, 2)  # (B, L, C)
        return self.w_2(self.activation(hs))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        reduction_ratio: float,
        dropout_rate: float,
    ):
        super().__init__()
        self.esa = EfficientSelfAttention(
            d_model, num_attention_heads, reduction_ratio, dropout_rate
        )
        self.mf = MixFfn(d_model, d_ff)
        self.esa_norm = nn.LayerNorm(d_model)
        self.mf_norm = nn.LayerNorm(d_model)

    def forward(self, hs: torch.Tensor, hight: int, width: int) -> torch.Tensor:
        shortcut = hs
        hs = self.esa_norm(hs)
        hs = shortcut + self.esa(hs, hight=hight, width=width)
        shortcut = hs
        hs = self.mf_norm(hs)
        hs = shortcut + self.mf(hs, hight=hight, width=width)
        return hs


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        d_ff: int,
        patch_size: int,
        stride: int,
        padding_size: int,
        num_attention_heads: int,
        reduction_ratio: float,
        dropout_rate: float,
    ):
        super().__init__()
        self.embedding = OverlappedPatchMerging(
            in_channels,
            d_model,
            patch_size=patch_size,
            stride=stride,
            padding_size=padding_size,
        )
        self.transformer_block = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    d_ff,
                    num_attention_heads=num_attention_heads,
                    reduction_ratio=reduction_ratio,
                    dropout_rate=dropout_rate,
                )
                for _ in range(2)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hs: torch.Tensor, hight: int, width: int) -> torch.Tensor:
        batch_size = hs.size(0)
        hs = self.embedding(hs, hight, width)
        for layer in self.transformer_block:
            hs = layer(hs, hight, width)
        hs = self.norm(hs)
        return hs.transpose(1, 2).view(batch_size, -1, hight, width)


class SegformerModel(nn.Module):
    def __init__(
        self,
        d_model: List[int] = [32, 64, 160, 256],
        d_ff: List[int] = [128, 256, 640, 1024],
        patch_size: List[int] = [7, 3, 3, 3],
        stride: List[int] = [4, 2, 2, 2],
        padding_size: List[int] = [3, 1, 1, 1],
        num_attention_heads: List[int] = [1, 2, 5, 8],
        reduction_ratio: List[int] = [8, 4, 2, 1],
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.stage0 = EncoderStage(
            3,
            d_model[0],
            d_ff[0],
            patch_size[0],
            stride[0],
            padding_size[0],
            num_attention_heads[0],
            reduction_ratio[0],
            dropout_rate,
        )
        self.stage1 = EncoderStage(
            d_model[0],
            d_model[1],
            d_ff[1],
            patch_size[1],
            stride[1],
            padding_size[1],
            num_attention_heads[1],
            reduction_ratio[1],
            dropout_rate,
        )
        self.stage2 = EncoderStage(
            d_model[1],
            d_model[2],
            d_ff[2],
            patch_size[2],
            stride[2],
            padding_size[2],
            num_attention_heads[2],
            reduction_ratio[2],
            dropout_rate,
        )
        self.stage3 = EncoderStage(
            d_model[2],
            d_model[3],
            d_ff[3],
            patch_size[3],
            stride[3],
            padding_size[3],
            num_attention_heads[3],
            reduction_ratio[3],
            dropout_rate,
        )

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        hte_hs = []
        b, _, h, w = images.size()
        # stage0
        hs = self.stage0(images, hight=h // 4, width=w // 4)
        hte_hs.append(hs)
        # stage1
        hs = self.stage1(hs, hight=h // 8, width=w // 8)
        hte_hs.append(hs)
        # stage2
        hs = self.stage2(hs, hight=h // 16, width=w // 16)
        hte_hs.append(hs)
        # stage3
        hs = self.stage3(hs, hight=h // 32, width=w // 32)
        hte_hs.append(hs)
        return hte_hs

    @classmethod
    def from_pretrained(cls):
        pretrained_model = PretrainedModel.from_pretrained("nvidia/mit-b0")
        state_dict = pretrained_model.state_dict()
        tgt_dict = {}
        for k, v in tqdm(state_dict.items()):
            key_parts = k.split(".")
            stage_id = key_parts[2]
            main_module = key_parts[1]
            if main_module == "patch_embeddings":
                sub_module = ".".join(key_parts[-2:])
                tgt_key = (
                    f"stage{stage_id}.embedding.{KEY_DICT[main_module][sub_module]}"
                )
            elif main_module == "block":
                sub_module = ".".join(key_parts[4:])
                block_id = key_parts[3]
                tgt_key = f"stage{stage_id}.transformer_block.{block_id}.{KEY_DICT[main_module][sub_module]}"
            else:
                weight_type = key_parts[-1]
                tgt_key = f"stage{stage_id}.norm.{weight_type}"
            tgt_dict[tgt_key] = v
        model = cls()
        model.load_state_dict(tgt_dict)
        return model
