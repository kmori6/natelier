from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from .modules.factorized_embedding import FactorizedEmbedding
from .modules.multi_head_attention import MultiHeadAttention
from .modules.feed_forward import FeedForward
from transformers import AlbertModel as PretrainedModel

KEY_DICT = {
    "embeddings": {
        "word_embeddings.weight": "embedding.token_embedding.weight",
        "position_embeddings.weight": "embedding.position_embedding.weight",
        "token_type_embeddings.weight": "embedding.segment_embedding.weight",
        "LayerNorm.weight": "embedding.embedding_norm.weight",
        "LayerNorm.bias": "embedding.embedding_norm.bias",
    },
    "encoder": {
        "embedding_hidden_mapping_in.weight": "embedding.embedding_projection.weight",
        "embedding_hidden_mapping_in.bias": "embedding.embedding_projection.bias",
        "attention.query.weight": "mha.w_q.weight",
        "attention.query.bias": "mha.w_q.bias",
        "attention.key.weight": "mha.w_k.weight",
        "attention.key.bias": "mha.w_k.bias",
        "attention.value.weight": "mha.w_v.weight",
        "attention.value.bias": "mha.w_v.bias",
        "attention.dense.weight": "mha.w_o.weight",
        "attention.dense.bias": "mha.w_o.bias",
        "ffn.weight": "ff.w_1.weight",
        "ffn.bias": "ff.w_1.bias",
        "ffn_output.weight": "ff.w_2.weight",
        "ffn_output.bias": "ff.w_2.bias",
        "attention.LayerNorm.weight": "mha_norm.weight",
        "attention.LayerNorm.bias": "mha_norm.bias",
        "full_layer_layer_norm.weight": "ff_norm.weight",
        "full_layer_layer_norm.bias": "ff_norm.bias",
    },
}


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, num_attention_heads: int, dropout_rate: float
    ):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_attention_heads, dropout_rate)
        self.ff = FeedForward(d_model, d_ff)
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


class AlbertModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30000,
        position_size: int = 512,
        embedding_size: int = 128,
        d_model: int = 768,
        d_ff: int = 3072,
        num_attention_heads: int = 12,
        num_layers: int = 12,
        dropout_rate: float = 0.0,
        padding_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding = FactorizedEmbedding(
            vocab_size, position_size, embedding_size, d_model, dropout_rate, padding_id
        )
        self.shared_layer = EncoderLayer(
            d_model, d_ff, num_attention_heads, dropout_rate
        )

    def forward(
        self, tokens: torch.Tensor, masks: torch.Tensor, segments: torch.Tensor
    ) -> torch.Tensor:
        hs = self.embedding(tokens, segments)
        for _ in range(self.num_layers):
            hs, masks = self.shared_layer(hs, masks)
        return hs

    @classmethod
    def from_pretrained(cls):
        pretrained_model = PretrainedModel.from_pretrained(
            "albert-base-v2", add_pooling_layer=False
        )
        state_dict = pretrained_model.state_dict()
        del state_dict["embeddings.position_ids"]
        tgt_dict = {}
        for k in tqdm(state_dict.keys()):
            module, sub_module = k.split(".")[:2]
            if module == "encoder" and sub_module == "albert_layer_groups":
                sub_module = k.split(".", 5)[-1]
                tgt_key = f"shared_layer.{KEY_DICT[module][sub_module]}"
            else:
                sub_module = k.split(".", 1)[-1]
                tgt_key = KEY_DICT[module][sub_module]
            tgt_dict[tgt_key] = state_dict[k]
        model = cls()
        model.load_state_dict(tgt_dict)
        return model
