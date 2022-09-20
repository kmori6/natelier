from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
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
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): (B, L1, D)
            k (torch.Tensor): (B, L2, D)
            v (torch.Tensor): (B, L2, D)
            mask (torch.Tensor): (B, L1) or (B, L1, L2)
        """
        q = self.w_q(q).view(q.size(0), -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(k.size(0), -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(v.size(0), -1, self.h, self.d_v).transpose(1, 2)
        matmul = torch.matmul(q, k.transpose(2, 3))  # (B, H, L1, L2)
        scale = matmul / torch.sqrt(torch.tensor(self.d_k))  # (B, H, L1, L2)
        b, h, l1, l2 = scale.size()
        if len(mask.size()) == 2:
            mask = mask.repeat_interleave(h * l1, dim=0).view(b, h, l1, l2)
        else:
            mask = mask.repeat_interleave(h, dim=0).view(b, h, l1, l2)
        scale[mask == 0] = scale.new_tensor(torch.finfo(torch.float32).min)
        softmax = torch.softmax(scale, dim=-1)  # (B, H, L1, L2)
        matmul = torch.matmul(self.dropout(softmax), v)  # (B, H, L1, D_k)
        concat = matmul.transpose(1, 2).contiguous().view(b, l1, self.d_model)
        return self.w_o(concat)  # (B, L1, D)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(x)))


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
