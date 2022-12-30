from tqdm import tqdm
import torch
import torch.nn as nn
from .transformer import Embedding, EncoderLayer, Encoder
from transformers import AlbertModel as PretrainedModel

KEY_DICT = {
    "embeddings": {
        "word_embeddings.weight": "embedding.embedding.weight",
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


class FactorizedEmbedding(Embedding):
    def __init__(
        self,
        vocab_size: int,
        position_size: int,
        embedding_size: int,
        d_model: int,
        dropout_rate: float,
        padding_id: int,
    ):
        super().__init__(dropout_rate=dropout_rate, embedding=nn.Embedding)
        self.padding_id = padding_id
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_id)
        self.segment_embedding = nn.Embedding(2, embedding_size)
        self.position_embedding = nn.Embedding(position_size, embedding_size)
        self.embedding_norm = nn.LayerNorm(embedding_size)
        self.embedding_projection = nn.Linear(embedding_size, d_model)

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        token_embedding = self.embedding(tokens)
        segment_embedding = self.segment_embedding(segments)

        length = tokens.size(1)
        positions = torch.arange(length, dtype=torch.long, device=tokens.device)
        position_embedding = self.position_embedding(positions.unsqueeze(0))  # (1, L)

        hs = token_embedding + segment_embedding + position_embedding
        hs = self.embedding_norm(hs)
        hs = self.dropout(hs)
        return self.embedding_projection(hs)


class Albert(Encoder):
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
        ff_activation: nn.Module = nn.GELU(),
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            ff_activation=ff_activation,
            embedding=nn.Embedding,
        )
        self.d_model = d_model
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding = FactorizedEmbedding(
            vocab_size=vocab_size,
            position_size=position_size,
            embedding_size=embedding_size,
            d_model=d_model,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
        )
        self.layers = EncoderLayer(
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            ff_activation=ff_activation,
        )

    def forward(
        self, tokens: torch.Tensor, masks: torch.Tensor, segments: torch.Tensor
    ) -> torch.Tensor:
        hs = self.embedding(tokens, segments)
        for _ in range(self.num_layers):
            hs, masks = self.layers(hs, masks)
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
                tgt_key = f"layers.{KEY_DICT[module][sub_module]}"
            else:
                sub_module = k.split(".", 1)[-1]
                tgt_key = KEY_DICT[module][sub_module]
            tgt_dict[tgt_key] = state_dict[k]
        model = cls()
        model.load_state_dict(tgt_dict)
        return model
