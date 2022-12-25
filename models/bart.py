import torch
import torch.nn as nn
from tqdm import tqdm

from .transformer import Embedding, Encoder, Decoder, TransformerModel

KEY_DICT = {
    "encoder": {
        "embed_tokens.weight": "embedding.token_embedding.weight",
        "embed_positions.weight": "embedding.position_embedding.weight",
        "layernorm_embedding.weight": "embedding.embedding_norm.weight",
        "layernorm_embedding.bias": "embedding.embedding_norm.bias",
        "layers": {
            "self_attn.q_proj.weight": "mha.w_q.weight",
            "self_attn.q_proj.bias": "mha.w_q.bias",
            "self_attn.k_proj.weight": "mha.w_k.weight",
            "self_attn.k_proj.bias": "mha.w_k.bias",
            "self_attn.v_proj.weight": "mha.w_v.weight",
            "self_attn.v_proj.bias": "mha.w_v.bias",
            "self_attn.out_proj.weight": "mha.w_o.weight",
            "self_attn.out_proj.bias": "mha.w_o.bias",
            "fc1.weight": "ff.w_1.weight",
            "fc1.bias": "ff.w_1.bias",
            "fc2.weight": "ff.w_2.weight",
            "fc2.bias": "ff.w_2.bias",
            "self_attn_layer_norm.weight": "mha_norm.weight",
            "self_attn_layer_norm.bias": "mha_norm.bias",
            "final_layer_norm.weight": "ff_norm.weight",
            "final_layer_norm.bias": "ff_norm.bias",
        },
    },
    "decoder": {
        "embed_tokens.weight": "embedding.token_embedding.weight",
        "embed_positions.weight": "embedding.position_embedding.weight",
        "layernorm_embedding.weight": "embedding.embedding_norm.weight",
        "layernorm_embedding.bias": "embedding.embedding_norm.bias",
        "layers": {
            "self_attn.q_proj.weight": "masked_mha.w_q.weight",
            "self_attn.q_proj.bias": "masked_mha.w_q.bias",
            "self_attn.k_proj.weight": "masked_mha.w_k.weight",
            "self_attn.k_proj.bias": "masked_mha.w_k.bias",
            "self_attn.v_proj.weight": "masked_mha.w_v.weight",
            "self_attn.v_proj.bias": "masked_mha.w_v.bias",
            "self_attn.out_proj.weight": "masked_mha.w_o.weight",
            "self_attn.out_proj.bias": "masked_mha.w_o.bias",
            "encoder_attn.q_proj.weight": "mha.w_q.weight",
            "encoder_attn.q_proj.bias": "mha.w_q.bias",
            "encoder_attn.k_proj.weight": "mha.w_k.weight",
            "encoder_attn.k_proj.bias": "mha.w_k.bias",
            "encoder_attn.v_proj.weight": "mha.w_v.weight",
            "encoder_attn.v_proj.bias": "mha.w_v.bias",
            "encoder_attn.out_proj.weight": "mha.w_o.weight",
            "encoder_attn.out_proj.bias": "mha.w_o.bias",
            "fc1.weight": "ff.w_1.weight",
            "fc1.bias": "ff.w_1.bias",
            "fc2.weight": "ff.w_2.weight",
            "fc2.bias": "ff.w_2.bias",
            "self_attn_layer_norm.weight": "masked_mha_norm.weight",
            "self_attn_layer_norm.bias": "masked_mha_norm.bias",
            "encoder_attn_layer_norm.weight": "mha_norm.weight",
            "encoder_attn_layer_norm.bias": "mha_norm.bias",
            "final_layer_norm.weight": "ff_norm.weight",
            "final_layer_norm.bias": "ff_norm.bias",
        },
        "output_projection.weight": "classifier.weight",
    },
}


class LearnableEmbedding(Embedding):
    def __init__(
        self,
        vocab_size: int,
        position_size: int,
        d_model: int,
        dropout_rate: float,
        padding_id: int,
        token_embedding: nn.Embedding,
    ):
        super().__init__(dropout_rate=dropout_rate, token_embedding=token_embedding)
        self.padding_id = padding_id
        self.position_embedding = nn.Embedding(position_size, d_model, padding_id)
        self.embedding_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_embedding = self.token_embedding(tokens)
        position_embedding = self.embed_positions(tokens)
        hs = self.embedding_norm(token_embedding + position_embedding)
        hs = self.dropout(hs)
        return hs

    def embed_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, length = tokens.size()
        padding_mask = tokens != self.padding_id
        # tokens: [[11, 12, 1, 1], [11, 12, 13, 14]] (padding_id = 1)
        # positions: [[2, 3, 1, 1], [2, 3, 4, 5]]
        positions = torch.arange(
            self.padding_id,
            length + self.padding_id,
            dtype=torch.long,
            device=tokens.device,
        ).repeat(batch_size, 1)
        positions[padding_mask] = self.padding_id
        return self.position_embedding(positions)


class BartEncoder(Encoder):
    def __init__(
        self,
        vocab_size: int,
        position_size: int,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        num_layers: int,
        dropout_rate: float,
        padding_id: int,
        ff_activation: nn.Module,
        token_embedding: nn.Embedding,
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
            token_embedding=token_embedding,
        )
        self.embedding = LearnableEmbedding(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            token_embedding=token_embedding,
        )


class BartDecoder(Decoder):
    def __init__(
        self,
        vocab_size: int,
        position_size: int,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        num_layers: int,
        dropout_rate: float,
        padding_id: int,
        ff_activation: nn.Module,
        token_embedding: nn.Embedding,
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
            token_embedding=token_embedding,
        )
        self.embedding = LearnableEmbedding(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            token_embedding=token_embedding,
        )


class BartModel(TransformerModel):
    def __init__(
        self,
        vocab_size: int = 51201,
        position_size: int = 1026,
        d_model: int = 768,
        d_ff: int = 3072,
        num_attention_heads: int = 12,
        num_layers: int = 6,
        dropout_rate: float = 0.1,
        bos_id: int = 0,
        eos_id: int = 2,
        padding_id: int = 1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        token_embedding = nn.Embedding(vocab_size, d_model, padding_id)
        self.encoder = BartEncoder(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            ff_activation=nn.GELU(),
            token_embedding=token_embedding,
        )
        self.decoder = BartDecoder(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            ff_activation=nn.GELU(),
            token_embedding=token_embedding,
        )

    @classmethod
    def from_pretrained(cls):
        pretrained_model = torch.hub.load("pytorch/fairseq", "bart.base")
        state_dict = pretrained_model.model.state_dict()
        tgt_dict = {}
        for k in ["encoder.version", "decoder.version"]:
            del state_dict[k]
        for k in tqdm(state_dict.keys()):
            part, module = k.split(".")[:2]
            if module == "layers":
                layer_id, sub_module = k.split(".", 3)[2:]
                tgt_key = (
                    f"{part}.{module}.{layer_id}.{KEY_DICT[part][module][sub_module]}"
                )
            else:
                module = k.split(".", 1)[-1]
                tgt_key = f"{part}.{KEY_DICT[part][module]}"
            tgt_dict[tgt_key] = state_dict[k]
        model = cls()
        model.load_state_dict(tgt_dict)
        return model
