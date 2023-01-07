from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import MBartModel, MBartTokenizerFast

from .bart import BartDecoder, BartEncoder, Bart

KEY_DICT = {
    "encoder": {
        "embed_tokens.weight": "embedding.embedding.weight",
        "embed_positions.weight": "embedding.position_embedding.weight",
        "layernorm_embedding.weight": "embedding.embedding_norm.weight",
        "layernorm_embedding.bias": "embedding.embedding_norm.bias",
        "layer_norm.weight": "norm.weight",
        "layer_norm.bias": "norm.bias",
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
        "embed_tokens.weight": "embedding.embedding.weight",
        "embed_positions.weight": "embedding.position_embedding.weight",
        "layernorm_embedding.weight": "embedding.embedding_norm.weight",
        "layernorm_embedding.bias": "embedding.embedding_norm.bias",
        "layer_norm.weight": "norm.weight",
        "layer_norm.bias": "norm.bias",
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
    },
}


class MbartEncoder(BartEncoder):
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
        embedding: nn.Embedding,
    ):
        super().__init__(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            ff_activation=ff_activation,
            embedding=embedding,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, tokens: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hs = self.embedding(tokens)
        for layer in self.layers:
            hs, masks = layer(hs, masks)
        hs = self.norm(hs)
        return hs, masks


class MbartDecoder(BartDecoder):
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
        embedding: nn.Embedding,
    ):
        super().__init__(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            ff_activation=ff_activation,
            embedding=embedding,
        )
        self.norm = nn.LayerNorm(d_model)

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
        hs = self.norm(hs)
        hs = self.classifier(hs)
        return hs


class Mbart(Bart):
    def __init__(
        self,
        vocab_size: int = 250027,
        position_size: int = 1026,
        d_model: int = 1024,
        d_ff: int = 4096,
        num_attention_heads: int = 16,
        num_layers: int = 12,
        dropout_rate: float = 0.3,
        bos_id: int = 0,
        eos_id: int = 2,
        padding_id: int = 1,
        load_pretrained_weight: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bos_id=bos_id,
            eos_id=eos_id,
            padding_id=padding_id,
        )
        embedding = nn.Embedding(vocab_size, d_model, padding_id)
        self.encoder = MbartEncoder(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            ff_activation=nn.GELU(),
            embedding=embedding,
        )
        self.decoder = MbartDecoder(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            ff_activation=nn.GELU(),
            embedding=embedding,
        )
        if load_pretrained_weight:
            self.load_pretrained_weight()

    def load_pretrained_weight(self):
        pretrained_model = MBartModel.from_pretrained("facebook/mbart-large-cc25")
        state_dict = pretrained_model.state_dict()
        tgt_dict = {}
        for k in tqdm(state_dict.keys()):
            part, module = k.split(".")[:2]
            if part == "shared":
                tgt_key = "decoder.classifier.weight"
            else:
                if module == "layers":
                    layer_id, sub_module = k.split(".", 3)[2:]
                    tgt_key = (
                        f"{part}.{module}.{layer_id}."
                        f"{KEY_DICT[part][module][sub_module]}"
                    )
                else:
                    module = k.split(".", 1)[-1]
                    tgt_key = f"{part}.{KEY_DICT[part][module]}"
            tgt_dict[tgt_key] = state_dict[k]
        self.load_state_dict(tgt_dict)

    @staticmethod
    def get_pretrained_tokenizer(src_lang: str, tgt_lang: str) -> MBartTokenizerFast:
        return MBartTokenizerFast.from_pretrained(
            "facebook/mbart-large-cc25", src_lang=src_lang, tgt_lang=tgt_lang
        )
