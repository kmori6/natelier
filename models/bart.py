from typing import Tuple, Dict, Any
from tqdm import tqdm
import torch
import torch.nn as nn
from .modules.learnable_embedding import LearnableEmbedding
from .modules.multi_head_attention import MultiHeadAttention
from .modules.feed_forward import FeedForward

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


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, num_attention_heads: int, dropout_rate: float
    ):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_attention_heads, dropout_rate)
        self.mha = MultiHeadAttention(d_model, num_attention_heads, dropout_rate)
        self.ff = FeedForward(d_model, d_ff)
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
        position_size: int,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        num_layers: int,
        dropout_rate: float,
        padding_id: int,
        input_embedding: nn.Embedding,
    ):
        super().__init__()
        self.embedding = LearnableEmbedding(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            token_embedding=input_embedding,
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, d_ff, num_attention_heads, dropout_rate)
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
        position_size: int,
        d_model: int,
        d_ff: int,
        num_attention_heads: int,
        num_layers: int,
        dropout_rate: float,
        padding_id: int,
        output_embedding: nn.Embedding,
    ):
        super().__init__()
        self.embedding = LearnableEmbedding(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            token_embedding=output_embedding,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, d_ff, num_attention_heads, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)
        self.classifier.weight = output_embedding.weight

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


class BartModel(nn.Module):
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
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.padding_id = padding_id
        common_token_embedding = nn.Embedding(vocab_size, d_model, padding_id)
        self.encoder = Encoder(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            input_embedding=common_token_embedding,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            position_size=position_size,
            d_model=d_model,
            d_ff=d_ff,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            padding_id=padding_id,
            output_embedding=common_token_embedding,
        )

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
        self,
        tokens: torch.Tensor,
        beam_size: int = 5,
        max_length: int = 128,
    ) -> Dict[str, Any]:

        # initial stats
        running_stats = [{"score": 0.0, "tokens": [self.bos_id]}]
        final_stats = []

        # encoder forward
        encoder_inputs = tokens.repeat_interleave(beam_size, dim=0)
        encoder_masks = torch.ones_like(encoder_inputs)
        encoder_hs, encoder_masks = self.encoder(encoder_inputs, encoder_masks)

        # beam search
        for i in range(1, max_length):

            # decoder forward
            decoder_tokens = tokens.new_zeros(beam_size, i)
            for beam_idx, stat in enumerate(running_stats):
                decoder_tokens[beam_idx, :] = tokens.new_tensor(stat["tokens"])
            decoder_masks = decoder_tokens.new_ones(beam_size, i, i).tril()
            logits = self.decoder(
                decoder_tokens, decoder_masks, encoder_hs, encoder_masks
            )
            next_token_scores = torch.log_softmax(logits[:, -1, :], dim=-1)  # (B, D)

            # scoring
            aggregator = []
            for beam_idx, stat in enumerate(running_stats):
                next_scores, next_tokens = torch.topk(
                    next_token_scores[beam_idx], beam_size
                )
                for topk_idx in range(beam_size):
                    candidate = {
                        "score": stat["score"] + next_scores[topk_idx].item(),
                        "tokens": stat["tokens"] + [next_tokens[topk_idx].item()],
                    }
                    aggregator.append(candidate)
            running_stats = sorted(aggregator, key=lambda x: x["score"], reverse=True)[
                :beam_size
            ]

            # add eos_token_id
            if i == max_length - 1:
                for stat in running_stats:
                    stat["tokens"].append(self.eos_id)

            # sort stats
            keep_stats = []
            for stat in running_stats:
                if stat["tokens"][-1] == self.eos_id:
                    final_stats.append(stat)
                else:
                    keep_stats.append(stat)
            running_stats = keep_stats

            # stop search
            if len(running_stats) < 1 or len(final_stats) >= beam_size:
                break

        return sorted(final_stats, key=lambda x: x["score"], reverse=True)[0]

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
