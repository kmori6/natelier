from argparse import Namespace
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer


class NMTBatchCollator:
    def __init__(self, args: Namespace, return_test_encodings: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, src_lang=args.src_lang, tgt_lang=args.tgt_lang
        )
        self.max_length = min(args.max_length, self.tokenizer.model_max_length)
        self.return_test_encodings = return_test_encodings
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang_id = self.tokenizer.lang_code_to_id[args.src_lang]
        self.tgt_lang_id = self.tokenizer.lang_code_to_id[args.tgt_lang]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encodings = self.tokenizer(
            [data["src_text"] for data in batch],
            text_target=[data["tgt_text"] for data in batch],
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        if self.return_test_encodings:
            return {
                "tokens": encodings["input_ids"],
                "labels": encodings["labels"].masked_fill(
                    encodings["labels"] == 1, -100
                ),
            }
        else:
            decoder_tokens = encodings["labels"].masked_fill(
                encodings["labels"] == self.tgt_lang_id, self.pad_token_id
            )
            decoder_tokens = decoder_tokens.roll(1, dims=1)
            decoder_tokens[:, 0] = self.tgt_lang_id
            decoder_masks = (decoder_tokens != self.pad_token_id).long()
            return {
                "encoder_tokens": encodings["input_ids"],
                "encoder_masks": encodings["attention_mask"],
                "decoder_tokens": decoder_tokens,
                "decoder_masks": decoder_masks,
                "labels": encodings["labels"].masked_fill(
                    encodings["labels"] == 1, -100
                ),
            }
