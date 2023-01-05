import os
from typing import Dict, List

import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class NMTTokenizer:
    def __init__(self, path: str, src_lang_token: str, tgt_lang_token: str):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(path)
        self.bos_token_id = self.tokenizer.bos_id()
        self.eos_token_id = self.tokenizer.eos_id()
        self.pad_token_id = self.tokenizer.pad_id()
        self.src_lang_id = self.tokenizer.PieceToId(src_lang_token)
        self.tgt_lang_id = self.tokenizer.PieceToId(tgt_lang_token)

    def __call__(
        self, text_source: str, text_target: str = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        src_tokens = self.encode(text_source)
        results = {"input_ids": torch.tensor([src_tokens], dtype=torch.long)}
        if text_target:
            tgt_tokens = self.encode(text_target)
            results["labels"] = torch.tensor([tgt_tokens], dtype=torch.long)
        return results

    def encode(self, text: str) -> List[int]:
        return (
            [self.bos_token_id] + self.tokenizer.EncodeAsIds(text) + [self.eos_token_id]
        )

    def decode(self, ids: List[int]):
        return self.tokenizer.DecodeIds(ids)

    @staticmethod
    def train(
        train_dataset: Dataset,
        vocab_size: int,
        out_dir: str,
        src_lang: str,
        tgt_lang: str,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        model_type: str = "bpe",
    ):
        os.makedirs(f"{out_dir}/tokenizer", exist_ok=True)
        text_file = f"{out_dir}/train_text.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            for i in tqdm(
                range(len(train_dataset)), total=len(train_dataset), desc="Loading"
            ):
                src_text = train_dataset[i]["src_text"]
                tgt_text = train_dataset[i]["tgt_text"]
                f.write(src_text + "\n" + tgt_text + "\n")
        spm.SentencePieceTrainer.train(
            f"--input={text_file} "
            f"--bos_id={bos_token_id} "
            f"--pad_id={pad_token_id} "
            f"--eos_id={eos_token_id} "
            f"--unk_id={unk_token_id} "
            f"--model_prefix={out_dir}/tokenizer "
            f"--vocab_size={vocab_size} "
            f"--model_type={model_type} "
            f"--user_defined_symbols=<{src_lang}>,<{tgt_lang}> "
        )
        os.remove(text_file)
