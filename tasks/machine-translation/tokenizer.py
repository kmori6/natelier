from typing import List, Dict
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence


class NMTTokenizer:
    def __init__(self, path: str):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(path)
        self.bos_token_id = self.tokenizer.bos_id()
        self.eos_token_id = self.tokenizer.eos_id()
        self.pad_token_id = self.tokenizer.pad_id()

    def __call__(self, text_list: List[str]) -> Dict[str, torch.Tensor]:
        encodings = {"input_ids": [], "attention_mask": []}
        for text in text_list:
            input_ids = torch.tensor(self.encode(text), dtype=torch.long)
            encodings["input_ids"].append(input_ids)
            encodings["attention_mask"].append(input_ids.new_ones(len(input_ids)))
        encodings = {
            "input_ids": pad_sequence(
                encodings["input_ids"],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                encodings["attention_mask"], batch_first=True, padding_value=0
            ),
        }
        return encodings

    def encode(self, text: str) -> List[int]:
        return (
            [self.bos_token_id] + self.tokenizer.EncodeAsIds(text) + [self.eos_token_id]
        )

    def decode(self, ids: List[int]):
        return self.tokenizer.DecodeIds(ids)

    @staticmethod
    def train_tokenizer(
        text_file: str,
        vocab_size: int,
        out_dir: str,
        bos_token_id: int = 0,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        model_type: str = "unigram",
    ):
        spm.SentencePieceTrainer.train(
            f"--input={text_file} "
            f"--bos_id={bos_token_id} "
            f"--pad_id={pad_token_id}"
            f"--eos_id={eos_token_id} "
            f"--unk_id={unk_token_id} "
            f"--model_prefix={out_dir}/tokenizer "
            f"--vocab_size={vocab_size} "
            f"--model_type={model_type} "
        )
