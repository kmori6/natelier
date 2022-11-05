from typing import List, Dict
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence


class NMTTokenizer:
    def __init__(self, path: str, src_lang_token: str, tgt_lang_token: str):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(path)
        self.bos_token_id = self.tokenizer.bos_id()
        self.eos_token_id = self.tokenizer.eos_id()
        self.pad_token_id = self.tokenizer.pad_id()
        self.src_lang_id = self.tokenizer.PieceToId(src_lang_token)
        self.tgt_lang_id = self.tokenizer.PieceToId(tgt_lang_token)

    def __call__(self, text_list: List[str], mode: str) -> Dict[str, torch.Tensor]:
        encodings = {"tokens": [], "masks": []}
        for text in text_list:
            if mode == "src":
                tokens = torch.tensor(self.src_encode(text), dtype=torch.long)
            elif mode == "tgt":
                tokens = torch.tensor(self.tgt_encode(text), dtype=torch.long)
            else:
                tokens = torch.tensor(self.label_encode(text), dtype=torch.long)
            encodings["tokens"].append(tokens)
            encodings["masks"].append(tokens.new_ones(len(tokens)))
        encodings = {
            "tokens": pad_sequence(
                encodings["tokens"],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "masks": pad_sequence(
                encodings["masks"], batch_first=True, padding_value=0
            ),
        }
        return encodings

    def src_encode(self, text: str) -> List[int]:
        return self.tokenizer.EncodeAsIds(text) + [self.eos_token_id, self.src_lang_id]

    def tgt_encode(self, text: str) -> List[int]:
        return (
            [self.tgt_lang_id] + self.tokenizer.EncodeAsIds(text) + [self.eos_token_id]
        )

    def label_encode(self, text: str) -> List[int]:
        return self.tokenizer.EncodeAsIds(text) + [self.eos_token_id, self.tgt_lang_id]

    def decode(self, ids: List[int]):
        return self.tokenizer.DecodeIds(ids)

    @staticmethod
    def train_tokenizer(
        text_file: str,
        vocab_size: int,
        out_dir: str,
        src_lang: str,
        tgt_lang: str,
        bos_token_id: int = 0,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        model_type: str = "bpe",
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
            f"--user_defined_symbols=<{src_lang}>,<{tgt_lang}> "
        )
