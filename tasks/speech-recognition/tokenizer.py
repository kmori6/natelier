import json
from argparse import Namespace
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset


class ASRTokenizer:
    def __init__(self, token_list: List[str]):
        self.token_list = token_list
        self.token2label = {token: i for i, token in enumerate(token_list)}
        self.label2token = {i: token for i, token in enumerate(token_list)}
        self.unk_label = self.token2label["<unk>"]
        self.bos_id = self.token2label["<bos>"]
        self.eos_id = self.token2label["<eos>"]
        self.pad_id = self.token2label["<pad>"]

    def __call__(self, text: str) -> List[int]:
        return [
            self.token2label.get(token, self.unk_label) for token in [c for c in text]
        ]

    @classmethod
    def build_from_dataset(cls, args: Namespace, train_dataset: Dataset):
        # build dictionary
        token_list = set()
        for data in tqdm(train_dataset):
            for char in data["transcript"].replace(" ", "|"):
                token_list.add(char)
        # ctc blank token: <blank>
        token_list = ["<blank>", "<bos>", "<eos>", "<pad>", "<unk>"] + sorted(
            list(token_list)
        )
        setattr(args, "token_list", token_list)
        setattr(args, "vocab_size", len(token_list))
        return cls(token_list)

    @classmethod
    def load_from_args(cls, args: Namespace):
        # load training args
        with open(args.out_dir + "/train_args.json", "r") as f:
            train_args = Namespace(**json.load(f))
        setattr(args, "token_list", train_args.token_list)
        setattr(args, "vocab_size", len(train_args.token_list))
        return cls(train_args.token_list)
