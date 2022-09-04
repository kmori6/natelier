from argparse import Namespace
from typing import Dict, Any, List
import torch
from torch.nn.utils.rnn import pad_sequence


class SCBatchCollator:
    def __init__(self, args: Namespace):
        self.command2label = {c: i for i, c in enumerate(args.command_list)}

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ws": pad_sequence(
                [data["waveform"] for data in batch], batch_first=True, padding_value=0
            ),
            "input_lens": torch.tensor(
                [len(data["waveform"]) for data in batch], dtype=torch.long
            ),
            "labels": torch.tensor(
                [self.command2label[data["label"]] for data in batch], dtype=torch.long
            ),
        }
