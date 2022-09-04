import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any
from tokenizer import ASRTokenizer


class ASRBatchCollator:
    def __init__(self, tokenizer: ASRTokenizer, return_transcript: bool = False):
        self.tokenizer = tokenizer
        self.return_transcript = return_transcript
        self.bos_token_id = len(tokenizer.token_list) - 1
        self.eos_token_id = len(tokenizer.token_list) - 1

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = {
            "input_wavs": pad_sequence(
                [data["waveform"] for data in batch],
                batch_first=True,
                padding_value=0,
            ),
            "input_wav_lens": torch.tensor(
                [len(data["waveform"]) for data in batch], dtype=torch.long
            ),
            "encoder_labels": pad_sequence(
                [
                    torch.tensor(
                        self.tokenizer(data["transcript"].replace(" ", "|")),
                        dtype=torch.long,
                    )
                    for data in batch
                ],
                batch_first=True,
                padding_value=-100,
            ),
            "encoder_label_lens": torch.tensor(
                [
                    len(self.tokenizer(data["transcript"].replace(" ", "|")))
                    for data in batch
                ],
                dtype=torch.long,
            ),
            "decoder_input_ids": pad_sequence(
                [
                    torch.tensor(
                        [self.bos_token_id]
                        + self.tokenizer(data["transcript"].replace(" ", "|")),
                        dtype=torch.long,
                    )
                    for data in batch
                ],
                batch_first=True,
                padding_value=self.eos_token_id,
            ),
            "decoder_labels": pad_sequence(
                [
                    torch.tensor(
                        self.tokenizer(data["transcript"].replace(" ", "|"))
                        + [self.eos_token_id],
                        dtype=torch.long,
                    )
                    for data in batch
                ],
                batch_first=True,
                padding_value=-100,
            ),
        }
        if self.return_transcript:
            inputs["transcript"] = [data["transcript"] for data in batch]
        return inputs
