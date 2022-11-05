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
        waves, wave_lens = [], []
        encoder_labels, decoder_tokens, decoder_labels = [], [], []
        for data in batch:
            # speech processing
            wave = data["waveform"]
            waves.append(wave)
            wave_lens.append(len(wave))
            # text processing
            label = self.tokenizer(data["transcript"].replace(" ", "|"))
            encoder_label = torch.tensor(label, dtype=torch.long)
            decoder_token = torch.tensor([self.bos_token_id] + label, dtype=torch.long)
            decoder_label = torch.tensor(label + [self.eos_token_id], dtype=torch.long)
            encoder_labels.append(encoder_label)
            decoder_tokens.append(decoder_token)
            decoder_labels.append(decoder_label)
        waves = pad_sequence(waves, batch_first=True, padding_value=0.0)
        wave_lens = torch.tensor(wave_lens, dtype=torch.long)
        encoder_labels = pad_sequence(
            encoder_labels, batch_first=True, padding_value=-100
        )
        decoder_tokens = pad_sequence(
            decoder_tokens, batch_first=True, padding_value=self.eos_token_id
        )
        decoder_labels = pad_sequence(
            decoder_labels, batch_first=True, padding_value=-100
        )
        inputs = {
            "encoder_waves": waves,
            "encoder_wave_lens": wave_lens,
            "encoder_labels": encoder_labels,
            "encoder_label_lens": (encoder_labels != -100).sum(-1),
            "decoder_tokens": decoder_tokens,
            "decoder_masks": self.decoder_masks(decoder_tokens),
            "decoder_labels": decoder_labels,
        }
        if self.return_transcript:
            inputs["transcript"] = [data["transcript"] for data in batch]
        return inputs

    def decoder_masks(self, decoder_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = len(decoder_tokens)
        lengths = (decoder_tokens != self.eos_token_id).sum(-1) + 1
        max_length = max(lengths)
        masks = pad_sequence(
            [torch.ones(length) for length in lengths],
            batch_first=True,
            padding_value=0,
        )
        masks = torch.tril(
            masks.repeat_interleave(max_length, 0).view(
                batch_size, max_length, max_length
            )
        )
        return masks
