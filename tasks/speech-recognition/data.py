import os
from torchaudio.datasets import LIBRISPEECH, LibriLightLimited
from torch.utils.data import Dataset
from typing import Any, Dict


class LibriDataset(Dataset):
    def __init__(
        self,
        download_dir: str,
        subset: str,
        min_sec: float = 1.0,
        max_sec: float = 20.0,
    ):
        super().__init__()
        os.makedirs(download_dir, exist_ok=True)
        if subset in ["train-10min", "train-1h", "train-10h"]:
            self.dataset = LibriLightLimited(
                root=download_dir, subset=subset.split("-")[-1], download=True
            )
        else:
            self.dataset = LIBRISPEECH(root=download_dir, url=subset, download=True)
        # remove short and long speech
        self.valid_indices = [
            i
            for i, data in enumerate(self.dataset)
            if len(data[0][0]) / 16000 >= min_sec and len(data[0][0]) / 16000 <= max_sec
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        waveform, sample_rate, transcript, *_ = self.dataset[self.valid_indices[idx]]
        assert sample_rate == 16000
        return {"waveform": waveform[0], "transcript": transcript}
