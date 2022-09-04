from typing import Dict, Any
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechcommandsDataset(Dataset):
    def __init__(self, subset: str, download_dir: str):
        super().__init__()
        self.dataset = SPEECHCOMMANDS(root=download_dir, subset=subset, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        waveform, _, label, *_ = self.dataset[idx]
        return {"waveform": waveform[0], "label": label}
