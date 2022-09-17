from typing import Dict
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation


class VocDataset(Dataset):
    def __init__(self, download_dir: str, split: str, size: int = 256):
        super().__init__()
        if split == "train":
            self.dataset = VOCSegmentation(download_dir, "2012", "train", True)
        else:
            self.dataset = VOCSegmentation(download_dir, "2012", "val", True)
        self.converter = transforms.Compose(
            [transforms.PILToTensor(), transforms.Resize((size, size))]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, label = self.dataset[idx]
        return {"image": self.converter(image), "label": self.converter(label)}
