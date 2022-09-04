from typing import Dict, Any
import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST


class MnistDataset(Dataset):
    def __init__(self, download_dir: str, subset: str, num_dev_samples=5000):
        super().__init__()
        if subset in ["train", "validation"]:
            datasets = MNIST(root=download_dir, train=True, download=True)
            datasets = random_split(
                datasets,
                [len(datasets) - (num_dev_samples), num_dev_samples],
                generator=torch.Generator().manual_seed(0),
            )
            if subset == "train":
                self.dataset = datasets[0]
            else:
                self.dataset = datasets[1]
        else:
            self.dataset = MNIST(root=download_dir, train=False, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.dataset[idx]
        return {"image": image, "label": label}
