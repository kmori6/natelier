from typing import Dict, List, Any
import numpy as np
import torch


def mnist_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images, labels = [], []
    for data in batch:
        image = (np.array(data["image"]) - 0.5) / 0.5
        images.append(torch.tensor(image, dtype=torch.float32).unsqueeze(0))
        labels.append(torch.tensor(data["label"], dtype=torch.long))
    batch = {"images": torch.stack(images), "labels": torch.stack(labels)}
    return batch
