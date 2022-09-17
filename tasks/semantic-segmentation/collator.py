from argparse import Namespace
from typing import Dict, List, Any
import random
import torch
from torchvision.transforms import functional
from transformers import SegformerFeatureExtractor


class SSBatchCollator:
    def __init__(self, args: Namespace, train: bool = True, flip_rate: float = 0.5):
        self.train = train
        self.flip_rate = flip_rate
        self.params = SegformerFeatureExtractor.from_pretrained(args.model_name)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images, labels = [], []
        for data in batch:
            image, label = data["image"], data["label"]
            # image: uint8 -> float32, label: unit8 -> int64
            image = functional.convert_image_dtype(image, torch.float32)
            label = functional.convert_image_dtype(label, torch.long)
            # random horizontal flip
            if self.train and random.random() < self.flip_rate:
                image = functional.hflip(image)
                label = functional.hflip(label)
            # normalize
            image = functional.normalize(
                image, mean=self.params.image_mean, std=self.params.image_std
            )
            images.append(image)
            labels.append(label)
        batch = {"images": torch.stack(images), "labels": torch.concat(labels)}
        return batch
