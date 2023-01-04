import random
from typing import Iterator, List

import numpy as np
from torch.utils.data import Dataset


class DefaultBatchSampler:
    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_full_batches = len(dataset) // batch_size
        self.num_remainder = 0 if drop_last else len(dataset) % batch_size

    def __len__(self) -> int:
        if self.num_remainder == 0:
            return self.num_full_batches
        else:
            return self.num_full_batches + 1

    def __iter__(self) -> Iterator[List[int]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            random.shuffle(indices)
        indices = np.split(
            indices, np.arange(self.batch_size, len(indices), self.batch_size)
        )
        if self.drop_last and len(self.dataset) % self.batch_size > 0:
            indices = indices[:-1]
        iterator = iter([array.tolist() for array in indices])
        return iterator


class LengthGroupBatchSampler(DefaultBatchSampler):
    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool
    ):
        super().__init__(dataset, batch_size, shuffle, drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        lengths = [
            len(self.dataset[i]["encoder_tokens"]) for i in range(len(self.dataset))
        ]
        indices = np.split(
            np.argsort(lengths)[::-1],
            np.arange(self.batch_size, len(self.dataset), self.batch_size),
        )
        if self.drop_last and len(self.dataset) % self.batch_size > 0:
            indices = indices[:-1]
        if self.shuffle:
            random.shuffle(indices)
        iterator = iter([array.tolist() for array in indices])
        return iterator
