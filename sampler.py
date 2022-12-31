import random
from typing import Iterator, List

from torch.utils.data import BatchSampler, Dataset, Sampler


class DefaultSampler(Sampler):
    def __init__(self, dataset: Dataset, shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        idx_list = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idx_list)
        return iter(idx_list)
