import random
from typing import Iterable, Iterator, List

from torch.utils.data import Dataset


class DefaultSampler:
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


class DefaultBatchSampler:
    def __init__(self, sampler: Iterable[int], batch_size: int, drop_last: bool):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_full_batches = len(sampler) // batch_size
        self.num_remainder = 0 if drop_last else len(sampler) % batch_size

    def __len__(self) -> int:
        if self.num_remainder == 0:
            return self.num_full_batches
        else:
            return self.num_full_batches + 1

    def __iter__(self) -> Iterator[List[int]]:
        iterator = iter(self.sampler)
        for _ in range(self.num_full_batches):
            yield [next(iterator) for _ in range(self.batch_size)]
        if self.num_remainder > 0:
            yield [next(iterator) for _ in range(self.num_remainder)]
