import random
from typing import Iterator, List

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class DefaultBatchSampler:
    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last and len(self.dataset) % self.batch_size == 0:
            return len(self.dataset) // self.batch_size
        else:
            return len(self.dataset) // self.batch_size + 1

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

    def make_batches(self) -> List[List[int]]:
        raise NotImplementedError()


class LengthGroupBatchSampler(DefaultBatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        key: str = "encoder_tokens",
        **kwargs,
    ):
        super().__init__(dataset, batch_size, shuffle, drop_last)
        self.lengths = [len(self.dataset[i][key]) for i in range(len(self.dataset))]
        self.sorted_indices = np.argsort(self.lengths)[::-1]
        self.batches = self.make_batches()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}: "
            f"{{batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"drop_last={self.drop_last}}}"
        )

    def __iter__(self) -> Iterator[List[int]]:
        batches = self.batches.copy()
        if self.drop_last and len(self.dataset) % self.batch_size > 0:
            batches = batches[:-1]
        if self.shuffle:
            random.shuffle(batches)
        iterator = iter(batches)
        return iterator

    def make_batches(self) -> List[List[int]]:
        split_indices = np.arange(self.batch_size, len(self.dataset), self.batch_size)
        batches = np.split(self.sorted_indices, split_indices)
        return [batch.tolist() for batch in batches]


class TotalLengthBatchSampler(LengthGroupBatchSampler):
    def __init__(self, dataset: Dataset, max_length: int, shuffle: bool, **kwargs):
        super().__init__(dataset, batch_size=None, shuffle=shuffle, drop_last=False)
        self.max_length = max_length
        self.batches = self.make_batches()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}: "
            f"{{max_length={self.max_length}, shuffle={self.shuffle}}}"
        )

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[List[int]]:
        batches = self.batches.copy()
        if self.shuffle:
            random.shuffle(batches)
        iterator = iter(batches)
        return iterator

    def make_batches(self) -> List[List[int]]:
        batch, batches = [], []
        total_length = batch_length = 0
        for i in tqdm(self.sorted_indices, desc="Making batches"):
            length = self.lengths[i]
            if batch_length == 0:
                batch_length = length
            if total_length + batch_length <= self.max_length:
                batch.append(length)
                total_length += batch_length
            else:
                if len(batch) == 0:
                    raise RuntimeError(
                        f"empty batch exists for a {length} length input. "
                        f"increase max_batch_length from {self.max_length}."
                    )
                else:
                    batches.append(batch)
                    batch = [length]
                    total_length = batch_length = length
            if i == self.sorted_indices[-1]:
                if len(batch) > 0:
                    batches.append(batch)
        return batches
