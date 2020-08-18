import logging
from itertools import islice
from typing import List, Iterable, Tuple, Iterator, Sequence
import random
import math

from torch.utils import data
from data import CorefExample
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise


class BucketBatchSampler(DataLoader):
    def __init__(
        self,
        data_source: data.Dataset,
        batch_size: int,
        sorting_keys: List[str] = None,
        padding_noise: float = 0.1,
        drop_last: bool = False,
    ):

        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.batch_size = batch_size
        self.data_source = data_source
        self.drop_last = drop_last

    def _argsort_by_padding(
        self, instances: Iterable[CorefExample]
    ) -> Sequence[Tuple[CorefExample, int]]:
        """
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of `(field_name, padding_key)` tuples.
        """
        instances_with_length = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            length = len(instance[0])
            noisy_length = add_noise_to_value(length, self.padding_noise)
            instances_with_length.append((noisy_length, length, instance))
        with_indices = [((noisy_length, length, instance), i) for i, (noisy_length, length, instance) in enumerate(instances_with_length)]
        with_indices.sort(key=lambda x: x[0][0])
        return ((instance_with_index[0][2], instance_with_index[0][1]) for instance_with_index in with_indices)

    def __iter__(self) -> Iterable[List[int]]:
        instance_lengths = self._argsort_by_padding(self.data_source)
        batches = []
        for group in lazy_groups_of(instance_lengths, self.batch_size):
            batch_examples, lengths = zip(*list(group))
            batch = self.data_source.pad_batch(batch_examples, max(lengths))
            batches.append(batch)
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        batch_count_float = len(self.data_source) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)


def lazy_groups_of(iterable: Sequence[Tuple[CorefExample, int]], group_size: int) -> Iterator[List[Tuple[CorefExample, int]]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break