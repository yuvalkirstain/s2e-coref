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
            max_total_seq_len: int,
            sorting_keys: List[str] = None,
            padding_noise: float = 0.1,
            drop_last: bool = False,
            is_eval: bool = False,
    ):
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.max_total_seq_len = max_total_seq_len
        self.data_source = data_source
        data_source.examples.sort(key=lambda x: len(x.token_ids), reverse=True)
        self.drop_last = drop_last
        self.batches = self.prepare_batches() if not is_eval else self.prepare_eval_batches()

    def prepare_batches(self):
        batches = []
        batch = []
        per_example_batch_len = 0
        for elem in self.data_source:
            if len(batch) == 0:
                # TODO change to config.attention_window
                per_example_batch_len = self.calc_effective_per_example_batch_len(len(elem.token_ids))
            elif (len(batch) + 1) * per_example_batch_len > self.max_total_seq_len:
                batch = self.data_source.pad_batch(batch, len(batch[0].token_ids))
                batches.append(batch)
                batch = []
                per_example_batch_len = self.calc_effective_per_example_batch_len(len(elem.token_ids))
            batch.append(elem)
        if len(batch) == 0:
            return batches
        batch = self.data_source.pad_batch(batch, len(batch[0].token_ids))
        batches.append(batch)
        return batches


    def __iter__(self) -> Iterable[List[int]]:
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def calc_effective_per_example_batch_len(self, example_len):
        return math.ceil((example_len + 2) / 512) * 512

    def prepare_eval_batches(self):
        batches = []
        for elem in self.data_source:
            batch = self.data_source.pad_batch([elem], len(elem.token_ids))
            batches.append(batch)
        return batches

