from collections import namedtuple, Counter
import pickle
import numpy as np
# from scipy.optimize import linear_sum_assignment
NULL_ID = 0
EVAL_DATA_FILE_NAME = "eval_data_111.jsonl"
EvalDataPoint = namedtuple("EvalDataPoint", ["input_ids",
                                             "attention_mask",
                                             "start_entity_mentions_indices",
                                             "end_entity_mentions_indices",
                                             "start_antecedents_indices",
                                             "end_antecedents_indices",
                                             "gold_clusters",
                                             "mention_logits",
                                             "start_coref_logits",
                                             "end_coref_logits",
                                             ])


def write_examples(f, data):
    for data_point in zip(*data):
        data_point = EvalDataPoint(*data_point)
        pickle.dump(data_point._asdict(), f)


def read_examples(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield EvalDataPoint(**{k: np.array(v) for k, v in pickle.load(f).items()})
            except EOFError:
                break


def flatten_list_of_lists(lst):
    return [elem for sublst in lst for elem in sublst]
