from collections import namedtuple, Counter
import pickle
import numpy as np
# from scipy.optimize import linear_sum_assignment
NULL_ID = 0
NULL_ID_FOR_COREF = 0
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


def extract_clusters(gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    return gold_clusters

def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    return mention_to_gold


def extract_clusters_for_decode(mention_to_antecedent):
    mention_to_cluster = {}
    clusters = []
    for mention, antecedent in mention_to_antecedent:
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            clusters[cluster_idx].append(mention)
            mention_to_cluster[mention] = cluster_idx

        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])
    clusters = [tuple(cluster) for cluster in clusters]
    return clusters, mention_to_cluster