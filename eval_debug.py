# %%

import os
import metrics
from utils import read_examples, EVAL_DATA_FILE_NAME, NULL_ID
import numpy as np
from data import NULL_ID_FOR_COREF
from collections import defaultdict
from transformers import AutoTokenizer

# %%

OUTPUT_DIR = "output"
eval_data_path = os.path.join(OUTPUT_DIR, EVAL_DATA_FILE_NAME)

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


def find_and_remove_cluster_by_mention(mention, clusters):
    mention_cluster = None
    for cluster in clusters:
        if mention not in cluster:
            continue
        mention_cluster = cluster
        break
    if mention_cluster is None:
        return {mention}
    clusters.remove(mention_cluster)
    return mention_cluster


def calc_clusters_predicted_by_mention_to_antecedent(mention_to_antecedent):
    clusters = []
    for mention, antecedent in mention_to_antecedent.items():
        mention_cluster = find_and_remove_cluster_by_mention(mention, clusters)
        antecedent_cluster = find_and_remove_cluster_by_mention(antecedent, clusters)
        united_cluster = mention_cluster | antecedent_cluster
        clusters.append(united_cluster)
    return [tuple(cluster) for cluster in clusters]


# %% md

### Implement here decoding strategies

# %%

def trim_mentions(mention_logits, mention_to_gold_clusters):
    num_mentions = int(mention_logits.shape[0] * 0.4)
    candidate_mentions_ravel_ids = np.argpartition(mention_logits.reshape(-1), num_mentions)[-num_mentions:]
    candidate_mentions = {np.unravel_index(mention, mention_logits.shape) for mention in candidate_mentions_ravel_ids}
    # candidates = np.where(mention_logits > 7)
    # candidate_mentions = [(x, y) for x, y in zip(candidates[0], candidates[1])]
    print(calc_mention_metrics(mention_to_gold_clusters, candidate_mentions))

    return candidate_mentions


def trim_by_mention_then_brute_force(mention_logits, start_coref_logits, end_coref_logits, mention_to_gold_clusters):
    candidate_mentions = trim_mentions(mention_logits, mention_to_gold_clusters)
    candidate_mentions = [(s, e) for s, e in candidate_mentions]
    return brute_force_decode(mention_logits, start_coref_logits, end_coref_logits,
                              candidate_mentions, mention_to_gold_clusters), candidate_mentions


# %%
def brute_force_decode(mention_logits, start_coref_logits, end_coref_logits, candidate_mentions, mention_to_gold_clusters):
    mention_to_antecedent = {}
    mention_to_score = defaultdict(float)
    mention_to_antecedents_scores = defaultdict(list)
    gold_mentions = sorted(mention_to_gold_clusters.keys(), key=lambda x: x[0], reverse=True)
    for start_id, end_id in candidate_mentions:
        candidate_start_antecedents = np.argpartition(-start_coref_logits[start_id].reshape(-1), 50)[:50]
        candidate_end_antecedents = np.argpartition(-end_coref_logits[end_id].reshape(-1), 50)[:50]
        candidate_antecedents = np.transpose([np.tile(candidate_start_antecedents, len(candidate_end_antecedents)),
                                              np.repeat(candidate_end_antecedents, len(candidate_start_antecedents))])
        for start_antecedent_id, end_antecedent_id in candidate_antecedents:
            if not (start_antecedent_id <= end_antecedent_id < start_antecedent_id + 30):
                continue
            antecedent_start_score = start_coref_logits[start_id, start_antecedent_id]
            null_start_score = start_coref_logits[start_id, 0]
            if antecedent_start_score <= 0 or antecedent_start_score <= null_start_score:
                continue
            antecedent_end_score = end_coref_logits[end_id, end_antecedent_id]
            null_end_score = end_coref_logits[end_id, 0]
            if antecedent_end_score <= 0 or antecedent_end_score <= null_end_score:
                continue
            mention_score = mention_logits[start_id, end_id]
            antecedent_mention_score = mention_logits[start_antecedent_id, end_antecedent_id]
            antecedent_coref_score = antecedent_start_score + antecedent_end_score + antecedent_mention_score
            if mention_to_score[(start_id, end_id)] < antecedent_coref_score:
                mention_to_antecedent[(start_id, end_id)] = [(start_antecedent_id, end_antecedent_id), antecedent_coref_score]
                mention_to_score[(start_id, end_id)] = antecedent_coref_score
            mention_to_antecedents_scores[(start_id, end_id)].append([(start_antecedent_id, end_antecedent_id), antecedent_coref_score])
    return calc_clusters_predicted_by_mention_to_antecedent(mention_to_antecedent)


# %%

def debug_decode(clusters):
    clusters = clusters[:]
    clusters.remove(clusters[0])
    return clusters


# %%

decoding_func2name = {  # debug_decode: "debug_func",
    # brute_force_decode: "brute_force",
    trim_by_mention_then_brute_force: "trim_then_brute_force"}


# %%

def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    return mention_to_gold


# %%


# predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
def extract_clusters(gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    return gold_clusters


# %%

def calc_mention_metrics(mention_to_gold_clusters, predicted_mentions):
    predicted_mentions = set(predicted_mentions)
    gold_mentions = set(mention_to_gold_clusters.keys())
    precision = len(predicted_mentions & gold_mentions) / len(predicted_mentions)
    recall = len(gold_mentions & predicted_mentions) / len(gold_mentions)
    f1 = 2 * precision * recall / (recall + precision) if recall != 0 and precision != 0 else 0
    return precision, recall, f1


for decoding_func, name in decoding_func2name.items():
    coref_evaluator = metrics.CorefEvaluator()
    num_examples = 0
    mention_precision, mentions_recall, mention_f1 = 0, 0, 0
    for eval_data_point in read_examples(eval_data_path):
        num_examples += 1
        gold_clusters = extract_clusters(eval_data_point.gold_clusters)
        mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)

        predicted_clusters, predicted_mentions = decoding_func(eval_data_point.mention_logits,
                                                               eval_data_point.start_coref_logits,
                                                               eval_data_point.end_coref_logits,
                                                               mention_to_gold_clusters)
        cur_mention_precision, cur_mentions_recall, cur_mention_f1 = calc_mention_metrics(mention_to_gold_clusters, predicted_mentions)
        mention_precision += cur_mention_precision
        mentions_recall += cur_mentions_recall
        mention_f1 += cur_mention_f1
        mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
        coref_evaluator.update(predicted_clusters,
                               gold_clusters,
                               mention_to_predicted_clusters,
                               mention_to_gold_clusters)
    dev_prec, dev_rec, dev_f1 = coref_evaluator.get_prf()
    # print("***** Current ckpt path is ***** : {}".format(checkpoint_path))
    print("***** EVAL ON DEV SET *****")
    print(f"function is {decoding_func2name[decoding_func]}")
    print(f"number of examples is {num_examples}")
    print(f"precision: {dev_prec:.4f}, recall: {dev_rec:.4f}, f1: {dev_f1:.4f}")
    print(f"mention precision: {mention_precision / num_examples:.4f}, "
          f"mention recall: {mentions_recall / num_examples:.4f}, "
          f"mention f1: {mention_f1 / num_examples:.4f}")
# TODO: .logging.info

# %%
