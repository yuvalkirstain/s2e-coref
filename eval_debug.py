# %%

import os
import metrics
from utils import read_examples, EVAL_DATA_FILE_NAME, NULL_ID
import numpy as np
from data import NULL_ID_FOR_COREF
from collections import defaultdict
from transformers import AutoTokenizer

# %%

OUTPUT_DIR = "output_pos"
eval_data_path = os.path.join(OUTPUT_DIR, EVAL_DATA_FILE_NAME)


# tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


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

def trim_by_mention_then_brute_force(mention_logits, start_coref_logits, end_coref_logits):
    candidate_mentions_ravel_ids = np.argpartition(mention_logits.reshape(-1), mention_logits.shape[0] * 10)[
                                   -(mention_logits.shape[0] * 10):]
    candidate_mentions = {np.unravel_index(mention, mention_logits.shape) for mention in candidate_mentions_ravel_ids}
    return brute_force_decode(mention_logits, start_coref_logits, end_coref_logits, candidate_mentions)


# %%
def brute_force_decode(mention_logits, start_coref_logits, end_coref_logits, candidate_mentions=None):
    seq_len = len(start_coref_logits)
    mention_to_antecedent = {}
    for start_mention_idx in range(seq_len):
        for end_mention_idx in range(start_mention_idx, seq_len):
            if candidate_mentions and (start_mention_idx, end_mention_idx) not in candidate_mentions:
                continue
            # mention_score = mention_logits[start_mention_idx, end_mention_idx]
            max_score = -1000000
            antecedent_ids = (NULL_ID, NULL_ID)  # null span
            for end_antecedent_mention_idx in range(start_mention_idx):
                for start_antecedent_mention_idx in range(end_antecedent_mention_idx + 1):
                    if candidate_mentions and (
                    start_antecedent_mention_idx, end_antecedent_mention_idx) not in candidate_mentions:
                        continue
                    # antecedent_mention_score = mention_logits[start_antecedent_mention_idx, end_antecedent_mention_idx]
                    antecedent_start_score = start_coref_logits[start_mention_idx, start_antecedent_mention_idx]
                    antecedent_end_score = end_coref_logits[end_mention_idx, end_antecedent_mention_idx]
                    antecedent_coref_score = antecedent_start_score + antecedent_end_score
                    if max_score < antecedent_coref_score:
                        max_score = antecedent_coref_score
                        antecedent_ids = (start_antecedent_mention_idx, end_antecedent_mention_idx)
            if NULL_ID not in antecedent_ids:
                mention_to_antecedent[(start_mention_idx, end_mention_idx)] = antecedent_ids
    return calc_clusters_predicted_by_mention_to_antecedent(mention_to_antecedent)


# %%

def debug_decode(clusters):
    clusters = clusters[:]
    clusters.remove(clusters[0])
    return clusters


# %%

decoding_func2name = {debug_decode: "debug_func",
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

#%%

for decoding_func, name in decoding_func2name.items():
    coref_evaluator = metrics.CorefEvaluator()
    for eval_data_point in read_examples(eval_data_path):
        gold_clusters = extract_clusters(eval_data_point.gold_clusters)
        mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)

        if "debug" in name:
            predicted_clusters = decoding_func(gold_clusters)
        else:
            predicted_clusters = decoding_func(eval_data_point.mention_logits,
                                               eval_data_point.start_coref_logits,
                                               eval_data_point.end_coref_logits)
        mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
        coref_evaluator.update(predicted_clusters,
                               gold_clusters,
                               mention_to_predicted_clusters,
                               mention_to_gold_clusters)
    dev_prec, dev_rec, dev_f1 = coref_evaluator.get_prf()
    # print("***** Current ckpt path is ***** : {}".format(checkpoint_path))
    print("***** EVAL ON DEV SET *****")
    print(f"***** [DEV EVAL USING {decoding_func2name[decoding_func]}] ***** :\n"
          f"precision: {dev_prec:.4f}, recall: {dev_rec:.4f}, f1: {dev_f1:.4f}")
# TODO: .logging.info

# %%
