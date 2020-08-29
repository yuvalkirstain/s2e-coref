from collections import defaultdict

import numpy as np

NULL_SPAN_IDX = 0

def prune_top_k(mention_logits, top_k):
    """
    Gets a numpy array of size [seq_length, seq_length] of logits, and returns the indices of the top_k top-scored spans
    :param mention_logits:
    :param top_k:
    """
    candidate_mentions_ravel_ids_partition = np.argpartition(-mention_logits.reshape(-1), kth=top_k)
    # Taking the last k:
    candidate_mentions_ravel_ids = candidate_mentions_ravel_ids_partition[:top_k]
    candidate_mentions = sorted(
        np.unravel_index(mention, mention_logits.shape) for mention in candidate_mentions_ravel_ids)
    candidate_mentions = [(start, end) for start, end in candidate_mentions if mention_logits[start, end] > 0]
    return candidate_mentions


def find_antecedent_of_mentions(candidate_spans, mention_logits, start_coref_logits, end_coref_logits):
    mention_to_antecedent = []
    for span_idx, (start_id, end_id) in enumerate(candidate_spans):
        # Calculate the null-antecedent score
        null_start_score = start_coref_logits[start_id, 0]
        null_end_score = end_coref_logits[end_id, 0]
        null_antecedent_score = null_start_score + null_end_score

        # Get all "previous" spans (these are the real candidates for antecedents
        candidate_antecedents = candidate_spans[:span_idx]
        if not candidate_antecedents:
            continue
        candidate_antecedents_start_indices, candidate_antecedents_end_indices = zip(*candidate_antecedents)

        candidate_antecedent_mention_logits = mention_logits[candidate_antecedents_start_indices, candidate_antecedents_end_indices]
        candidate_antecedent_start_logits = start_coref_logits[start_id, candidate_antecedents_start_indices]
        candidate_antecedent_end_logits = end_coref_logits[end_id, candidate_antecedents_end_indices]

        candidate_antecedent_scores = candidate_antecedent_mention_logits + candidate_antecedent_start_logits + candidate_antecedent_end_logits
        max_antecedent_score, argmax_antecedent = np.max(candidate_antecedent_scores), np.argmax(candidate_antecedent_scores)
        argmax_antecedent_start, argmax_antecedent_end = candidate_antecedents[argmax_antecedent]
        argmax_antecedent_mention_logit = mention_logits[argmax_antecedent_start, argmax_antecedent_end]

        # TODO check if we need to deduct mention
        if max_antecedent_score - argmax_antecedent_mention_logit > null_antecedent_score:
            mention_to_antecedent.append(((start_id, end_id), (argmax_antecedent_start, argmax_antecedent_end)))

    return mention_to_antecedent


def cluster_mentions(mention_logits, start_coref_logits, end_coref_logits, attention_mask, top_lambda=0.4):
    top_k = int(np.sum(attention_mask) * top_lambda)
    candidate_mentions = prune_top_k(mention_logits, top_k)
    mention_to_antecedent = find_antecedent_of_mentions(candidate_mentions, mention_logits, start_coref_logits,
                                                        end_coref_logits)
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

    return [tuple(cluster) for cluster in clusters], candidate_mentions


if __name__ == "__main__":
    a = np.random.random((10, 10))
    b = np.random.random((10, 10))
    c = np.random.random((10, 10))
    mentions = cluster_mentions(a, b, c, [1]*10, 0.4)
    pass
