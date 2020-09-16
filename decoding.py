from collections import defaultdict

import numpy as np

from utils import extract_clusters_for_decode

NULL_SPAN_IDX = 0


class Decoder:
    def __init__(self, use_mention_logits_for_antecedents, use_mention_oracle, gold_mentions, gold_clusters,
                 use_crossing_mentions_pruning, only_top_k):
        self.use_mention_logits_for_antecedents = use_mention_logits_for_antecedents
        self.use_mention_oracle = use_mention_oracle
        self.gold_mentions = sorted(gold_mentions)
        self.gold_clusters = gold_clusters
        self.use_crossing_mentions_pruning = use_crossing_mentions_pruning
        self.only_top_k = only_top_k

    def cluster_mentions(self, mention_logits, start_coref_logits, end_coref_logits, attention_mask, top_lambda=0.4):
        top_k = int(np.sum(attention_mask) * top_lambda)
        candidate_mentions = self.prune_top_k(mention_logits, top_k)
        mention_to_antecedent = self.find_mention_to_antecedent(candidate_mentions,
                                                                mention_logits,
                                                                start_coref_logits,
                                                                end_coref_logits)
        return extract_clusters_for_decode(mention_to_antecedent)

    def prune_top_k(self, mention_logits, top_k):
        """
        Gets a numpy array of size [seq_length, seq_length] of logits, and returns the indices of the top_k top-scored spans
        :param mention_logits:
        :param top_k:
        """
        # candidate_mentions_ravel_ids_partition = np.argpartition(-mention_logits.reshape(-1), kth=top_k)
        # # Taking the last k:
        # candidate_mentions_ravel_ids = candidate_mentions_ravel_ids_partition[:top_k]
        # candidate_mentions = sorted(
        #     np.unravel_index(mention, mention_logits.shape) for mention in candidate_mentions_ravel_ids)
        if self.use_crossing_mentions_pruning:
            candidate_starts, candidate_ends = np.unravel_index(np.argsort(-mention_logits, axis=None),
                                                                mention_logits.shape)
            candidate_mentions = list(zip((candidate_starts[:top_k]).tolist(),
                                          (candidate_ends[:top_k]).tolist()))
            candidate_mentions = list(filter(lambda x: mention_logits[x[0], x[1]] > 4, candidate_mentions))
            candidate_mentions = sorted(self.filter_crossing_mentions(candidate_mentions))
        elif self.use_mention_oracle:
            candidate_mentions = sorted(self.gold_mentions)
        elif self.only_top_k:
            candidate_starts, candidate_ends = np.unravel_index(np.argsort(-mention_logits, axis=None),
                                                                mention_logits.shape)
            candidate_mentions = list(zip((candidate_starts[:top_k]).tolist(),
                                          (candidate_ends[:top_k]).tolist()))
        else:
            candidate_starts, candidate_ends = np.where(mention_logits > 4)
            candidate_mentions = sorted(zip(candidate_starts.tolist(), candidate_ends.tolist()))
            if len(candidate_mentions) > top_k:
                candidate_starts, candidate_ends = np.unravel_index(np.argsort(-mention_logits, axis=None),
                                                                    mention_logits.shape)
                candidate_mentions = list(zip((candidate_starts[:top_k]).tolist(),
                                              (candidate_ends[:top_k]).tolist()))
        return candidate_mentions

    def find_mention_to_antecedent(self, candidate_spans, mention_logits, start_coref_logits, end_coref_logits):
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
            if not self.use_mention_logits_for_antecedents:
                mention_logits = np.zeros_like(mention_logits)
            candidate_antecedents_start_indices, candidate_antecedents_end_indices = zip(*candidate_antecedents)

            candidate_antecedent_mention_logits = mention_logits[
                candidate_antecedents_start_indices, candidate_antecedents_end_indices]
            candidate_antecedent_start_logits = start_coref_logits[start_id, candidate_antecedents_start_indices]
            candidate_antecedent_end_logits = end_coref_logits[end_id, candidate_antecedents_end_indices]

            candidate_antecedent_scores = candidate_antecedent_mention_logits + candidate_antecedent_start_logits + candidate_antecedent_end_logits
            # if span_idx > 2:
            #     max_antecedent_scores = candidate_antecedent_scores[np.argpartition(-candidate_antecedent_scores, 2)[:2]]
            #     argmax_antecedent = np.argmax(candidate_antecedent_scores)
            # else:
            max_antecedent_scores = [np.max(candidate_antecedent_scores)]
            argmax_antecedent = np.argmax(candidate_antecedent_scores)
            argmax_antecedent_start, argmax_antecedent_end = candidate_antecedents[argmax_antecedent]

            if np.sum(max_antecedent_scores) > null_antecedent_score:
                mention_to_antecedent.append(((start_id, end_id), (argmax_antecedent_start, argmax_antecedent_end)))

        return mention_to_antecedent

    def filter_crossing_mentions(self, candidate_mentions):
        filtered_candidate_mentions = []
        for i, (start, end) in enumerate(candidate_mentions):
            to_add = True
            for prev_start, prev_end in filtered_candidate_mentions:
                if prev_start < start < prev_end < end or start < prev_start < end < prev_end:
                    to_add = False
                    break
            if to_add:
                filtered_candidate_mentions.append((start, end))
        return filtered_candidate_mentions
