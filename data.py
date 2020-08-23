import json
import logging
from collections import namedtuple, defaultdict

import torch

from utils import flatten_list_of_lists
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

CorefExample = namedtuple("CorefExample", ["token_ids", "clusters"])

SPEAKER_START = 49518  # 'Ġ#####'
SPEAKER_END = 22560  # 'Ġ###'

# TODO - Should be magic numbers?
PAD_ID_FOR_COREF = -1
NULL_ID_FOR_COREF = 0

logger = logging.getLogger(__name__)


# TODO: bucketization
class CorefDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length=-1):
        self.tokenizer = tokenizer
        examples, self.max_mention_num, self.max_cluster_size, self.max_num_clusters = self._parse_jsonlines(file_path)
        print("Finished parsing file")
        self.max_seq_length = max_seq_length
        self.examples, self.lengths, self.num_examples_filtered = self._tokenize(examples)
        logger.info(
            f"Finished preprocessing Coref dataset. {len(self.examples)} examples were processed, {self.num_examples_filtered} were filtered due to sequence length.")

    def _parse_jsonlines(self, file_path):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                input_words = flatten_list_of_lists(d["sentences"])
                clusters = d["clusters"]
                max_mention_num = max(max_mention_num, len(flatten_list_of_lists(clusters)))
                max_cluster_size = max(max_cluster_size, max(len(cluster) for cluster in clusters) if clusters else 0)
                max_num_clusters = max(max_num_clusters, len(clusters) if clusters else 0)
                speakers = flatten_list_of_lists(d["speakers"])
                examples.append((input_words, clusters, speakers))
        return examples, max_mention_num, max_cluster_size, max_num_clusters

    def _add_speaker_info(self):
        raise NotImplementedError

    def _tokenize(self, examples):
        coref_examples = []
        lengths = []
        num_examples_filtered = 0
        for words, clusters, speakers in examples:
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()

            token_ids = []
            last_speaker = None
            for idx, (word, speaker) in enumerate(zip(words, speakers)):
                # TODO: fix tokenization to deal also with Whitespace
                if last_speaker != speaker:
                    speaker_prefix = [SPEAKER_START] + self.tokenizer.encode(" " + speaker,
                                                                             add_special_tokens=False) + [SPEAKER_END]
                    last_speaker = speaker
                else:
                    speaker_prefix = []
                    token_ids.extend(speaker_prefix)
                word_idx_to_start_token_idx[idx] = len(token_ids) + 1  # +1 for <s>
                tokenized = self.tokenizer.encode(" " + word, add_special_tokens=False)
                token_ids.extend(tokenized)
                word_idx_to_end_token_idx[idx] = len(token_ids)  # old_seq_len + 1 + len(tokenized_word) - 1 (we start counting from zero) = len(token_ids)

            if 0 < self.max_seq_length < len(token_ids):
                num_examples_filtered += 1
                continue
            # bad_ids = [49518, 111, 22560, 3465, 1489, 692, 25586, 12417, 8,
            #            5791, 1489, 692, 23470, 12342, 33, 11908, 307, 128]
            # if not set(bad_ids).issubset(set(token_ids)):
            #     continue
            new_clusters = [
                [(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for
                cluster in clusters]
            lengths.append(len(token_ids))

            coref_examples.append(CorefExample(token_ids=token_ids, clusters=new_clusters))
        return coref_examples, lengths, num_examples_filtered

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def pad_mentions(self, pairs_lst):
        return pairs_lst + [(PAD_ID_FOR_COREF, PAD_ID_FOR_COREF)] * (self.max_mention_num - len(pairs_lst))

    def extract_boundaries_antecedents_indices(self, clusters, max_length):
        start_ids2start_cluster = defaultdict(list)
        end_ids2end_cluster = defaultdict(list)
        for cluster in clusters:
            starts, ends = zip(*cluster)
            self.update_boundry_ids2boundry_cluster(start_ids2start_cluster, starts)
            self.update_boundry_ids2boundry_cluster(end_ids2end_cluster, ends)
        start_antecedent_labels = self.fill_antecedents(start_ids2start_cluster, max_length)
        end_ids2end_cluster = self.fill_antecedents(end_ids2end_cluster, max_length)
        return start_antecedent_labels, end_ids2end_cluster

    def pad_antecedents(self, antecedent_lst):
        antecedent_lst = antecedent_lst if len(antecedent_lst) > 0 else [NULL_ID_FOR_COREF]
        return antecedent_lst + [PAD_ID_FOR_COREF] * (self.max_cluster_size - len(antecedent_lst))

    def update_boundry_ids2boundry_cluster(self, boundry_ids2boundry_cluster, boundries):
        boundries = sorted(boundries)
        for i, boundry in enumerate(boundries):
            boundry_ids2boundry_cluster[boundry] = boundries[:i]

    def fill_antecedents(self, boundry_ids2boundry_cluster, max_length):
        return [self.pad_antecedents(boundry_ids2boundry_cluster[idx]) for idx in range(max_length)]

    def extract_boundries_mention_indices(self, clusters):
        return zip(*self.pad_mentions(flatten_list_of_lists(clusters)))

    def pad_clusters_inside(self, clusters):
        return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (self.max_cluster_size - len(cluster)) for cluster
                in clusters]

    def pad_clusters(self, clusters):
        clusters = self.pad_clusters_outside(clusters)
        clusters = self.pad_clusters_inside(clusters)
        return clusters

    def pad_clusters_outside(self, clusters):
        return clusters + [[]] * (self.max_num_clusters - len(clusters))

    def pad_batch(self, batch, max_length):
        max_length += 2  # we have additional two special tokens <s>, </s>
        padded_batch = []
        for example in batch:
            encoded_dict = self.tokenizer.encode_plus(example[0],
                                                      add_special_tokens=True,
                                                      pad_to_max_length=True,
                                                      max_length=max_length,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')
            mention_antecedent_tensors = self.create_mention_antecedent_tensors(example, max_length)
            example = (encoded_dict["input_ids"], encoded_dict["attention_mask"]) + mention_antecedent_tensors
            padded_batch.append(example)
        tensored_batch = tuple(torch.stack([example[i].squeeze() for example in padded_batch], dim=0) for i in range(7))
        return tensored_batch

    def create_mention_antecedent_tensors(self, example, max_length):
        start_entity_mentions_indices, end_entity_mentions_indices = self.extract_boundries_mention_indices(
            example.clusters)
        start_antecedents_indices, end_antecedents_indices = self.extract_boundaries_antecedents_indices(
            example.clusters, max_length)
        clusters = self.pad_clusters(example.clusters)
        return (torch.tensor(start_entity_mentions_indices),
                torch.tensor(end_entity_mentions_indices),
                torch.tensor(start_antecedents_indices),
                torch.tensor(end_antecedents_indices),
                torch.tensor(clusters))


def get_dataset(args, tokenizer, evaluate=False):
    file_path = args.predict_file if evaluate else args.train_file
    return CorefDataset(file_path, tokenizer, max_seq_length=args.max_seq_length)


if __name__ == "__main__":
    # TODO get max_seq_len and max_mention_num
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dataset = CorefDataset(file_path="data/train.english.jsonlines",
                           tokenizer=tokenizer, max_seq_length=4500)
    print(len(dataset))
    print(dataset.num_examples_filtered)
