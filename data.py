import json
from collections import namedtuple, defaultdict

import torch

from utils import flatten_list_of_lists
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

CorefExample = namedtuple("CorefExample", ["input_ids", "attention_mask", "clusters"])

SPEAKER_START = 49518  # 'Ġ#####'
SPEAKER_END = 22560  # 'Ġ###'

# TODO - Should be magic numbers?
PAD_ID_FOR_COREF = -1
NULL_ID_FOR_COREF = 0


# TODO: bucketization
class CorefDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        examples, self.max_mention_num, self.max_cluster_size = self._parse_jsonlines(file_path)
        self.examples = self._tokenize(examples)

    def _parse_jsonlines(self, file_path):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                input_words = flatten_list_of_lists(d["sentences"])
                clusters = d["clusters"]
                max_mention_num = max(max_mention_num, len(flatten_list_of_lists(clusters)))
                max_cluster_size = max(max_cluster_size, max(len(cluster) for cluster in clusters))
                speakers = flatten_list_of_lists(d["speakers"])
                examples.append((input_words, clusters, speakers))
        return examples, max_mention_num, max_cluster_size

    def _add_speaker_info(self):
        raise NotImplementedError

    def _tokenize(self, examples):
        coref_examples = []
        for words, clusters, speakers in examples:
            word_idx_to_token_idx = dict()
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
                word_idx_to_token_idx[idx] = len(token_ids) + 1  # +1 for <s>
                tokenized = self.tokenizer.encode(" " + word, add_special_tokens=False)
                token_ids.extend(tokenized)

            new_clusters = [[(word_idx_to_token_idx[start], word_idx_to_token_idx[end]) for start, end in cluster] for
                            cluster in clusters]
            encoded_dict = self.tokenizer.encode_plus(token_ids, add_special_tokens=True, pad_to_max_length=True,
                                                      max_length=self.max_seq_len, return_attention_mask=True)
            coref_examples.append(
                CorefExample(input_ids=encoded_dict['input_ids'], attention_mask=encoded_dict['attention_mask'],
                             clusters=new_clusters))
        return coref_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        start_entity_mentions_indices, end_entity_mentions_indices = self.extract_boundries_mention_indices(
            example.clusters)
        start_antecedents_indices, end_antecedents_indices = self.extract_boundaries_antecedents_indices(
            example.clusters)
        outputs = (torch.tensor(example.input_ids), torch.tensor(example.attention_mask), torch.tensor(
            start_entity_mentions_indices), torch.tensor(end_entity_mentions_indices), torch.tensor(
            start_antecedents_indices), torch.tensor(end_antecedents_indices), torch.tensor(example.clusters))
        return outputs

    def pad_mentions(self, pairs_lst):
        return pairs_lst + [(PAD_ID_FOR_COREF, PAD_ID_FOR_COREF)] * (self.max_mention_num - len(pairs_lst))

    def extract_boundaries_antecedents_indices(self, clusters):
        start_ids2start_cluster = defaultdict(list)
        end_ids2end_cluster = defaultdict(list)
        for cluster in clusters:
            starts, ends = zip(*cluster)
            self.update_boundry_ids2boundry_cluster(start_ids2start_cluster, starts)
            self.update_boundry_ids2boundry_cluster(end_ids2end_cluster, ends)
        start_antecedent_labels = self.fill_antecedents(start_ids2start_cluster)
        end_ids2end_cluster = self.fill_antecedents(end_ids2end_cluster)
        return start_antecedent_labels, end_ids2end_cluster

    def pad_antecedents(self, antecedent_lst):
        antecedent_lst = antecedent_lst if len(antecedent_lst) > 0 else [NULL_ID_FOR_COREF]
        return antecedent_lst + [PAD_ID_FOR_COREF] * (self.max_cluster_size - len(antecedent_lst))

    def update_boundry_ids2boundry_cluster(self, boundry_ids2boundry_cluster, boundries):
        boundries = sorted(boundries)
        for i, boundry in enumerate(boundries):
            boundry_ids2boundry_cluster[boundry] = boundries[:i]

    def fill_antecedents(self, boundry_ids2boundry_cluster):
        return [self.pad_antecedents(boundry_ids2boundry_cluster[idx]) for idx in range(self.max_seq_len)]

    def extract_boundries_mention_indices(self, clusters):
        return zip(*self.pad_mentions(flatten_list_of_lists(clusters)))


def get_dataset(args, tokenizer, evaluate=False):
    file_path = args.predict_file if evaluate else args.train_file
    return CorefDataset(file_path, tokenizer, args.max_seq_length)


if __name__ == "__main__":
    # TODO get max_seq_len and max_mention_num
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dataset = CorefDataset(file_path="data/sample.train.english.jsonlines",
                           tokenizer=tokenizer,
                           max_seq_len=1000)
    example = dataset[0]
