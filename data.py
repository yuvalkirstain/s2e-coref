import json
from collections import namedtuple
from utils import flatten_list_of_lists
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

CorefExample = namedtuple("CorefExample", ["input_ids", "clusters"])

SPEAKER_START = 49518  # 'Ġ#####'
SPEAKER_END = 22560  # 'Ġ###'


# TODO: bucketization
class CorefDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        examples = self._parse_jsonlines(file_path)
        self.examples = self._tokenize(examples)

    def _parse_jsonlines(self, file_path):
        examples = []
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                input_words = flatten_list_of_lists(d["sentences"])
                clusters = d["clusters"]
                speakers = flatten_list_of_lists(d["speakers"])
                examples.append((input_words, clusters, speakers))
        return examples

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
                word_idx_to_token_idx[idx] = len(token_ids)
                tokenized = self.tokenizer.encode(" " + word, add_special_tokens=False)
                token_ids.extend(tokenized)

            new_clusters = [[(word_idx_to_token_idx[start], word_idx_to_token_idx[end]) for start, end in cluster] for
                            cluster in clusters]
            token_ids = self.tokenizer.encode(token_ids, add_special_tokens=True, pad_to_max_length=True,
                                              max_length=self.max_len)
            coref_examples.append(CorefExample(input_ids=token_ids, clusters=new_clusters))
        return coref_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]

        # TODO: Output tensors:
        # TODO: input_ids and input_mask
        # TODO: For entity mentions: a tensor of size [max_entity_mentions] representing the positive entity mentions
        # TODO: For antecedents: Two tensors of size [seq_length --or-- max_entity_mentions, max_elements_in_cluster]


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    CorefDataset("data/sample.train.english.jsonlines", tokenizer, 1000)
