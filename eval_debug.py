# %%

import os
import metrics
from utils import read_examples, EVAL_DATA_FILE_NAME
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer

# %%

OUTPUT_DIR = "output"
eval_data_path = os.path.join(OUTPUT_DIR, EVAL_DATA_FILE_NAME)

# %% md

### Implement here decoding strategies

# %%

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


# %%

def debug_decode(clusters):
    clusters = clusters[:]
    clusters.remove(clusters[0])
    mentions_to_predicted = extract_mentions_to_gold(clusters)
    return clusters, mentions_to_predicted


# %%

decoding_func2name = {debug_decode: "debug_func"}

# %%

coref_evaluator = metrics.CorefEvaluator()


# %%

def extract_mentions_to_gold(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    return mention_to_gold


# %%


# predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
for decoding_func, name in decoding_func2name.items():
    for eval_data_point in read_examples(eval_data_path):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in eval_data_point.gold_clusters.tolist()]
        mention_to_gold_clusters = extract_mentions_to_gold(gold_clusters)

        if "debug" in name:
            predicted_clusters, mention_to_predicted_clusters = decoding_func(gold_clusters)
        else:
            predicted_clusters, mention_to_predicted_clusters = decoding_func(eval_data_point.mention_logits,
                                                                              eval_data_point.start_coref_logits,
                                                                              eval_data_point.end_coref_logits)
        coref_evaluator.update(predicted_clusters,
                               gold_clusters,
                               mention_to_predicted_clusters,
                               mention_to_gold_clusters)
dev_prec, dev_rec, dev_f1 = coref_evaluator.get_prf()
# print("***** Current ckpt path is ***** : {}".format(checkpoint_path))
print("***** EVAL ON DEV SET *****")
print("***** [DEV EVAL] ***** : precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(dev_prec, dev_rec, dev_f1))
# TODO: .logging.info

# %%
