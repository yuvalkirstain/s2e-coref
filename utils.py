import json
import os
from datetime import datetime
from time import time
import git
import torch

from consts import NULL_ID_FOR_COREF


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
    mention_to_antecedent = sorted(mention_to_antecedent)
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


def mask_tensor(t, mask):
    t = t + ((1.0 - mask.float()) * -10000.0)
    t = torch.clamp(t, min=-10000.0, max=10000.0)
    return t


def write_meta_data(output_dir, args):
    output_path = os.path.join(output_dir, "meta.json")
    repo = git.Repo(search_parent_directories=True)
    hexsha = repo.head.commit.hexsha
    ts = time()
    print(f"Writing {output_path}")
    with open(output_path, mode='w') as f:
        json.dump(
            {
                'git_hexsha': hexsha,
                'args': {k: str(v) for k, v in args.__dict__.items()},
                'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            },
            f,
            indent=4,
            sort_keys=True)
        print(file=f)
