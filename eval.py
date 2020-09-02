import os
import logging
import random
from collections import namedtuple
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from coref_bucket_batch_sampler import BucketBatchSampler
from data import get_dataset
from decoding import Decoder
from metrics import CorefEvaluator, MentionEvaluator
from utils import write_examples, EVAL_DATA_FILE_NAME, EvalDataPoint, extract_clusters, \
    extract_mentions_to_predicted_clusters_from_clusters

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, tokenizer, sampling_prob=1.0):
        self.args = args
        self.eval_output_dir = args.output_dir
        self.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        self.tokenizer = tokenizer
        self.sampling_prob = sampling_prob

    def evaluate(self, model, prefix=""):
        eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, evaluate=True)

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = BucketBatchSampler(eval_dataset, max_total_seq_len=self.args.max_total_seq_len)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Batch size = %d", self.eval_batch_size)
        logger.info("  Examples number: %d", len(eval_dataset))
        model.eval()

        post_pruning_mention_evaluator = MentionEvaluator()
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        losses = {"loss": [], "entity_mention_loss": [], "start_coref_loss": [], "end_coref_loss": []}
        for batch in eval_dataloader:
            if random.random() > self.sampling_prob:
                continue

            batch = tuple(tensor.to(self.args.device) for tensor in batch)
            input_ids, attention_mask, start_entity_mentions_indices, end_entity_mentions_indices, start_antecedents_indices, end_antecedents_indices, gold_clusters = batch

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                start_entity_mention_labels=start_entity_mentions_indices,
                                end_entity_mention_labels=end_entity_mentions_indices,
                                start_antecedent_labels=start_antecedents_indices,
                                end_antecedent_labels=end_antecedents_indices,
                                return_all_outputs=True)
                loss = outputs[0]
                entity_mention_loss, start_coref_loss, end_coref_loss = outputs[-3:]

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                entity_mention_loss, start_coref_loss, end_coref_loss = entity_mention_loss.mean(), start_coref_loss.mean(), end_coref_loss.mean()

            losses["loss"].append(loss)
            losses["entity_mention_loss"].append(entity_mention_loss)
            losses["start_coref_loss"].append(start_coref_loss)
            losses["end_coref_loss"].append(end_coref_loss)

            outputs = outputs[1:-3]

            batch_np = tuple(tensor.cpu().numpy() for tensor in batch)
            outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs[:3])
            for output in zip(*(batch_np + outputs_np)):
                data_point = EvalDataPoint(*output)

                gold_clusters = extract_clusters(data_point.gold_clusters)
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
                gold_mentions = list(mention_to_gold_clusters.keys())

                decoder = Decoder(use_mention_logits_for_antecedents=self.args.use_mention_logits_for_antecedents,
                                  use_mention_oracle=self.args.use_mention_oracle,
                                  gold_mentions=gold_mentions,
                                  gold_clusters=gold_clusters,
                                  use_crossing_mentions_pruning=self.args.use_crossing_mentions_pruning)

                predicted_clusters, candidate_mentions = decoder.cluster_mentions(data_point.mention_logits,
                                                                                  data_point.start_coref_logits,
                                                                                  data_point.end_coref_logits,
                                                                                  data_point.attention_mask,
                                                                                  self.args.top_lambda)

                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())

                post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters,
                                       mention_to_gold_clusters)

        post_pruning_mention_precision, post_pruning_mentions_recall, post_pruning_mention_f1 = post_pruning_mention_evaluator.get_prf()
        mention_precision, mentions_recall, mention_f1 = mention_evaluator.get_prf()
        prec, rec, f1 = coref_evaluator.get_prf()

        results = [
            ("eval loss", sum(losses["loss"]) / len(losses["loss"])),
            ("eval entity_mention_loss", sum(losses["entity_mention_loss"]) / len(losses["entity_mention_loss"])),
            ("eval start_coref_loss", sum(losses["start_coref_loss"]) / len(losses["start_coref_loss"])),
            ("eval end_coref_loss", sum(losses["end_coref_loss"]) / len(losses["end_coref_loss"])),
            ("post pruning mention precision", post_pruning_mention_precision),
            ("post pruning mention recall", post_pruning_mentions_recall),
            ("post pruning mention f1", post_pruning_mention_f1),
            ("mention precision", mention_precision),
            ("mention recall", mentions_recall),
            ("mention f1", mention_f1),
            ("precision", prec),
            ("recall", rec),
            ("f1", f1)
        ]
        logger.info("***** Eval results {} *****".format(prefix))
        for key, values in results:
            if isinstance(values, float):
                logger.info(f"  {key} = {values:.3f}")
            else:
                logger.info(f"  {key} = {values}")

        if self.eval_output_dir:
            output_eval_file = os.path.join(self.eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                if prefix:
                    writer.write(f'\n{prefix}:\n')
                for key, values in results:
                    if isinstance(values, float):
                        writer.write(f"{key} = {values:.3f}\n")
                    else:
                        writer.write(f"{key} = {values}\n")

        return results
