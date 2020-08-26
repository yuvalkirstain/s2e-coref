import os
import logging
from collections import namedtuple
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from coref_bucket_batch_sampler import BucketBatchSampler
from data import get_dataset
from decoding import cluster_mentions
from metrics import CorefEvaluator, MentionEvaluator
from utils import write_examples, EVAL_DATA_FILE_NAME, EvalDataPoint, extract_clusters, \
    extract_mentions_to_predicted_clusters_from_clusters

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.eval_output_dir = args.output_dir
        self.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        self.tokenizer = tokenizer

    def evaluate(self, model, prefix=""):
        eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, evaluate=True)

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = BucketBatchSampler(eval_dataset, batch_size=self.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Batch size = %d", self.eval_batch_size)
        logger.info("  Examples number: %d", len(eval_dataset))
        model.eval()

        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        for batch in eval_dataloader:
            # batch = tuple(tensor.to(self.args.device) for tensor in batch)
            input_ids, attention_mask, start_entity_mentions_indices, end_entity_mentions_indices, start_antecedents_indices, end_antecedents_indices, gold_clusters = batch
            input_ids = input_ids.to(self.args.device)
            attention_mask = attention_mask.to(self.args.device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, return_all_outputs=True)

            batch_np = tuple(tensor.cpu().numpy() for tensor in batch)
            outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs[:3])
            for output in zip(*(batch_np + outputs_np)):
                data_point = EvalDataPoint(*output)
                gold_clusters = extract_clusters(data_point.gold_clusters)
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
                predicted_clusters = cluster_mentions(data_point.mention_logits,
                                                      data_point.start_coref_logits,
                                                      data_point.end_coref_logits,
                                                      data_point.attention_mask,
                                                      self.args.top_lambda)
                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())
                gold_mentions = list(mention_to_gold_clusters.keys())
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters, mention_to_gold_clusters)

        mention_precision, mentions_recall, mention_f1 = mention_evaluator.get_prf()
        logger.info(f"mention precision: {mention_precision:.4f}, "
                    f"mention recall: {mentions_recall:.4f}, "
                    f"mention f1: {mention_f1:.4f}")
        prec, rec, f1 = coref_evaluator.get_prf()
        logger.info(f"precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

        results = [
            ("mention_precision", mention_precision),
            ("mentions_recall", mentions_recall),
            ("mention_f1", mention_f1),
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