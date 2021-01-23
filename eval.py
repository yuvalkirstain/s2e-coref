import json
import os
import logging
import random
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from coref_bucket_batch_sampler import BucketBatchSampler
from data import get_dataset
from metrics import CorefEvaluator, MentionEvaluator
from utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode
from conll import evaluate_conll

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.eval_output_dir = args.output_dir
        self.tokenizer = tokenizer

    def evaluate(self, model, prefix="", tb_writer=None, global_step=None, official=False):
        eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, evaluate=True)

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = BucketBatchSampler(eval_dataset, max_total_seq_len=self.args.max_total_seq_len, batch_size_1=True)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Examples number: %d", len(eval_dataset))
        model.eval()

        post_pruning_mention_evaluator = MentionEvaluator()
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        losses = defaultdict(list)
        doc_to_prediction = {}
        doc_to_subtoken_map = {}
        for (doc_key, subtoken_maps), batch in eval_dataloader:

            batch = tuple(tensor.to(self.args.device) for tensor in batch)
            input_ids, attention_mask, gold_clusters = batch

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                gold_clusters=gold_clusters,
                                return_all_outputs=True)
                loss_dict = outputs[-1]

            if self.args.n_gpu > 1:
                loss_dict = {key: val.mean() for key, val in loss_dict.items()}

            for key, val in loss_dict.items():
                losses[key].append(val.item())

            outputs = outputs[1:-1]

            batch_np = tuple(tensor.cpu().numpy() for tensor in batch)
            outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)
            for output in zip(*(batch_np + outputs_np)):
                gold_clusters = output[2]
                gold_clusters = extract_clusters(gold_clusters)
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
                gold_mentions = list(mention_to_gold_clusters.keys())

                starts, end_offsets, coref_logits, mention_logits = output[-4:]

                max_antecedents = np.argmax(coref_logits, axis=1).tolist()
                mention_to_antecedent = {((int(start), int(end)), (int(starts[max_antecedent]), int(end_offsets[max_antecedent]))) for start, end, max_antecedent in
                                         zip(starts, end_offsets, max_antecedents) if max_antecedent < len(starts)}

                predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)
                candidate_mentions = list(zip(starts, end_offsets))

                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())
                post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters,
                                       mention_to_gold_clusters)
                doc_to_prediction[doc_key] = predicted_clusters
                doc_to_subtoken_map[doc_key] = subtoken_maps

        post_pruning_mention_precision, post_pruning_mentions_recall, post_pruning_mention_f1 = post_pruning_mention_evaluator.get_prf()
        mention_precision, mentions_recall, mention_f1 = mention_evaluator.get_prf()
        prec, rec, f1 = coref_evaluator.get_prf()

        results = [(key, sum(val) / len(val)) for key, val in losses.items()]
        results += [
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
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, values, global_step)

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

        results = OrderedDict(results)
        results["experiment_name"] = self.args.experiment_name
        results["data"] = prefix
        with open(os.path.join(self.args.output_dir, "results.jsonl"), "a+") as f:
            f.write(json.dumps(results) + '\n')

        if official:
            with open(os.path.join(self.args.output_dir, "preds.jsonl"), "w") as f:
                f.write(json.dumps(doc_to_prediction) + '\n')
                f.write(json.dumps(doc_to_subtoken_map) + '\n')

            if self.args.conll_path_for_eval is not None:
                conll_results = evaluate_conll(self.args.conll_path_for_eval, doc_to_prediction, doc_to_subtoken_map)
                official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
                logger.info('Official avg F1: %.4f' % official_f1)

        return results
