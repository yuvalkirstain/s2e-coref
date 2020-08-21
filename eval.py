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
from utils import write_examples, EVAL_DATA_FILE_NAME
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.eval_output_dir = args.output_dir
        self.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        self.tokenizer = tokenizer

    def evaluate(self, model, prefix=""):
        if self.args.overwrite_datasets or not os.path.exists(os.path.join(self.args.dataset_output_dir, "eval_dataset.pkl")):
            logger.info(f"Writing eval dataset to {os.path.realpath(os.path.join(self.args.dataset_output_dir, 'eval_dataset.pkl'))}")
            eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, evaluate=True)
            with open(os.path.join(self.args.dataset_output_dir, "eval_dataset.pkl"), "wb") as f:
                pickle.dump(eval_dataset, f)
        logger.info(f"Reading eval dataset from {os.path.realpath(os.path.join(self.args.dataset_output_dir, 'eval_dataset.pkl'))}")
        with open(os.path.join(self.args.dataset_output_dir, "eval_dataset.pkl"), "rb") as f:
            eval_dataset = pickle.load(f)


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

        with open(os.path.join(self.eval_output_dir, EVAL_DATA_FILE_NAME), "wb") as f:
            for batch in eval_dataloader:
                # batch = tuple(tensor.to(self.args.device) for tensor in batch)
                input_ids, attention_mask, start_entity_mentions_indices, end_entity_mentions_indices, start_antecedents_indices, end_antecedents_indices, gold_clusters = batch
                input_ids = input_ids.to(self.args.device)
                attention_mask = attention_mask.to(self.args.device)

                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, return_all_outputs=True)

                write_examples(f, tuple(tensor.cpu().tolist() for tensor in tuple(batch) + outputs[:3]))


        results = {}
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