import json
import os
import logging
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from coref_bucket_batch_sampler import BucketBatchSampler
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def log_batch_eval_results(outputs):
    pass


def train(args, train_dataset, model, tokenizer, evaluator):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader = BucketBatchSampler(train_dataset, max_total_seq_len=args.max_total_seq_len)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    head_params = ['coref', 'mention', 'antecedent']

    model_decay = [p for n, p in model.named_parameters() if
                   not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
    model_no_decay = [p for n, p in model.named_parameters() if
                      not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
    head_decay = [p for n, p in model.named_parameters() if
                  any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
    head_no_decay = [p for n, p in model.named_parameters() if
                     any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]

    head_learning_rate = args.head_learning_rate if args.head_learning_rate else args.learning_rate
    optimizer_grouped_parameters = [
        {'params': model_decay, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': model_no_decay, 'lr': args.learning_rate, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': head_learning_rate, 'weight_decay': args.weight_decay},
        {'params': head_decay, 'lr': head_learning_rate, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    loaded_saved_optimizer = False
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
        loaded_saved_optimizer = True

    if args.amp:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    if os.path.exists(args.model_name_or_path) and 'checkpoint' in args.model_name_or_path:
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from global step %d", global_step)
            if not loaded_saved_optimizer:
                logger.warning("Training is continued from checkpoint, but didn't load optimizer and scheduler")
        except ValueError:
            logger.info("  Starting fine-tuning.")
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    # If nonfreeze_params is not empty, keep all params that are
    # not in nonfreeze_params fixed.
    if args.nonfreeze_params:
        names = []
        for name, param in model.named_parameters():
            freeze = True
            for nonfreeze_p in args.nonfreeze_params.split(','):
                if nonfreeze_p in name:
                    freeze = False

            if freeze:
                param.requires_grad = False
            else:
                names.append(name)

        print('nonfreezing layers: {}'.format(names))

    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproducibility
    set_seed(args)
    best_f1 = -1
    best_global_step = -1
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(tensor.to(args.device) for tensor in batch)
            input_ids, attention_mask, start_entity_mentions_indices, end_entity_mentions_indices, start_antecedents_indices, end_antecedents_indices, gold_clusters = batch
            model.train()

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            start_entity_mention_labels=start_entity_mentions_indices,
                            end_entity_mention_labels=end_entity_mentions_indices,
                            start_antecedent_labels=start_antecedents_indices,
                            end_antecedent_labels=end_antecedents_indices,
                            return_all_outputs=False)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            entity_mention_loss, start_coref_loss, end_coref_loss = outputs[-3:]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if loss > 1000 or str(loss.item()) == 'nan':
                logger.info(f"\nglobal_step: {global_step},"
                            f"loss: {loss}, "
                            f"entity_mention_loss: {entity_mention_loss}, "
                            f"start_coref_loss: {start_coref_loss},"
                            f"end_coref_loss: {end_coref_loss}")
                for example_input_ids in input_ids:
                    logger.info(tokenizer.convert_ids_to_tokens(example_input_ids)[:20])
                log_batch_eval_results(outputs)
                for example_input_ids in input_ids:
                    logger.info(example_input_ids[:20])

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # if args.amp:
                #     scaler.step(optimizer)
                #     scaler.update()
                # else:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info(f"\nloss step {global_step}: {(tr_loss - logging_loss) / args.logging_steps}")
                    logger.info(f"entity_mention_loss step {global_step}: {entity_mention_loss}")
                    logger.info(f"start_coref_loss step {global_step}: {start_coref_loss}")
                    logger.info(f"end_coref_loss step {global_step}: {end_coref_loss}")

                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.do_eval and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    results = evaluator.evaluate(model, prefix=f'step_{global_step}')
                    f1 = results["f1"]
                    if f1 > best_f1:
                        best_f1 = f1
                        best_global_step = global_step
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if 0 < t_total < global_step:
            train_iterator.close()
            break
    if args.results_dir is None:
        args.results_dir = args.output_dir
    with open(os.path.join(args.results_dir, f"{args.experiment_name}_bestf1.json"), "w") as f:
        json.dump({"best_f1": best_f1, "best_global_step": best_global_step}, f)
    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
