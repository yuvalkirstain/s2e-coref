import argparse

MODEL_TYPES = ['longformer']


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default="longformer",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="allenai/longformer-base-4096",
        type=str,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--train_file_cache",
        default=None,
        type=str,
        required=True,
        help="The output directory where the datasets will be written and read from.",
    )
    parser.add_argument(
        "--predict_file_cache",
        default=None,
        type=str,
        required=True,
        help="The output directory where the datasets will be written and read from.",
    )

    # Other parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default=None, type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument("--tokenizer_name",
                        default=None,
                        type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=-1, type=int)

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    # parser.add_argument(
    #    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    # )
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--nonfreeze_params", default=None, type=str,
                        help="named parameters to update while training (separated by ,). The rest will kept frozen. If None or empty - train all")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--head_learning_rate", default=0.0, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.3, type=float)
    parser.add_argument("--encoder_dropout_prob", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_beta1", default=0.9, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--max_steps",
                        default=-1,
                        type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--verbose_logging",
                        action="store_true",
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_datasets", action="store_true", help="Overwrite the content of the datasets saved"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Whether to use automatic mixed precision instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--antecedent_loss", type=str, required=True, choices=["bce", "allowed"])
    parser.add_argument("--max_span_length", type=int, required=False, default=30)
    parser.add_argument("--seperate_mention_loss", action='store_true')
    parser.add_argument("--top_lambda", type=float, default=0.4)
    parser.add_argument("--num_neighboring_antecedents", type=int, default=0)
    parser.add_argument("--pos_coeff", type=float, default=0.5, help="coefficient to interpolate positive/non_null loss and negative/null loss")

    parser.add_argument("--prune_mention_for_antecedents", action="store_true")
    parser.add_argument("--normalize_antecedent_loss", action='store_true')
    parser.add_argument("--only_joint_mention_logits", action='store_true')
    parser.add_argument("--no_joint_mention_logits", action='store_true')

    parser.add_argument("--use_mention_logits_for_antecedents", action="store_true")
    parser.add_argument("--use_mention_oracle", action="store_true")
    parser.add_argument("--use_crossing_mentions_pruning", action="store_true")

    parser.add_argument("--max_total_seq_len", type=int, default=3500)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)

    parser.add_argument("--end_to_end", action="store_true")
    parser.add_argument("--only_top_k", action="store_true")

    parser.add_argument("--zero_null_logits", action="store_true")

    parser.add_argument("--independent_mention_loss", action="store_true")
    parser.add_argument("--normalise_loss", action="store_true")

    parser.add_argument("--independent_start_end_loss", action="store_true")

    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--sampling_prob", type=float, default=1)

    parser.add_argument("--coarse_to_fine", action="store_true")
    parser.add_argument("--max_c", type=int, default=50)
    parser.add_argument("--ffnn_size", type=int, default=3072)
    parser.add_argument("--apply_attended_reps", action="store_true")
    parser.add_argument("--separate_mention_logits", action="store_true")
    parser.add_argument("--separate_mention_reps", action="store_true")

    parser.add_argument("--save_if_best", action="store_true")

    args = parser.parse_args()
    return args
