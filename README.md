# Coreference Resolution without Span Representations

This repository contains the code implementation from the paper ["Coreference Resolution without Span Representations"](https://arxiv.org/abs/2101.00434).



## Set up

#### Requirements
Set up a virtual environment and run: 
```
pip install -r requirements.txt
```

Follow the [Quick Start](https://github.com/NVIDIA/apex) to enable mixed precision using apex.

#### Download the official evaluation script
Run (from inside the repo):
 
```
git clone https://github.com/conll/reference-coreference-scorers.git
```

#### Prepare the dataset

This repo assumes access to the [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) corpus.
Convert the original dataset into jsonlines format using:
```
python minimize.py <data_dir>
``` 
Credit: This script is originally from the [e2e-coref](https://github.com/kentonl/e2e-coref/) repo.

## Evaluation
Download our trained model from ....? and run:
```
export OUTPUT_DIR=<output_dir>
export CACHE_DIR=<cache_dir>
export MODEL_DIR=<model_dir>

python run_coref.py \
        --output_dir=$OUTPUT_DIR \
        --cache_dir=$CACHE_DIR \
        --model_type=longformer \
        --model_name_or_path=$MODEL_DIR \
        --tokenizer_name=allenai/longformer-large-4096 \
        --config_name=allenai/longformer-large-4096  \
        --train_file=dataset/train.english.jsonlines \
        --predict_file=dataset/test.english.jsonlines \
        --do_eval \
        --num_train_epochs=129 \
        --logging_steps=500 \
        --save_steps=3000 \
        --eval_steps=1000 \
        --max_seq_length=4096 \
        --train_file_cache=dataset/train.english.4096.pkl \
        --predict_file_cache=dataset/test.english.4096.pkl \
        --amp \
        --normalise_loss \
        --max_total_seq_len=5000 \
        --experiment_name=eval_model \
        --warmup_steps=5600 \
        --adam_epsilon=1e-6 \
        --head_learning_rate=3e-4 \
        --learning_rate=1e-5 \
        --adam_beta2=0.98 \
        --weight_decay=0.01 \
        --dropout_prob=0.3 \
        --save_if_best \
        --top_lambda=0.4  \
        --tensorboard_dir=$OUTPUT_DIR/tb
```


## Training
Train a coreference model using:
```
export OUTPUT_DIR=<output_dir>
export CACHE_DIR=<cache_dir>
export MODEL_DIR=<model_dir>

python run_coref.py \
        --output_dir=<output_dir> \
        --cache_dir=<cache_dir> \
        --model_type=longformer \
        --model_name_or_path=allenai/longformer-large-4096 \
        --tokenizer_name=allenai/longformer-large-4096 \
        --config_name=allenai/longformer-large-4096  \
        --train_file=dataset/train.english.jsonlines \
        --predict_file=dataset/dev.english.jsonlines \
        --do_train \
        --do_eval \
        --num_train_epochs=129 \
        --logging_steps=500 \
        --save_steps=3000 \
        --eval_steps=1000 \
        --max_seq_length=4096 \
        --train_file_cache=dataset/train.english.4096.pkl \
        --predict_file_cache=dataset/dev.english.4096.pkl \
        --gradient_accumulation_steps=1 \
        --amp \
        --normalise_loss \
        --max_total_seq_len=5000 \
        --experiment_name="s2e-model" \
        --warmup_steps=5600 \
        --adam_epsilon=1e-6 \
        --head_learning_rate=3e-4 \
        --learning_rate=1e-5 \
        --adam_beta2=0.98 \
        --weight_decay=0.01 \
        --dropout_prob=0.3 \
        --save_if_best \
        --top_lambda=0.4  \
        --tensorboard_dir=$OUTPUT_DIR/tb
```

You can evaluate your trained model on the test set using the evaluation cmd.

## Cite

If you use this code in your research, please cite our paper:

```
@article{kirstain2021coreference,
  title={Coreference Resolution without Span Representations},
  author={Kirstain, Yuval and Ram, Ori and Levy, Omer},
  journal={arXiv preprint arXiv:2101.00434},
  year={2021}
}
```