# Flexible and Adaptable Summarization via Expertise Separation (SIGIR 2024)

## 1. How to Install

### Requirements
- `python3`
- `conda create --name env`
- `pip3 install -r requirements.txt`

### Description of Codes
- `run_mybart.py` - Training and evaluation procedure
- `magic_bart.py` - Main models
- `dataset_maker.py` - Data preprocessing

### Workspace
`./log/seq2seqV4/` will be created for storing model checkpoints and scores.

## 2. How to Run the Code

**For data preprocessing, in the directory `datasets`:**

1. Run `cnndm_from_port.py` to obtain CNN/DM data in JSON format.
2. Run `cnn_wiki_pubmed.py` to mix datasets together.

Or download data from [this link](https://drive.google.com/file/d/1eXyECBTTY4dio3Gx3910TIa1qM-yNolZ/view?usp=sharing).


**Then make trainable data:**

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_mybart.py \
  --model_name_or_path facebook/bart-base \
  --do_train --do_eval \
  --train_file cnndm_wiki_pubmed_train.json \
  --validation_file cnndm_wiki_pubmed_valid.json \
  --test_file cnndm_wiki_pubmed_test.json \
  --output_dir das \
  --exp_name first \
  --max_source_length 1024 \
  --max_target_length 300 \
  --gene_dataset_path cnndm_wiki_pubmed
```

**Finally, train the model:**

```bash
python3 run_mybart.py \
  --model_name_or_path facebook/bart-large \
  --do_train --output_dir das \
  --exp_name train_model \
  --max_source_length 1024 --max_target_length 300 \
  --save_dataset_path cnndm_wiki_pubmed \
  --num_train_epochs 100 \
  --per_device_train_batch_size 8 --save_strategy epoch \
  --label_smoothing_factor 0.1 --weight_decay 0.01 \
  --max_grad_norm 0.1 --warmup_steps 500 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type polynomial --learning_rate 3e-05 \
  --moe_load False \
  --moe_model True --intermediate_size 512 \
  --num_experts 3 --num_datasets 3 --margin_loss True \
  --moe_model_enc True
```

Or it converges faster by training from a pretrained BART_mix summarization model:

```bash
python3 run_mybart.py \
  --model_name_or_path bart-mix \
  --do_train --output_dir das \
  --exp_name train_model \
  --max_source_length 1024 --max_target_length 300 \
  --save_dataset_path cnndm_wiki_pubmed \
  --num_train_epochs 100 \
  --per_device_train_batch_size 8 --save_strategy epoch \
  --label_smoothing_factor 0.1 --weight_decay 0.01 \
  --max_grad_norm 0.1 --warmup_steps 500 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type polynomial --learning_rate 3e-05 \
  --moe_load False \
  --moe_model True --intermediate_size 512 \
  --num_experts 3 --num_datasets 3 --margin_loss True \
  --moe_model_enc True
```

The BART_mix model can be downloaded from [this link](https://drive.google.com/file/d/14PK4Jk3wIYYwLb0OBwAadGR5Yx_JwVSx/view?usp=sharing).


## 3. How to Evaluate
```bash
CUDA_VISIBLE_DEVICES=0 python3 run_mybart.py \
  --per_device_eval_batch_size 16 \
  --log_root ./log \
  --save_dataset_path cnndm \
  --exp_name train_model \
  --do_predict --predict_with_generate True \
  --output_dir das \
  --val_max_target_length 300 \
  --model_name_or_path train_model \
  --moe_model True --intermediate_size 512 \
  --moe_model_enc True \
  --num_experts 3 --num_datasets 3 \
  --margin_loss False --max_val_samples 1000
```
