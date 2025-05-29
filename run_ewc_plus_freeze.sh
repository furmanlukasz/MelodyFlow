#!/bin/bash

# Run MelodyFlow fine-tuning with both EWC and cross-attention freezing
# This uses a dual approach to prevent catastrophic forgetting:
# 1. EWC penalizes changes to important parameters from the original model
# 2. Cross-attention layers are frozen to maintain text conditioning capabilities

python train_melodyflow.py \
  --dataset_config dataset/artist-training-dataset.json \
  --epochs 2000 \
  --save_every 150 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --learning_rate 1e-5 \
  --output_dir ./melodyflow_finetuned_ewc_freeze \
  --cfg_coef 4.0 \
  --test_prompts test_prompts.txt \
  --generate_every 50 \
  --use_wandb \
  --wandb_project "melodyflow-finetuning" \
  --wandb_run_name "melodyflow-ewc-freeze" \
  --batch_size 32 \
  --use_mixed_precision \
  --gradient_accumulation_steps 2 \
  --partial_finetuning \
  --freeze_strategy "custom" \
  --freeze_patterns "cross_attention" \
  --use_ewc \
  --ewc_lambda 15000.0 \
  --ewc_samples 32 \
  --verbose 