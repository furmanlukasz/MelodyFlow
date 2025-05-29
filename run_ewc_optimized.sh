#!/bin/bash

# EWC training with mixed precision and regularization optimizations
# No partial fine-tuning - just EWC + dropout + weight decay

python train_melodyflow_ewc.py \
  --dataset_config dataset/artist-training-dataset.json \
  --epochs 2000 \
  --save_every 150 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-6 \
  --output_dir ./melodyflow_finetuned_ewc_optimized \
  --cfg_coef 4.0 \
  --test_prompts test_prompts.txt \
  --generate_every 10 \
  --use_wandb \
  --wandb_project "melodyflow-finetuning" \
  --wandb_run_name "melodyflow-ewc-optimized" \
  --batch_size 32 \
  --use_mixed_precision \
  --gradient_accumulation_steps 2 \
  --use_ewc \
  --ewc_lambda 75000.0 \
  --ewc_samples 64 \
  --verbose 