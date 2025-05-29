#!/bin/bash

# EWC training without mixed precision for maximum stability
# Mixed precision can interfere with EWC's delicate gradient calculations

python train_melodyflow_ewc.py \
  --dataset_config dataset/artist-training-dataset.json \
  --epochs 2000 \
  --save_every 100 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --learning_rate 3e-6 \
  --output_dir ./melodyflow_finetuned_ewc_stable \
  --cfg_coef 4.0 \
  --test_prompts test_prompts.txt \
  --generate_every 10 \
  --use_wandb \
  --wandb_project "melodyflow-finetuning" \
  --wandb_run_name "melodyflow-ewc-stable-fp32" \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --use_ewc \
  --ewc_lambda 100000.0 \
  --ewc_samples 64 \
  --partial_finetuning \
  --freeze_strategy custom \
  --freeze_patterns "cross_attention,timestep_embedder" \
  --verbose 