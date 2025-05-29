#!/bin/bash

# Run MelodyFlow fine-tuning with improved EWC implementation
# This script uses EWC to prevent catastrophic forgetting by penalizing changes
# to parameters that are important for the original task

python train_melodyflow_ewc.py \
  --dataset_config dataset/artist-training-dataset.json \
  --epochs 2000 \
  --save_every 150 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --learning_rate 5e-6 \
  --output_dir ./melodyflow_finetuned_ewc_conservative \
  --cfg_coef 4.0 \
  --test_prompts test_prompts.txt \
  --generate_every 10 \
  --use_wandb \
  --wandb_project "melodyflow-finetuning" \
  --wandb_run_name "melodyflow-ewc-conservative" \
  --batch_size 8 \
  --use_mixed_precision \
  --gradient_accumulation_steps 8 \
  --use_ewc \
  --ewc_lambda 75000.0 \
  --ewc_samples 64 \
  --partial_finetuning \
  --freeze_strategy custom \
  --freeze_patterns "cross_attention,timestep_embedder" \
  --verbose 