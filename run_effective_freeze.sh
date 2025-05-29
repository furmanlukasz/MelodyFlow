#!/bin/bash

# Run MelodyFlow fine-tuning with effective freezing strategy
# This freezes cross-attention layers which connect text understanding with audio generation
# Approximately 24% of parameters will be frozen - enough to prevent catastrophic forgetting
# while still allowing the model to learn your musical style

python train_melodyflow.py \
  --dataset_config dataset/artist-training-dataset.json \
  --epochs 2000 \
  --save_every 150 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --learning_rate 1e-5 \
  --output_dir ./melodyflow_finetuned_effective \
  --cfg_coef 4.0 \
  --test_prompts test_prompts.txt \
  --generate_every 50 \
  --use_wandb \
  --wandb_project "melodyflow-finetuning" \
  --wandb_run_name "melodyflow-effective-freeze" \
  --batch_size 32 \
  --use_mixed_precision \
  --gradient_accumulation_steps 2 \
  --partial_finetuning \
  --freeze_strategy "custom" \
  --freeze_patterns "cross_attention,timestep_embedder" \
  --verbose 