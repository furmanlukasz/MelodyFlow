#!/bin/bash

python train_melodyflow_ewc.py \
  --dataset_config dataset/artist-training-dataset.json \
  --epochs 3000 \
  --save_every 100 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --learning_rate 2e-6 \
  --output_dir ./melodyflow_finetuned_ewc_ultra \
  --cfg_coef 4.0 \
  --test_prompts test_prompts.txt \
  --generate_every 25 \
  --use_wandb \
  --wandb_project "melodyflow-finetuning" \
  --wandb_run_name "melodyflow-ewc-ultra-conservative" \
  --batch_size 4 \
  --use_mixed_precision \
  --gradient_accumulation_steps 16 \
  --use_ewc \
  --ewc_lambda 150000.0 \
  --ewc_samples 128 \
  --partial_finetuning \
  --freeze_strategy custom \
  --freeze_patterns "cross_attention,timestep_embedder,condition" \
  --verbose 