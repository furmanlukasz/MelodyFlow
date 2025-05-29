#!/bin/bash

# Run MelodyFlow fine-tuning with a more comprehensive freezing strategy
# This freezes more components to effectively prevent catastrophic forgetting
python train_melodyflow.py \
  --dataset_config dataset/artist-training-dataset.json \
  --epochs 2000 \
  --save_every 150 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --learning_rate 1e-5 \
  --output_dir ./melodyflow_finetuned_better \
  --cfg_coef 4.0 \
  --test_prompts test_prompts.txt \
  --generate_every 50 \
  --use_wandb \
  --wandb_project "melodyflow-finetuning" \
  --wandb_run_name "melodyflow-better-finetuning" \
  --batch_size 32 \
  --use_mixed_precision \
  --gradient_accumulation_steps 2 \
  --partial_finetuning \
  --freeze_strategy "custom" \
  --freeze_patterns "condition,conditioner,text_model,t5,encoder,query,key,value,qkv,block_0,block_1,block_2,block_3,block_4,block_5,block_6,embeddings" \
  --verbose 