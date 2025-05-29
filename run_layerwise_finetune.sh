#!/bin/bash

# Run MelodyFlow fine-tuning with layerwise partial fine-tuning
# This freezes the early layers of the model to preserve basic capabilities
python train_melodyflow.py \
  --dataset_config dataset/artist-training-dataset.json \
  --epochs 2000 \
  --save_every 150 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --learning_rate 1e-5 \
  --output_dir ./melodyflow_finetuned_layerwise \
  --cfg_coef 4.0 \
  --test_prompts test_prompts.txt \
  --generate_every 50 \
  --use_wandb \
  --wandb_project "melodyflow-finetuning" \
  --wandb_run_name "melodyflow-layerwise-finetuning" \
  --batch_size 32 \
  --use_mixed_precision \
  --gradient_accumulation_steps 2 \
  --partial_finetuning \
  --freeze_strategy "layerwise" \
  --freeze_ratio 0.3 \
  --verbose 