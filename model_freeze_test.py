import os

import torch
from audiocraft.models import MelodyFlow

# Load the model
print("Loading MelodyFlow model...")
model = MelodyFlow.get_pretrained("facebook/melodyflow-t24-30secs")
flow_model = model.lm

# Function to check if a parameter name matches any pattern
def matches_pattern(name, patterns):
    if isinstance(patterns, str):
        patterns = patterns.split(',')
    return any(pattern.strip() in name for pattern in patterns)

# Function to simulate freezing parameters
def simulate_freezing(strategy, patterns=None, freeze_ratio=0.5):
    total_params = 0
    frozen_params = 0
    frozen_names = []
    
    # Count all parameters
    for name, param in flow_model.named_parameters():
        total_params += param.numel()
    
    # Apply freezing strategy
    if strategy == 'custom' and patterns:
        for name, param in flow_model.named_parameters():
            if matches_pattern(name, patterns):
                frozen_params += param.numel()
                frozen_names.append(name)
    
    # Print results
    print(f"Strategy: {strategy}")
    if patterns:
        print(f"Patterns: {patterns}")
    
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params*100:.2f}%)")
    
    if len(frozen_names) > 0:
        print("\nFirst 10 frozen parameter names:")
        for name in frozen_names[:10]:
            print(f"  - {name}")
    
    return frozen_params, total_params

# Test various freezing patterns
print("\n=== Testing various freezing strategies ===")

# Original strategy (too small)
print("\n1. Original Patterns:")
simulate_freezing('custom', "condition_provider,conditioners")

# Text embedding related patterns
print("\n2. Text Model Patterns:")
simulate_freezing('custom', "text,embed,token,t5,transformer")

# Broader approach with more components
print("\n3. First 1/3 of model patterns:")
patterns_third = "block_0,block_1,block_2,block_3,block_4,block_5,block_6,block_7,block_8"
simulate_freezing('custom', patterns_third)

# Try different named components
print("\n4. Named Component Patterns:")
named_patterns = "condition,conditioner,text,embed,encoder,decoder,attention,self_attention,q_proj,k_proj,v_proj,t5"
simulate_freezing('custom', named_patterns)

# Try including partial names
print("\n5. Broader Name Patterns:")
broader_patterns = "cond,text,embed,enc,dec,attn,block_0,block_1,block_2,block_3,block_4,block_5"
simulate_freezing('custom', broader_patterns)

# Try experimenting with more aggressive freezing
print("\n6. Full Parameter Name List (first 20):")
for i, (name, _) in enumerate(flow_model.named_parameters()):
    if i < 20:
        print(f"  {name}")
    else:
        remaining = sum(1 for _ in flow_model.named_parameters()) - 20
        print(f"  ... and {remaining} more parameters")
        break 