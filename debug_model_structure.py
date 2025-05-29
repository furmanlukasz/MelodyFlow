import torch
from audiocraft.models import MelodyFlow

# Load the MelodyFlow model
model_name = "facebook/melodyflow-t24-30secs"
print(f"Loading MelodyFlow model: {model_name}")
melody_flow = MelodyFlow.get_pretrained(model_name)
flow_model = melody_flow.lm

# Analyze model structure and parameter names
print(f"\nMelodyFlow model structure:")
total_params = 0
grouped_params = {}

# Group parameters by their prefixes to understand model components
for name, param in flow_model.named_parameters():
    total_params += param.numel()
    
    # Extract the first part of the parameter name
    prefix = name.split('.')[0]
    if prefix not in grouped_params:
        grouped_params[prefix] = []
    grouped_params[prefix].append(name)

# Print model components
print(f"\nModel has {total_params:,} total parameters")
print(f"Main model components:")
for prefix, params in grouped_params.items():
    component_params = sum(flow_model.get_parameter(name).numel() for name in params)
    component_percent = component_params / total_params * 100
    print(f"{prefix}: {len(params)} parameters, {component_params:,} elements ({component_percent:.2f}%)")

# Print all parameter names in a structured way
print("\nDetailed parameter list:")
for name, param in flow_model.named_parameters():
    print(f"{name}: {param.shape}, {param.numel():,} elements")

# Suggest freezing patterns based on analysis
print("\nSuggested freezing patterns based on common model components:")
print("--freeze-patterns \"" + ",".join(grouped_params.keys()) + "\"")

# Generate more specific patterns for partial freezing
blocks = []
attention_components = []
mlp_components = []
conditioning_components = []

for name, _ in flow_model.named_parameters():
    if "block" in name:
        block_name = name.split(".")[1] if "." in name else name
        if block_name not in blocks and "block" in block_name:
            blocks.append(block_name)
    if any(x in name for x in ["attn", "attention", "qkv", "query", "key", "value"]):
        attention_components.append(name.split(".")[0])
    if any(x in name for x in ["mlp", "ffn", "feed_forward"]):
        mlp_components.append(name.split(".")[0])
    if any(x in name for x in ["condition", "text", "embedding", "token", "t5"]):
        conditioning_components.append(name.split(".")[0])

# Make lists unique
blocks = sorted(list(set(blocks)))
attention_components = sorted(list(set(attention_components)))
mlp_components = sorted(list(set(mlp_components)))
conditioning_components = sorted(list(set(conditioning_components)))

print("\nSuggested block freezing patterns:")
print("--freeze-patterns \"" + ",".join(blocks) + "\"")

print("\nSuggested attention component freezing patterns:")
print("--freeze-patterns \"" + ",".join(attention_components) + "\"")

print("\nSuggested MLP component freezing patterns:")
print("--freeze-patterns \"" + ",".join(mlp_components) + "\"")

print("\nSuggested conditioning component freezing patterns:")
print("--freeze-patterns \"" + ",".join(conditioning_components) + "\"") 