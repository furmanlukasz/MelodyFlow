#!/bin/bash

# Install compatible versions of required packages
echo "Installing compatible versions of required packages..."

# Downgrade transformers to a more stable version
pip install transformers==4.30.0

# Install other necessary packages if needed
pip install wandb

# Run the Python fix script
echo "Running fix script for transformers library..."
python fix_imports.py

echo "Setup complete. You should now be able to run train_melodyflow.py" 