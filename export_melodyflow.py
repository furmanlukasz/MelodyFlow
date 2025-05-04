#!/usr/bin/env python3
"""
Export script for fine-tuned MelodyFlow models.

This script converts checkpoints from the training process into the official
MelodyFlow format, which consists of:
- state_dict.bin: Contains the flow model weights
- compression_state_dict.bin: Contains the EnCodec model weights
- README.md: Documentation for the model

Usage:
  python export_melodyflow.py --checkpoint path/to/checkpoint.pt --output path/to/output/dir

Options:
  --checkpoint: Path to the checkpoint file (.pt) from fine-tuning
  --output: Directory to save the exported model
  --base_model: Base MelodyFlow model name (default: facebook/melodyflow-t24-30secs)
  --test: Run a test generation after export
  --description: Test prompt for generation test
"""

import argparse
import copy
import os
import shutil
import sys
from pathlib import Path

import torch
import torchaudio


def export_model(checkpoint_path, output_dir, base_model="facebook/melodyflow-t24-30secs", test=False, description=None):
    """Export a fine-tuned MelodyFlow model to the official format.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt) from fine-tuning
        output_dir: Directory to save the exported model
        base_model: Base MelodyFlow model name
        test: Whether to run a test generation after export
        description: Test prompt for generation
    """
    print(f"Exporting model from {checkpoint_path} to {output_dir}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Check if the checkpoint has the expected keys
    if "model_state_dict" not in checkpoint:
        print("Error: Checkpoint doesn't contain 'model_state_dict'")
        sys.exit(1)
    
    # Extract flow model state dict
    flow_model_state_dict = checkpoint["model_state_dict"]
    
    # Load the original pre-trained model to get the compression model
    print(f"Loading base model: {base_model}")
    try:
        from audiocraft.models import MelodyFlow
        original_model = MelodyFlow.get_pretrained(base_model, device="cpu")
        compression_model_state_dict = original_model.compression_model.state_dict()
        
        # Save any config attributes from the original model
        model_config = {}
        if hasattr(original_model.lm, 'xp'):
            model_config['xp'] = copy.deepcopy(original_model.lm.xp)
        
        if model_config:
            print("Saving additional model configuration")
            torch.save(model_config, output_dir / "model_config.bin")
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("Trying to continue with just the flow model weights...")
        compression_model_state_dict = None
    
    # Save the flow model state dict
    flow_state_dict_path = output_dir / "state_dict.bin"
    print(f"Saving flow model state dict to {flow_state_dict_path}")
    torch.save(flow_model_state_dict, flow_state_dict_path)
    
    # Save the compression model state dict if available
    if compression_model_state_dict is not None:
        compression_state_dict_path = output_dir / "compression_state_dict.bin"
        print(f"Saving compression model state dict to {compression_state_dict_path}")
        torch.save(compression_model_state_dict, compression_state_dict_path)
    
    # Create README.md
    readme_path = output_dir / "README.md"
    print(f"Creating README at {readme_path}")
    
    # Extract some info from the checkpoint
    epoch = checkpoint.get("epoch", "unknown")
    loss = checkpoint.get("loss", "unknown")
    
    with open(readme_path, "w") as f:
        f.write("---\nlicense: cc-by-nc-4.0\n---\n")
        f.write(f"# Fine-tuned MelodyFlow Model\n\n")
        f.write(f"This is a fine-tuned version of the {base_model} model.\n\n")
        f.write(f"## Training Information\n\n")
        f.write(f"- Base model: {base_model}\n")
        f.write(f"- Epochs: {epoch}\n")
        f.write(f"- Best validation loss: {loss}\n\n")
        f.write(f"## Usage\n\n")
        f.write("```python\n")
        f.write("import torch\n")
        f.write("from audiocraft.models import MelodyFlow\n")
        f.write("from audiocraft.data.audio import audio_write\n\n")
        f.write("# Load the model\n")
        f.write(f"model = MelodyFlow.get_pretrained('path/to/model')\n\n")
        f.write("# Generate samples\n")
        f.write("descriptions = ['disco beat', 'energetic EDM', 'funky groove']\n")
        f.write("wav = model.generate(descriptions)\n\n")
        f.write("for idx, one_wav in enumerate(wav):\n")
        f.write("    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy=\"loudness\", loudness_compressor=True)\n")
        f.write("```\n")
    
    # Create a .gitattributes file for LFS
    gitattributes_path = output_dir / ".gitattributes"
    with open(gitattributes_path, "w") as f:
        f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    
    print(f"Model successfully exported to {output_dir}")
    
    # Test the exported model
    if test:
        test_model(output_dir, base_model, description)


def test_model(model_path, base_model="facebook/melodyflow-t24-30secs", description=None):
    """Test the exported model by generating a sample.
    
    Args:
        model_path: Path to the exported model
        base_model: Base model name to get configuration
        description: Test prompt for generation
    """
    if description is None:
        description = "energetic electronic music with a catchy melody"
    
    print(f"\nTesting exported model with prompt: '{description}'")
    
    try:
        import torch
        from audiocraft.models import MelodyFlow

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the original model to get configuration
        original_model = MelodyFlow.get_pretrained(base_model, device=device)
        
        # Try a custom loading approach to fix potential config issues
        model_path = Path(model_path)
        
        # Load the compression model
        compression_model = original_model.compression_model
        
        # Load our fine-tuned model state dict
        flow_state_dict = torch.load(model_path / "state_dict.bin", map_location=device)
        
        # Load the model configuration if available
        model_config_path = model_path / "model_config.bin"
        model_config = torch.load(model_config_path) if model_config_path.exists() else {}
        
        # Create a custom model by copying original model structure but using our weights
        custom_model = MelodyFlow(
            name=f"finetuned-melodyflow",
            compression_model=compression_model,
            lm=original_model.lm  # Start with original lm
        )
        
        # Load our weights into the lm
        custom_model.lm.load_state_dict(flow_state_dict)
        
        # Apply saved configuration if available
        if 'xp' in model_config:
            custom_model.lm.xp = model_config['xp']
        
        # Generate sample
        print("Generating audio...")
        wav = custom_model.generate([description], progress=True)
        
        # Save the sample
        from audiocraft.data.audio import audio_write
        output_path = Path(model_path) / "test_generation.wav"
        audio_write(
            str(output_path.with_suffix('')), 
            wav[0].cpu(), 
            custom_model.sample_rate, 
            strategy="loudness", 
            loudness_compressor=True
        )
        
        print(f"Test generation saved to {output_path}")
        print("Test successful!")
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned MelodyFlow model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for exported model")
    parser.add_argument("--base_model", type=str, default="facebook/melodyflow-t24-30secs", help="Base model name")
    parser.add_argument("--test", action="store_true", help="Test the exported model")
    parser.add_argument("--description", type=str, default=None, help="Test prompt for generation")
    
    args = parser.parse_args()
    
    export_model(
        args.checkpoint,
        args.output,
        args.base_model,
        args.test,
        args.description
    )


if __name__ == "__main__":
    main() 