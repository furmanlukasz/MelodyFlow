#!/usr/bin/env python3
"""
Generation script for MelodyFlow models.

This script generates audio samples from exported MelodyFlow models.

Usage:
  python generate_melodyflow.py --model_path path/to/exported/model --description "your prompt here" --output output_folder

Options:
  --model_path: Path to the exported model directory containing state_dict.bin and compression_state_dict.bin
  --description: Text description for generation (prompt)
  --output: Directory to save the generated audio
  --base_model: Base MelodyFlow model name (default: facebook/melodyflow-t24-30secs)
  --num_samples: Number of samples to generate (default: 1)
  --duration: Duration of generated audio in seconds (default: 10.0)
  --steps: Number of generation steps (default: 64)
  --solver: ODE solver to use (euler or midpoint, default: midpoint)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def generate_audio(
    model_path, 
    descriptions, 
    output_dir, 
    base_model="facebook/melodyflow-t24-30secs", 
    duration=10.0, 
    steps=64, 
    solver="midpoint"
):
    """Generate audio samples using a fine-tuned MelodyFlow model.
    
    Args:
        model_path: Path to the exported model directory
        descriptions: List of text descriptions for generation
        output_dir: Directory to save the generated audio
        base_model: Base MelodyFlow model name
        duration: Duration of generated audio in seconds
        steps: Number of generation steps
        solver: ODE solver to use (euler or midpoint)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading model from {model_path}")
    try:
        import torch
        from audiocraft.data.audio import audio_write
        from audiocraft.models import MelodyFlow

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load the original model to get configuration
        original_model = MelodyFlow.get_pretrained(base_model, device=device)
        
        # Convert model_path to Path object
        model_path = Path(model_path)
        
        # Load the compression model
        compression_model = original_model.compression_model
        
        # Load our fine-tuned model state dict
        flow_state_dict = torch.load(model_path / "state_dict.bin", map_location=device)
        
        # Load the model configuration if available
        model_config_path = model_path / "model_config.bin"
        model_config = torch.load(model_config_path) if model_config_path.exists() else {}
        
        # Create a custom model with your fine-tuned weights
        custom_model = MelodyFlow(
            name="finetuned-melodyflow",
            compression_model=compression_model,
            lm=original_model.lm  # Start with original lm structure
        )
        
        # Load your fine-tuned weights
        custom_model.lm.load_state_dict(flow_state_dict)
        
        # Apply saved configuration if available
        if 'xp' in model_config:
            custom_model.lm.xp = model_config['xp']
        
        # Set generation parameters
        custom_model.set_generation_params(
            solver=solver,
            steps=steps,
            duration=duration
        )
        
        # Generate samples
        print(f"Generating {len(descriptions)} sample(s) with prompts:")
        for i, desc in enumerate(descriptions):
            print(f"  {i+1}. '{desc}'")
        
        start_time = time.time()
        wav = custom_model.generate(descriptions, progress=True)
        end_time = time.time()
        
        print(f"Generation completed in {end_time - start_time:.2f} seconds")
        
        # Save the samples
        for idx, (one_wav, desc) in enumerate(zip(wav, descriptions)):
            # Create a filename from the description (limited to 50 chars)
            safe_desc = "".join([c if c.isalnum() else "_" for c in desc])[:50]
            filename = f"{idx+1:02d}_{safe_desc}"
            
            output_path = output_dir / filename
            audio_write(
                str(output_path), 
                one_wav.cpu(), 
                custom_model.sample_rate, 
                strategy="loudness", 
                loudness_compressor=True
            )
            print(f"Saved sample {idx+1} to {output_path}.wav")
        
        print(f"All samples saved to {output_dir}")
        
    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate audio with MelodyFlow")
    parser.add_argument("--model_path", type=str, required=True, help="Path to exported model directory")
    parser.add_argument("--description", type=str, action="append", help="Text description(s) for generation")
    parser.add_argument("--output", type=str, default="generated_samples", help="Output directory for generated audio")
    parser.add_argument("--base_model", type=str, default="facebook/melodyflow-t24-30secs", help="Base model name")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of generated audio in seconds")
    parser.add_argument("--steps", type=int, default=64, help="Number of generation steps")
    parser.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint"], help="ODE solver to use")
    
    args = parser.parse_args()
    
    # If no descriptions provided, use a default one
    if not args.description:
        args.description = ["energetic electronic music with a catchy melody"]
    
    generate_audio(
        args.model_path,
        args.description,
        args.output,
        args.base_model,
        args.duration,
        args.steps,
        args.solver
    )


if __name__ == "__main__":
    main() 


# python generate_melodyflow.py \
#   --model_path melodyflow_finetuned_export \
#   --description "alien speech with reverb and delay" \
#   --description "funky disco beat with synth" \
#   --output generated_samples \
#   --duration 15.0