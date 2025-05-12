import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from audiocraft.data.audio import audio_write
from audiocraft.models import MelodyFlow
from audiocraft.models.encodec import CompressionModel
from audiocraft.modules.conditioners import ConditioningAttributes
from dataset_loader.dataset import create_dataloader_from_config
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader


# Custom implementation of batch_linear_assignment using scipy
def batch_linear_assignment(cost):
    """Solve a batch of linear assignment problems.
    
    This is a CPU fallback implementation using scipy's linear_sum_assignment.
    
    Args:
      cost: Cost matrix with shape (B, W, T), where W is the number of workers
            and T is the number of tasks.
            
    Returns:
      Matching tensor with shape (B, W), with assignments for each worker. If the
      task was not assigned, the corresponding index will be -1.
    """
    if cost.ndim != 3:
        raise ValueError("Need 3-dimensional tensor with shape (B, W, T).")
    
    # Move to CPU and convert to numpy
    cost_np = cost.detach().cpu().numpy()
    b, w, t = cost_np.shape
    
    # Initialize result tensor
    matching = torch.full([b, w], -1, dtype=torch.long, device=cost.device)
    
    # Process each batch element
    for i in range(b):
        # Use scipy's linear_sum_assignment
        workers, tasks = linear_sum_assignment(cost_np[i])
        
        # Convert back to torch tensors
        workers_torch = torch.from_numpy(workers).to(cost.device)
        tasks_torch = torch.from_numpy(tasks).to(cost.device)
        
        # Update matching tensor
        matching[i].scatter_(0, workers_torch, tasks_torch)
    
    return matching


class MelodyFlowTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Load dataset configuration
        with open(args.dataset_config, 'r') as f:
            self.dataset_config = json.load(f)
        
        # Load model
        print("Loading MelodyFlow model...")
        self.melody_flow = MelodyFlow.get_pretrained(args.model_name, device=self.device)
        self.encodec = self.melody_flow.compression_model
        self.flow_model = self.melody_flow.lm
        
        # Store dropout rate for use during training
        self.dropout_rate = args.dropout
        if args.dropout > 0:
            print(f"Using dropout rate {args.dropout} during training")
            # We don't modify the model structure here as the expected parameters are floats
            # The dropout will be applied during the forward pass in scaled_dot_product_attention
        
        # Get conditioning provider from the flow model
        self.condition_provider = self.flow_model.condition_provider
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.flow_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        # Create dataloaders
        print("Setting up dataloaders...")
        self.setup_dataloaders()
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize best loss for model saving
        self.best_val_loss = float('inf')
        
    def setup_dataloaders(self):
        """Set up training and validation dataloaders"""
        self.train_loader = create_dataloader_from_config(
            self.dataset_config,
            batch_size=self.args.batch_size,
            sample_size=self.args.sample_size,
            sample_rate=self.encodec.sample_rate,
            audio_channels=2,  # MelodyFlow uses stereo
            num_workers=self.args.num_workers
        )
        
        # Load validation dataset if provided
        if self.args.validation_dataset_config:
            with open(self.args.validation_dataset_config, 'r') as f:
                val_config = json.load(f)
                
            self.val_loader = create_dataloader_from_config(
                val_config,
                batch_size=self.args.batch_size,
                sample_size=self.args.sample_size,
                sample_rate=self.encodec.sample_rate,
                audio_channels=2,
                num_workers=self.args.num_workers
            )
        else:
            self.val_loader = None
    
    def encode_audio_to_latent(self, audio):
        """Encode audio to latent space using EnCodec's VAE structure"""
        with torch.no_grad():
            # Get the original latent
            latent = self.encodec.encode(audio)[0]
            latent = latent.squeeze(1)
            
            # Print latent shape for debugging (first time only)
            if not hasattr(self, '_printed_latent_shape'):
                print(f"Debug - Original latent shape: {latent.shape}")
                print(f"Debug - Flow model latent dim: {self.flow_model.latent_dim}")
                self._printed_latent_shape = True
            
            # Handle VAE structure - EnCodec produces [mean, scale] concatenated on dim=1
            if latent.shape[1] == 256 and self.flow_model.latent_dim == 128:
                # Split 256-dimensional latent into mean and scale components
                mean, scale = latent.chunk(2, dim=1)
                
                # Use only the mean component - this matches the model's expectations
                # This preserves the core representation without the stochastic element
                if not hasattr(self, '_shown_vae_notice'):
                    print("Using VAE mean component from EnCodec (discarding scale)")
                    self._shown_vae_notice = True
                
                return mean
            
            # If dimensions already match or for other dimension combinations, warn and truncate
            if latent.shape[1] != self.flow_model.latent_dim:
                if not hasattr(self, '_shown_dimension_warning'):
                    print(f"Warning: Unexpected latent dimensions - got {latent.shape[1]}, expected {self.flow_model.latent_dim}")
                    print(f"Truncating to expected dimension - this may affect quality")
                    self._shown_dimension_warning = True
                
                # Truncate to expected dimension as fallback
                latent = latent[:, :self.flow_model.latent_dim, :]
            
            return latent
    
    def prepare_text_conditioning(self, prompts):
        """Simplified CFG preparation - fixed token access"""
        # Create conditional and unconditional prompts
        cond_attributes = [ConditioningAttributes(text={'description': p}) for p in prompts]
        uncond_attributes = [ConditioningAttributes(text={'description': ""}) for _ in prompts]
        
        # Tokenize
        cond_tokens = self.condition_provider.tokenize(cond_attributes)
        uncond_tokens = self.condition_provider.tokenize(uncond_attributes)
        
        # Get conditioning tensors - access directly without attempting shape manipulation
        cond_tensors = self.condition_provider(cond_tokens)
        uncond_tensors = self.condition_provider(uncond_tokens)
        
        # Print debug info once to understand structure
        if not hasattr(self, '_printed_tensor_debug'):
            print(f"Debug - Cond tensor keys: {list(cond_tensors.keys())}")
            print(f"Debug - Cond tensor description type: {type(cond_tensors['description'])}")
            if isinstance(cond_tensors['description'], tuple):
                print(f"Debug - Cond tensor tuple len: {len(cond_tensors['description'])}")
                print(f"Debug - Cond[0] shape: {cond_tensors['description'][0].shape}")
                print(f"Debug - Cond[1] shape: {cond_tensors['description'][1].shape}")
            self._printed_tensor_debug = True
        
        # Return the tensors separately - no manipulation needed
        return {
            'conditional': cond_tensors['description'],
            'unconditional': uncond_tensors['description']
        }
    
    def compute_pairwise_distances(self, x, n):
        """Compute pairwise L2 distances between all pairs of x and n samples"""
        # x, n shape: [B, C, T]
        # Reshape to [B, C*T]
        x_flat = x.reshape(x.shape[0], -1)
        n_flat = n.reshape(n.shape[0], -1)
        
        # Compute squared L2 distance between all pairs
        # Output shape: [1, B, B]
        x_norm = (x_flat**2).sum(1, keepdim=True)
        n_norm = (n_flat**2).sum(1, keepdim=True)
        
        # [(x-y)^2 = x^2 + y^2 - 2xy] for L2 distance
        # Reshape to match expected input for batch_linear_assignment
        distances = x_norm + n_norm.transpose(0, 1) - 2 * torch.matmul(x_flat, n_flat.transpose(0, 1))
        distances = distances.unsqueeze(0)  # Add batch dimension for torch-linear-assignment
        
        return distances.to(x.device)
    
    def align_latents(self, x, n):
        """Align noise samples n to match with true latents x using linear assignment"""
        # Compute pairwise distances
        distances = self.compute_pairwise_distances(x, n)
        
        # Solve linear assignment problem
        assignment = batch_linear_assignment(distances)
        
        # Reorder noise samples according to assignment
        n_aligned = torch.zeros_like(n)
        for i, j in enumerate(assignment[0]):
            if j >= 0:  # Skip unassigned samples
                n_aligned[i] = n[j]
        
        return n_aligned
    
    def train_epoch(self, epoch):
        """Train for one epoch with separate forward passes for CFG"""
        self.flow_model.train()
        total_loss = 0
        
        progress_bar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (audio, info) in enumerate(progress_bar):
            audio = audio.to(self.device)
            prompts = [item.get('prompt', '') for item in info]
            
            # Encode audio to latent space
            x = self.encode_audio_to_latent(audio)
            
            # Generate noise and align
            n = torch.randn_like(x)
            n = self.align_latents(x, n)
            
            # Sample flow step t
            t = torch.sigmoid(torch.randn(x.shape[0], 1, device=self.device))
            
            # Create training input
            z = x * t.unsqueeze(-1) + (1 - t.unsqueeze(-1)) * n + 1e-5 * torch.randn_like(x)
            
            # Get conditioning - note we'll use separate forward passes
            cond_tensors = self.prepare_text_conditioning(prompts)
            
            # Apply classifier-free guidance 
            cfg_coef = self.args.cfg_coef if epoch >= self.args.cfg_free_epochs else 0.0
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Run conditional forward pass
            cond_src, cond_mask = cond_tensors['conditional']
            pred_cond = self.flow_model(
                z, 
                t, 
                cond_src, 
                torch.log(cond_mask.unsqueeze(1).unsqueeze(1))
            )
            
            # Run unconditional forward pass 
            if cfg_coef > 0:
                uncond_src, uncond_mask = cond_tensors['unconditional']
                pred_uncond = self.flow_model(
                    z, 
                    t, 
                    uncond_src, 
                    torch.log(uncond_mask.unsqueeze(1).unsqueeze(1))
                )
                # Apply CFG
                pred_velocity = pred_cond + cfg_coef * (pred_cond - pred_uncond)
            else:
                # Skip CFG if coefficient is zero
                pred_velocity = pred_cond
            
            # Target is x - n
            target = x - n
            
            # Compute loss and backprop
            loss = F.mse_loss(pred_velocity, target)
            loss.backward()
            
            # Clip gradients and update
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.args.clip_grad)
            self.optimizer.step()
            
            # Update progress tracking
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))
            
            # Debug output
            if batch_idx < 2 and epoch == 0:
                print(f"Prompt example: {prompts[0][:50]}...")
                print(f"Pred velocity shape: {pred_velocity.shape}")
                print(f"Target shape: {target.shape}")
                print(f"Loss: {loss.item():.6f}")
                print(f"CFG coefficient: {cfg_coef}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate using separate conditional/unconditional passes"""
        if self.val_loader is None:
            return 0.0
        
        self.flow_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for audio, info in self.val_loader:
                audio = audio.to(self.device)
                prompts = [item.get('prompt', '') for item in info]
                
                # Core processing
                x = self.encode_audio_to_latent(audio)
                n = torch.randn_like(x)
                n = self.align_latents(x, n)
                t = torch.sigmoid(torch.randn(x.shape[0], 1, device=self.device))
                z = x * t.unsqueeze(-1) + (1 - t.unsqueeze(-1)) * n + 1e-5 * torch.randn_like(x)
                
                # Get text conditioning
                cond_tensors = self.prepare_text_conditioning(prompts)
                
                # Run conditional pass
                cond_src, cond_mask = cond_tensors['conditional']
                pred_cond = self.flow_model(
                    z, t, cond_src, torch.log(cond_mask.unsqueeze(1).unsqueeze(1))
                )
                
                # Run unconditional pass
                uncond_src, uncond_mask = cond_tensors['unconditional'] 
                pred_uncond = self.flow_model(
                    z, t, uncond_src, torch.log(uncond_mask.unsqueeze(1).unsqueeze(1))
                )
                
                # Apply CFG
                pred_velocity = pred_cond + self.args.cfg_coef * (pred_cond - pred_uncond)
                
                # Compute loss
                target = x - n
                loss = F.mse_loss(pred_velocity, target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def generate_test_samples(self, epoch, baseline=False):
        """Generate test samples using the current model at the specified epoch.
        
        Args:
            epoch: Current epoch number
            baseline: If True, this is generating baseline samples before training
        """
        if not self.args.test_prompts:
            print("No test prompts provided, skipping sample generation.")
            return
        
        # Skip generation based on frequency, but always generate for baseline or final epoch
        if not baseline and epoch != self.args.epochs - 1:
            if not self.args.generate_every or epoch % self.args.generate_every != 0:
                return
        
        # Choose appropriate directory name
        if baseline:
            dir_name = "samples_baseline"
            print(f"Starting baseline sample generation...")
        else:
            dir_name = f"samples_epoch_{epoch+1}"
            print(f"Starting sample generation for epoch {epoch+1}...")
        
        # Create output directory for this epoch's samples
        output_dir = self.output_dir / dir_name
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Sample directory created at: {output_dir}")
        
        # Set model to eval mode
        self.flow_model.eval()
        
        # Set generation parameters - using high quality settings for evaluation
        print(f"Using {self.args.eval_solver} solver with {self.args.eval_steps} steps for high-quality generation")
        try:
            self.melody_flow.set_generation_params(
                solver=self.args.eval_solver,
                steps=self.args.eval_steps,
                duration=self.args.eval_duration
            )
            
            # Process each prompt
            all_samples = []
            sample_info = []
            current_time = 0.0
            
            for idx, prompt in enumerate(self.test_prompts):
                try:
                    # Generate sample
                    with torch.no_grad():
                        start_time = time.time()
                        wav = self.melody_flow.generate([prompt], progress=False)
                        generation_time = time.time() - start_time
                    
                    # Save the individual sample
                    safe_prompt = f"prompt_{idx+1}"
                    output_path = output_dir / safe_prompt
                    audio_write(
                        str(output_path), 
                        wav[0].cpu(), 
                        self.melody_flow.sample_rate, 
                        strategy="loudness", 
                        loudness_compressor=True
                    )
                    
                    # Save prompt text
                    with open(output_dir / f"{safe_prompt}.txt", "w") as f:
                        f.write(prompt)
                    
                    # Add to the concatenated list
                    all_samples.append(wav[0].cpu())
                    
                    # Track sample info for the concatenated file
                    duration = wav[0].shape[-1] / self.melody_flow.sample_rate
                    sample_info.append({
                        "prompt": prompt,
                        "start_time": current_time,
                        "end_time": current_time + duration,
                        "duration": duration
                    })
                    current_time += duration
                    
                    # Add 1 second silence between samples (if not the last sample)
                    if idx < len(self.test_prompts) - 1:
                        silence = torch.zeros(2, self.melody_flow.sample_rate)  # 1 second of silence
                        all_samples.append(silence)
                        current_time += 1.0  # 1 second
                    
                    print(f"  Generated sample {idx+1}/{len(self.test_prompts)} in {generation_time:.2f}s")
                except Exception as e:
                    print(f"  Error generating sample for prompt {idx+1}: {e}")
                    traceback.print_exc()
            
            # Create concatenated sample if we have any samples
            if all_samples:
                concat_path = output_dir / "all_samples_concatenated"
                concatenated = torch.cat(all_samples, dim=1)
                
                audio_write(
                    str(concat_path),
                    concatenated,
                    self.melody_flow.sample_rate,
                    strategy="loudness",
                    loudness_compressor=True
                )
                
                # Create a text file with sample timestamps
                with open(output_dir / "concatenated_timestamps.txt", "w") as f:
                    f.write(f"Concatenated samples for {'baseline' if baseline else f'epoch {epoch+1}'}\n\n")
                    for idx, info in enumerate(sample_info):
                        f.write(f"Sample {idx+1}:\n")
                        f.write(f"  Start time: {info['start_time']:.2f}s\n")
                        f.write(f"  End time: {info['end_time']:.2f}s\n")
                        f.write(f"  Duration: {info['duration']:.2f}s\n")
                        f.write(f"  Prompt: {info['prompt']}\n\n")
                
                print(f"  Concatenated sample saved to {concat_path}.wav (total duration: {current_time:.2f}s)")
            
            print(f"Samples saved to {output_dir}")
            
            # Set model back to train mode
            self.flow_model.train()
        except Exception as e:
            print(f"Error setting up generation parameters: {e}")
            traceback.print_exc()
    
    def train(self):
        """Train the MelodyFlow model"""
        print(f"Starting training for {self.args.epochs} epochs...")
        print(f"Files will be saved to: {self.output_dir}")
        
        # Track epochs without improvement for early stopping
        epochs_without_improvement = 0
        
        # Load test prompts if specified
        self.test_prompts = []
        if self.args.test_prompts:
            try:
                # Check if it's a file path
                if os.path.isfile(self.args.test_prompts):
                    with open(self.args.test_prompts, 'r') as f:
                        self.test_prompts = [line.strip() for line in f if line.strip()]
                else:
                    # Assume it's a comma-separated list of prompts
                    self.test_prompts = [p.strip() for p in self.args.test_prompts.split(',') if p.strip()]
                
                print(f"Loaded {len(self.test_prompts)} test prompts for generation")
            except Exception as e:
                print(f"Error loading test prompts: {e}")
                self.test_prompts = []
        
        # Generate baseline samples with the initial model before training
        if self.test_prompts:
            try:
                print("\nGenerating baseline samples with the initial model...")
                self.generate_test_samples(-1, baseline=True)
                print("Baseline sample generation complete.")
            except Exception as e:
                print(f"Error generating baseline samples: {e}")
                traceback.print_exc()
        
        for epoch in range(self.args.epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1}/{self.args.epochs} - Train Loss: {train_loss:.6f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch+1}/{self.args.epochs} - Validation Loss: {val_loss:.6f}")
                
                # Save model if validation loss improved
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model(f"best_model.pt")
                    print(f"Saved best model with validation loss: {val_loss:.6f}")
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement} epochs (current best: {self.best_val_loss:.6f})")
            
            # Save checkpoint only if we're not in store_only_best mode
            if not self.args.store_only_best and (epoch + 1) % self.args.save_every == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pt")
                print(f"Saved checkpoint at epoch {epoch+1}")
                
            # Always save specified epoch checkpoint if configured
            if self.args.save_specific_epoch is not None and (epoch + 1) == self.args.save_specific_epoch:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pt")
                print(f"Saved specified checkpoint at epoch {epoch+1}")
            
            # Always save the latest epoch checkpoint (overwrites previous)
            self.save_model("latest_checkpoint.pt")
            print(f"Saved latest checkpoint at epoch {epoch+1}")
            
            # Generate test samples if configured
            self.generate_test_samples(epoch)
        
        # Save final model
        if not self.args.store_only_best or not self.val_loader:
            # Always save final model if there's no validation set
            # or if we're not only storing the best model
            self.save_model("final_model.pt")
            print("Final model saved")
        else:
            print("Training complete - only best model was saved")
        
        # Final test generation
        if self.args.test_prompts:
            self.generate_test_samples(self.args.epochs - 1)
        
        print(f"Training complete! Best validation loss: {self.best_val_loss:.6f}")
        
        # Provide guidance on number of epochs based on dataset size
        batches_per_epoch = len(self.train_loader)
        samples_per_epoch = batches_per_epoch * self.args.batch_size
        print("\n===== Training Recommendations =====")
        print(f"Dataset size: approximately {samples_per_epoch} samples")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Current epochs: {self.args.epochs}")
        
        recommended_min = max(50, int(10000 / samples_per_epoch * 100))
        recommended_good = max(200, int(10000 / samples_per_epoch * 300))
        recommended_max = max(500, int(10000 / samples_per_epoch * 700))
        
        print(f"\nFor a dataset of this size, recommended training duration:")
        print(f"- Minimum epochs for basic results: {recommended_min}")
        print(f"- Recommended epochs for good results: {recommended_good}")
        print(f"- Maximum epochs before diminishing returns: {recommended_max}")
        print("\nConsider using --store_only_best flag for long training runs to save disk space.")
        print("=======================================")
    
    def save_model(self, filename):
        """Save the model state"""
        save_path = self.output_dir / filename
        torch.save({
            'epoch': self.args.epochs,
            'model_state_dict': self.flow_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_val_loss,
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.flow_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="MelodyFlow Fine-tuning")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/melodyflow-t24-30secs", 
                        help="Pretrained MelodyFlow model name")
    
    # Dataset arguments
    parser.add_argument("--dataset_config", type=str, required=True,
                        help="Path to dataset configuration JSON")
    parser.add_argument("--validation_dataset_config", type=str, default=None,
                        help="Path to validation dataset configuration JSON")
    parser.add_argument("--sample_size", type=int, default=65536,
                        help="Audio sample size (default: 65536)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate (0.0-0.5 recommended)")
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--output_dir", type=str, default="./melodyflow_finetuned",
                        help="Output directory for saving models")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming training")
    parser.add_argument("--cfg_coef", type=float, default=4.0,
                        help="Classifier-free guidance coefficient")
    parser.add_argument("--cfg_free_epochs", type=int, default=0,
                        help="Number of initial epochs without classifier-free guidance")
    parser.add_argument("--store_only_best", action="store_true",
                        help="Store only the best model and final model, skipping intermediate checkpoints")
    parser.add_argument("--save_specific_epoch", type=int, default=None,
                        help="Always save checkpoint at this specific epoch number (regardless of store_only_best)")
    parser.add_argument("--test_prompts", type=str, default=None,
                        help="File containing test prompts or comma-separated list of prompts")
    parser.add_argument("--generate_every", type=int, default=5,
                        help="Generate test samples every N epochs (0 to disable)")
    
    # Sample generation parameters
    parser.add_argument("--eval_solver", type=str, default="euler",
                       help="ODE solver for sample generation (euler or midpoint)")
    parser.add_argument("--eval_steps", type=int, default=125,
                       help="Number of steps for sample generation (higher = better quality)")
    parser.add_argument("--eval_duration", type=float, default=10.0,
                       help="Duration in seconds for generated samples")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MelodyFlowTrainer(args)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_model(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 