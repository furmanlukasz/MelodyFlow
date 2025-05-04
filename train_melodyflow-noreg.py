import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
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
        """Encode audio to latent space using EnCodec"""
        with torch.no_grad():
            # EnCodec expects [B, C, T] format
            latent = self.encodec.encode(audio)[0]
            # Return latent in [B, C, T] format where C is the latent dimension
            latent = latent.squeeze(1)
            
            # Print latent shape for debugging
            if not hasattr(self, '_printed_latent_shape'):
                print(f"Debug - Original latent shape: {latent.shape}")
                print(f"Debug - Flow model latent dim: {self.flow_model.latent_dim}")
                self._printed_latent_shape = True
                
            # Ensure the latent dimension matches what the flow model expects
            if latent.shape[1] != self.flow_model.latent_dim:
                # Only display the warning once
                if not hasattr(self, '_shown_dimension_warning'):
                    print(f"Warning: Latent dimension mismatch - got {latent.shape[1]}, expected {self.flow_model.latent_dim}")
                    print(f"Automatically adjusting dimensions for all batches.")
                    self._shown_dimension_warning = True
                
                # Reshape or resize latent to match the expected dimension
                if latent.shape[1] < self.flow_model.latent_dim:
                    # Pad with zeros
                    padding = torch.zeros(latent.shape[0], 
                                         self.flow_model.latent_dim - latent.shape[1], 
                                         latent.shape[2], 
                                         device=latent.device)
                    latent = torch.cat([latent, padding], dim=1)
                else:
                    # Truncate to the right size
                    latent = latent[:, :self.flow_model.latent_dim, :]
                
                # Only print the adjusted shape once
                if not hasattr(self, '_shown_adjusted_shape'):
                    print(f"Debug - Adjusted latent shape: {latent.shape}")
                    self._shown_adjusted_shape = True
            
            return latent
    
    def prepare_text_conditioning(self, prompts):
        """Prepare text conditioning from prompts"""
        # Create conditioning attributes from text prompts
        conditions = []
        for prompt in prompts:
            # Create a conditioning attribute with the text description
            condition = ConditioningAttributes(text={'description': prompt})
            conditions.append(condition)
        
        # Tokenize the conditions
        tokenized = self.condition_provider.tokenize(conditions)
        
        # Get conditioning tensors
        cond_tensors = self.condition_provider(tokenized)
        
        # For training, we need both conditional and unconditional tensors
        # to implement classifier-free guidance
        # Instead of empty prompts, use the same prompts for unconditional
        # but set a classifier-free guidance flag
        null_conditions = []
        for prompt in prompts:
            # Using the same prompt but with classifier-free guidance flag
            # This ensures tensor dimensions will match
            null_condition = ConditioningAttributes(text={'description': prompt})
            null_condition.classifier_free_guidance = True
            null_conditions.append(null_condition)
        
        null_tokenized = self.condition_provider.tokenize(null_conditions)
        null_tensors = self.condition_provider(null_tokenized)
        
        # Check tensor shapes for debugging
        cond_shape = cond_tensors['description'][0].shape
        null_shape = null_tensors['description'][0].shape
        
        # If shapes don't match, we need to handle it carefully
        if cond_shape != null_shape:
            print(f"Warning: Conditional tensor shape {cond_shape} doesn't match unconditional tensor shape {null_shape}")
            # If the tensors have incompatible shapes, return only conditional tensors
            # and handle CFG differently in the forward pass
            return cond_tensors
        
        # If shapes match, combine conditional and unconditional tensors
        # The flow model will handle the classifier-free guidance
        return {
            'description': (
                torch.cat([cond_tensors['description'][0], null_tensors['description'][0]], dim=0),
                torch.cat([cond_tensors['description'][1], null_tensors['description'][1]], dim=0)
            )
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
        """Train for one epoch"""
        self.flow_model.train()
        total_loss = 0
        
        progress_bar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (audio, info) in enumerate(progress_bar):
            audio = audio.to(self.device)
            
            # Get text prompts from info
            prompts = [item.get('prompt', '') for item in info]
            
            # Encode audio to latent space
            x = self.encode_audio_to_latent(audio)
            
            # Debug print shapes
            if batch_idx == 0 and epoch == 0:
                print(f"Debug - Audio shape: {audio.shape}")
                print(f"Debug - Encoded latent shape: {x.shape}")
            
            # Generate random noise with same shape as x
            n = torch.randn_like(x)
            
            # Align noise samples with true latents
            n = self.align_latents(x, n)
            
            # Sample flow step t using sigmoid of random normal
            t = torch.sigmoid(torch.randn(x.shape[0], 1, device=self.device))
            
            # Create training input: z = x*t + (1-t)*n + small noise
            z = x * t.unsqueeze(-1) + (1 - t.unsqueeze(-1)) * n + 1e-5 * torch.randn_like(x)
            
            # Get text conditioning
            cond_tensors = self.prepare_text_conditioning(prompts)
            
            # Forward pass through flow model with text conditioning
            self.optimizer.zero_grad()
            
            # Apply classifier-free guidance based on the structure of cond_tensors
            cfg_coef = self.args.cfg_coef if epoch >= self.args.cfg_free_epochs else 0.0
            
            # Debugging print for first batch
            if batch_idx == 0 and epoch == 0:
                print(f"Debug - z shape: {z.shape}")
                if isinstance(cond_tensors, dict) and 'description' in cond_tensors:
                    if isinstance(cond_tensors['description'], tuple):
                        print(f"Debug - condition_src shape: {cond_tensors['description'][0].shape}")
                        print(f"Debug - condition_mask shape: {cond_tensors['description'][1].shape}")
                    else:
                        print(f"Debug - condition tensor type: {type(cond_tensors['description'])}")
            
            try:
                if isinstance(cond_tensors, dict) and 'description' in cond_tensors and isinstance(cond_tensors['description'], tuple):
                    # We have both conditional and unconditional tensors concatenated
                    # For training, repeat z for conditional and unconditional passes
                    z_repeated = z.repeat(2, 1, 1)
                    t_repeated = t.repeat(2, 1)
                    
                    # Forward pass using the conditioning tensors
                    pred_velocity = self.flow_model(
                        z_repeated, 
                        t_repeated, 
                        cond_tensors['description'][0], 
                        torch.log(cond_tensors['description'][1].unsqueeze(1).unsqueeze(1))
                    )
                    
                    # Split predicted velocities into conditional and unconditional
                    pred_cond = pred_velocity[:x.shape[0]]
                    pred_uncond = pred_velocity[x.shape[0]:]
                    
                    # Apply classifier-free guidance
                    pred_velocity_guided = (1 + cfg_coef) * pred_cond - cfg_coef * pred_uncond
                else:
                    # We only have conditional tensors
                    # Run two separate forward passes
                    # First, conditional pass
                    pred_cond = self.flow_model(
                        z, 
                        t, 
                        cond_tensors['description'][0], 
                        torch.log(cond_tensors['description'][1].unsqueeze(1).unsqueeze(1))
                    )
                    
                    # Second, unconditional pass (same input but no text conditioning)
                    # Create empty condition tensors of the same shape
                    empty_cond = torch.zeros_like(cond_tensors['description'][0])
                    empty_mask = torch.zeros_like(cond_tensors['description'][1].unsqueeze(1).unsqueeze(1))
                    
                    pred_uncond = self.flow_model(
                        z,
                        t,
                        empty_cond,
                        torch.log(empty_mask + 1e-8)  # Add small value to avoid log(0)
                    )
                    
                    # Apply classifier-free guidance
                    pred_velocity_guided = (1 + cfg_coef) * pred_cond - cfg_coef * pred_uncond
            except RuntimeError as e:
                print(f"Error in forward pass: {e}")
                print(f"Debug - z shape: {z.shape}")
                if isinstance(cond_tensors, dict) and 'description' in cond_tensors:
                    if isinstance(cond_tensors['description'], tuple):
                        print(f"Debug - condition_src shape: {cond_tensors['description'][0].shape}")
                        print(f"Debug - condition_mask shape: {cond_tensors['description'][1].shape}")
                    else:
                        print(f"Debug - condition tensor type: {type(cond_tensors['description'])}")
                raise e
            
            # Target is x - n
            target = x - n
            
            # Compute MSE loss
            loss = F.mse_loss(pred_velocity_guided, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.args.clip_grad)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))
            
            # Debugging - print shapes and values for first few batches
            if batch_idx < 2 and epoch == 0:
                print(f"Audio shape: {audio.shape}")
                print(f"Latent x shape: {x.shape}")
                print(f"Flow step t shape: {t.shape}, range: {t.min().item():.3f}-{t.max().item():.3f}")
                print(f"Prompt example: {prompts[0][:50]}...")
                print(f"Pred velocity shape: {pred_velocity_guided.shape}")
                print(f"Target shape: {target.shape}")
                print(f"Loss: {loss.item():.6f}")
                print(f"CFG coefficient: {cfg_coef}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return 0.0
            
        self.flow_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for audio, info in self.val_loader:
                audio = audio.to(self.device)
                
                # Get text prompts from info
                prompts = [item.get('prompt', '') for item in info]
                
                # Encode audio to latent space
                x = self.encode_audio_to_latent(audio)
                
                # Generate random noise
                n = torch.randn_like(x)
                n = self.align_latents(x, n)
                
                # Sample flow step t
                t = torch.sigmoid(torch.randn(x.shape[0], 1, device=self.device))
                
                # Create validation input
                z = x * t.unsqueeze(-1) + (1 - t.unsqueeze(-1)) * n + 1e-5 * torch.randn_like(x)
                
                # Get text conditioning
                cond_tensors = self.prepare_text_conditioning(prompts)
                
                # Apply classifier-free guidance based on the structure of cond_tensors
                cfg_coef = self.args.cfg_coef
                
                try:
                    if isinstance(cond_tensors, dict) and 'description' in cond_tensors and isinstance(cond_tensors['description'], tuple):
                        # We have both conditional and unconditional tensors concatenated
                        # For validation, repeat z for conditional and unconditional passes
                        z_repeated = z.repeat(2, 1, 1)
                        t_repeated = t.repeat(2, 1)
                        
                        # Forward pass using the conditioning tensors
                        pred_velocity = self.flow_model(
                            z_repeated, 
                            t_repeated, 
                            cond_tensors['description'][0], 
                            torch.log(cond_tensors['description'][1].unsqueeze(1).unsqueeze(1))
                        )
                        
                        # Split predicted velocities into conditional and unconditional
                        pred_cond = pred_velocity[:x.shape[0]]
                        pred_uncond = pred_velocity[x.shape[0]:]
                        
                        # Apply classifier-free guidance
                        pred_velocity_guided = (1 + cfg_coef) * pred_cond - cfg_coef * pred_uncond
                    else:
                        # We only have conditional tensors
                        # Run two separate forward passes
                        # First, conditional pass
                        pred_cond = self.flow_model(
                            z, 
                            t, 
                            cond_tensors['description'][0], 
                            torch.log(cond_tensors['description'][1].unsqueeze(1).unsqueeze(1))
                        )
                        
                        # Second, unconditional pass (same input but no text conditioning)
                        # Create empty condition tensors of the same shape
                        empty_cond = torch.zeros_like(cond_tensors['description'][0])
                        empty_mask = torch.zeros_like(cond_tensors['description'][1].unsqueeze(1).unsqueeze(1))
                        
                        pred_uncond = self.flow_model(
                            z,
                            t,
                            empty_cond,
                            torch.log(empty_mask + 1e-8)  # Add small value to avoid log(0)
                        )
                        
                        # Apply classifier-free guidance
                        pred_velocity_guided = (1 + cfg_coef) * pred_cond - cfg_coef * pred_uncond
                except RuntimeError as e:
                    print(f"Error in validation forward pass: {e}")
                    print(f"Debug - z shape: {z.shape}")
                    if isinstance(cond_tensors, dict) and 'description' in cond_tensors:
                        if isinstance(cond_tensors['description'], tuple):
                            print(f"Debug - condition_src shape: {cond_tensors['description'][0].shape}")
                            print(f"Debug - condition_mask shape: {cond_tensors['description'][1].shape}")
                        else:
                            print(f"Debug - condition tensor type: {type(cond_tensors['description'])}")
                    # Skip this batch instead of crashing
                    continue
                
                # Target is x - n
                target = x - n
                
                # Compute loss
                loss = F.mse_loss(pred_velocity_guided, target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Train the MelodyFlow model"""
        print(f"Starting training for {self.args.epochs} epochs...")
        
        # Track epochs without improvement for early stopping
        epochs_without_improvement = 0
        
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
        
        # Save final model
        if not self.args.store_only_best or not self.val_loader:
            # Always save final model if there's no validation set
            # or if we're not only storing the best model
            self.save_model("final_model.pt")
            print("Final model saved")
        else:
            print("Training complete - only best model was saved")
        
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
                        help="Weight decay")
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