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
from dataset_loader.dataset import create_dataloader_from_config
from torch.utils.data import DataLoader
from torch_linear_assignment import batch_linear_assignment


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
            # Return latent in [B, C, T] format
            return latent.squeeze(1)
    
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
            
            # Generate random noise with same shape as x
            n = torch.randn_like(x)
            
            # Align noise samples with true latents
            n = self.align_latents(x, n)
            
            # Sample flow step t using sigmoid of random normal
            t = torch.sigmoid(torch.randn(x.shape[0], 1, device=self.device))
            
            # Create training input: z = x*t + (1-t)*n + small noise
            z = x * t.unsqueeze(-1) + (1 - t.unsqueeze(-1)) * n + 1e-5 * torch.randn_like(x)
            
            # Get text conditioning
            # For simplicity, use empty conditions during initial implementation
            # Will be replaced with proper text conditioning
            condition_src = torch.zeros((x.shape[0], 1, self.flow_model.latent_dim), device=self.device)
            condition_mask = torch.zeros((x.shape[0], 1, 1, 1), device=self.device)
            
            # Forward pass through flow model
            self.optimizer.zero_grad()
            pred_velocity = self.flow_model(z, t, condition_src, condition_mask)
            
            # Target is x - n
            target = x - n
            
            # Compute MSE loss
            loss = F.mse_loss(pred_velocity, target)
            
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
                print(f"Pred velocity shape: {pred_velocity.shape}")
                print(f"Target shape: {target.shape}")
                print(f"Loss: {loss.item():.6f}")
        
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
                
                # Encode audio to latent space
                x = self.encode_audio_to_latent(audio)
                
                # Generate random noise
                n = torch.randn_like(x)
                n = self.align_latents(x, n)
                
                # Sample flow step t
                t = torch.sigmoid(torch.randn(x.shape[0], 1, device=self.device))
                
                # Create validation input
                z = x * t.unsqueeze(-1) + (1 - t.unsqueeze(-1)) * n + 1e-5 * torch.randn_like(x)
                
                # Placeholder conditioning
                condition_src = torch.zeros((x.shape[0], 1, self.flow_model.latent_dim), device=self.device)
                condition_mask = torch.zeros((x.shape[0], 1, 1, 1), device=self.device)
                
                # Forward pass
                pred_velocity = self.flow_model(z, t, condition_src, condition_mask)
                
                # Target is x - n
                target = x - n
                
                # Compute loss
                loss = F.mse_loss(pred_velocity, target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Train the MelodyFlow model"""
        print(f"Starting training for {self.args.epochs} epochs...")
        
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
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pt")
        
        # Save final model
        self.save_model("final_model.pt")
        print("Training complete!")
    
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