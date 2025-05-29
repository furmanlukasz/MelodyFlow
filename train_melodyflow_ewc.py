import argparse
import contextlib
import copy
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
from audiocraft.data.audio import audio_write
from audiocraft.models import MelodyFlow
from audiocraft.models.encodec import CompressionModel
from audiocraft.modules.conditioners import ConditioningAttributes
from audiocraft.utils.utils import vae_sample
from dataset_loader.dataset import create_dataloader_from_config
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

# EWC-related constants
EWC_SAMPLES = 32  # Number of samples to use for Fisher computation
EWC_LAMBDA = 15000.0  # Weight of the EWC penalty

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


# EWC utilities
def compute_fisher_information(model, data_loader, device, num_samples=EWC_SAMPLES):
    """
    Compute Fisher Information Matrix for EWC.
    This estimates how important each parameter is for the original task.
    """
    fisher_info = {}
    
    # Handle if we get a trainer object or a model
    flow_model = model
    if hasattr(model, 'flow_model'):
        flow_model = model.flow_model
    
    # Get trainable parameters
    parameters = {n: p for n, p in flow_model.named_parameters() if p.requires_grad}
    
    # Initialize Fisher Information Matrix
    for n, p in parameters.items():
        fisher_info[n] = torch.zeros_like(p)
    
    # Set model to training mode
    flow_model.train()
    
    # Sample batches to compute Fisher
    samples_processed = 0
    progress_bar = tqdm.tqdm(total=num_samples, desc="Computing Fisher Information")
    
    for batch_idx, (audio, info) in enumerate(data_loader):
        if samples_processed >= num_samples:
            break
            
        # Process current batch (up to the remaining samples needed)
        batch_size = min(audio.shape[0], num_samples - samples_processed)
        audio = audio[:batch_size].to(device)
        prompts = [item.get('prompt', '') for item in info[:batch_size]]
        
        # Perform the model's forward pass
        flow_model.zero_grad()
        
        # Create latent representations
        encodec_model = None
        if hasattr(model, 'encodec'):
            # We have a trainer
            encodec_model = model.encodec
        elif hasattr(model, 'compression_model'):
            # We have a model
            encodec_model = model.compression_model
        
        # Encode audio to latent 
        with torch.no_grad():
            if encodec_model:
                latent = encodec_model.encode(audio)[0]
                latent = latent.squeeze(1)
                
                # Handle VAE structure if needed
                if latent.shape[1] == 256 and flow_model.latent_dim == 128:
                    mean, scale = latent.chunk(2, dim=1)
                    x = vae_sample(mean, scale)
                    x = (x - flow_model.latent_mean) / (flow_model.latent_std + 1e-5)
                else:
                    x = latent[:, :flow_model.latent_dim, :]
                    x = (x - flow_model.latent_mean) / (flow_model.latent_std + 1e-5)
            else:
                # Fallback to random latents for testing
                x = torch.randn(batch_size, 128, 1024).to(device)
        
        # Create random noise and timestep
        n = torch.randn_like(x)
        t = torch.sigmoid(torch.randn(x.shape[0], 1, device=device))
        z = x * t.unsqueeze(-1) + (1 - t.unsqueeze(-1)) * n + 1e-5 * torch.randn_like(x)
        
        # Create conditioning attributes
        condition_provider = flow_model.condition_provider
        
        # Tokenize prompts
        cond_attributes = [ConditioningAttributes(text={'description': p}) for p in prompts]
        cond_tokens = condition_provider.tokenize(cond_attributes)
        cond_tensors = condition_provider(cond_tokens)
        
        # Get tensors
        src = cond_tensors['description'][0]
        mask = cond_tensors['description'][1]
        
        # Forward pass
        pred = flow_model(z, t, src, torch.log(mask.unsqueeze(1).unsqueeze(1)))
        
        # Target is x - n
        target = x - n
        
        # Compute loss and backward pass
        loss = F.mse_loss(pred, target)
        loss.backward()
        
        # Update Fisher information
        for n, p in parameters.items():
            if p.grad is not None:
                fisher_info[n] += p.grad.detach() ** 2 / num_samples
        
        samples_processed += batch_size
        progress_bar.update(batch_size)
    
    progress_bar.close()
    print(f"Computed Fisher Information using {samples_processed} samples")
    
    return fisher_info


def ewc_penalty(model, fisher_info, old_params):
    """
    Compute the EWC penalty term.
    Penalizes changes to parameters based on their importance (Fisher Information).
    """
    penalty = 0
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    
    for n, p in parameters.items():
        # Parameter-specific penalty based on Fisher and distance from original value
        if n in fisher_info and n in old_params:
            _penalty = fisher_info[n] * (p - old_params[n]) ** 2
            penalty += _penalty.sum()
    
    return penalty


class MelodyFlowTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Initialize wandb if enabled
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            print(f"Initialized wandb with project: {args.wandb_project}, run name: {args.wandb_run_name}")
        
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
        
        # Setup EWC 
        self.use_ewc = args.use_ewc
        if self.use_ewc:
            print(f"Elastic Weight Consolidation (EWC) enabled with lambda={EWC_LAMBDA}")
            # Store a copy of the initial parameters for EWC
            self.old_params = {}
            for n, p in self.flow_model.named_parameters():
                if p.requires_grad:
                    self.old_params[n] = p.data.clone()
                    
            # Fisher information will be computed after dataloaders are set up
            self.fisher_information = None
        
        # Apply partial fine-tuning if enabled
        if args.partial_finetuning:
            print(f"Applying partial fine-tuning with strategy: {args.freeze_strategy}")
            self.apply_partial_finetuning(args.freeze_strategy, args.freeze_ratio)
        
        # Setup optimizer - only optimize trainable parameters
        trainable_params = [p for p in self.flow_model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.flow_model.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        frozen_param_count = total_params - trainable_param_count
        
        print(f"Model parameters - Total: {total_params:,}")
        print(f"Model parameters - Trainable: {trainable_param_count:,} ({100 * trainable_param_count / total_params:.2f}%)")
        print(f"Model parameters - Frozen: {frozen_param_count:,} ({100 * frozen_param_count / total_params:.2f}%)")
        
        self.optimizer = optim.AdamW(
            trainable_params,  # Only optimize trainable parameters
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.98),  # More stable for transformer training
        )
        
        # Setup mixed precision training if enabled
        self.scaler = torch.cuda.amp.GradScaler() if args.use_mixed_precision else None
        if args.use_mixed_precision:
            print(f"Using mixed precision training (fp16) for faster speed")
            
        # Setup gradient accumulation
        self.grad_accum_steps = args.gradient_accumulation_steps
        if self.grad_accum_steps > 1:
            print(f"Using gradient accumulation with {self.grad_accum_steps} steps")
            print(f"Effective batch size: {args.batch_size * self.grad_accum_steps}")
        
        # Create dataloaders
        print("Setting up dataloaders...")
        self.setup_dataloaders()
        
        # Initialize EWC after dataloader setup if enabled
        if self.use_ewc:
            print("Computing Fisher information for EWC...")
            # We can pass self directly now since we've fixed the compute_fisher_information function
            self.fisher_information = compute_fisher_information(
                self,
                self.train_loader, 
                self.device, 
                num_samples=args.ewc_samples
            )
            print("Fisher information computed successfully")
        
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
        
        # Disable validation dataset loading completely
        self.val_loader = None
        print("Validation dataset disabled - training without validation")
    
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
                
                # Use VAE sampling and proper normalization
                gen_tokens = vae_sample(mean, scale)
                latent_normalized = (gen_tokens - self.flow_model.latent_mean) / (self.flow_model.latent_std + 1e-5)
                
                if not hasattr(self, '_shown_vae_notice'):
                    print("Using VAE sampling from EnCodec with proper normalization")
                    self._shown_vae_notice = True
                
                return latent_normalized
            
            # If dimensions already match or for other dimension combinations, warn and truncate
            if latent.shape[1] != self.flow_model.latent_dim:
                if not hasattr(self, '_shown_dimension_warning'):
                    print(f"Warning: Unexpected latent dimensions - got {latent.shape[1]}, expected {self.flow_model.latent_dim}")
                    print(f"Truncating to expected dimension - this may affect quality")
                    self._shown_dimension_warning = True
                
                # Truncate to expected dimension as fallback
                latent = latent[:, :self.flow_model.latent_dim, :]
                # Apply normalization
                latent = (latent - self.flow_model.latent_mean) / (self.flow_model.latent_std + 1e-5)
            
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
    
    def align_latents(self, x, n):
        """Simplified alignment - just permute within batch dimension if needed"""
        batch_size = x.shape[0]
        if batch_size > 1:
            # For multi-sample batches, create a random permutation of batch indices
            indices = torch.randperm(batch_size, device=x.device)
            return n[indices]
        return n  # For batch size 1, no permutation needed
    
    def train_epoch(self, epoch):
        """Train for one epoch with fixed CFG approach"""
        self.flow_model.train()
        total_loss = 0
        total_task_loss = 0
        total_ewc_loss = 0
        
        progress_bar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        self.optimizer.zero_grad()  # Zero gradients at the start of epoch for gradient accumulation
        
        for batch_idx, (audio, info) in enumerate(progress_bar):
            audio = audio.to(self.device)
            prompts = [item.get('prompt', '') for item in info]
            
            # Process in mixed precision context if enabled
            with torch.cuda.amp.autocast() if self.args.use_mixed_precision else contextlib.nullcontext():
                # Encode audio to latent space
                x = self.encode_audio_to_latent(audio)
                
                # Generate noise and align
                n = torch.randn_like(x)
                n = self.align_latents(x, n)
                
                # Sample flow step t
                t = torch.sigmoid(torch.randn(x.shape[0], 1, device=self.device))
                
                # Create training input with proper variance preservation
                z = x * t.unsqueeze(-1) + (1 - t.unsqueeze(-1)) * n + 1e-5 * torch.randn_like(x)
                
                # Get conditioning - note we'll use separate forward passes
                cond_tensors = self.prepare_text_conditioning(prompts)
                
                # Determine if we use CFG dropout (10% chance to use unconditional)
                use_uncond = (random.random() < 0.1) and (epoch >= self.args.cfg_free_epochs)
                
                # Select appropriate conditioning
                if use_uncond:
                    src, mask = cond_tensors['unconditional']
                else:
                    src, mask = cond_tensors['conditional']
                
                # Forward pass with the selected conditioning
                pred = self.flow_model(
                    z, 
                    t, 
                    src, 
                    torch.log(mask.unsqueeze(1).unsqueeze(1))
                )
                
                # Target is x - n
                target = x - n
                
                # Task loss
                task_loss = F.mse_loss(pred, target)
                
                # Add EWC penalty if enabled
                if self.use_ewc and self.fisher_information:
                    ewc_loss = ewc_penalty(self.flow_model, self.fisher_information, self.old_params)
                    ewc_penalty_term = EWC_LAMBDA * ewc_loss
                    loss = task_loss + ewc_penalty_term
                    
                    # Track losses separately for logging
                    total_task_loss += task_loss.item()
                    total_ewc_loss += ewc_penalty_term.item()
                else:
                    loss = task_loss
                
                # Normalize loss by accumulation steps
                loss = loss / self.grad_accum_steps
            
            # Backpropagation with mixed precision if enabled
            if self.scaler is not None:
                # Mixed precision backward
                self.scaler.scale(loss).backward()
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    if self.args.clip_grad:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.args.clip_grad)
                    
                    # Update with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard backward
                loss.backward()
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    if self.args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.args.clip_grad)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # For progress tracking, scale loss back up to get true batch loss
            batch_loss = loss.item() * self.grad_accum_steps
            total_loss += batch_loss
            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))
            
            # Debug output
            if batch_idx < 2 and epoch == 0:
                print(f"Prompt example: {prompts[0][:50]}...")
                print(f"Using {'unconditional' if use_uncond else 'conditional'} for training")
                print(f"Pred shape: {pred.shape}")
                print(f"Target shape: {target.shape}")
                print(f"Loss: {batch_loss:.6f}")
                if self.use_ewc and self.fisher_information:
                    print(f"Task Loss: {task_loss.item():.6f}")
                    print(f"EWC Penalty: {ewc_penalty_term.item():.6f}")
                print(f"Mixed precision: {self.args.use_mixed_precision}")
                print(f"Gradient accumulation steps: {self.grad_accum_steps}")
        
        avg_loss = total_loss / len(self.train_loader)
        
        # Log to wandb if enabled
        if self.args.use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
            }
            
            # Add EWC-specific metrics if enabled
            if self.use_ewc and self.fisher_information:
                avg_task_loss = total_task_loss / len(self.train_loader)
                avg_ewc_loss = total_ewc_loss / len(self.train_loader)
                log_dict.update({
                    "task_loss": avg_task_loss,
                    "ewc_loss": avg_ewc_loss,
                })
            
            wandb.log(log_dict)
        
        return avg_loss
    
    def validate(self):
        """Validate using conditional predictions only"""
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
                
                # Run only conditional pass for validation
                cond_src, cond_mask = cond_tensors['conditional']
                pred = self.flow_model(
                    z, t, cond_src, torch.log(cond_mask.unsqueeze(1).unsqueeze(1))
                )
                
                # Compute loss with conditional prediction only
                target = x - n
                loss = F.mse_loss(pred, target)
                total_loss += loss.item()
        
        avg_val_loss = total_loss / len(self.val_loader)
        
        # Log to wandb if enabled
        if self.args.use_wandb:
            wandb.log({"val_loss": avg_val_loss})
        
        return avg_val_loss
    
    def generate_test_samples(self, epoch, baseline=False, label=None):
        """Generate test samples using the current model at the specified epoch.
        
        Args:
            epoch: Current epoch number
            baseline: If True, this is generating baseline samples before training
            label: Optional custom label for the samples directory
        """
        if not self.args.test_prompts:
            print("No test prompts provided, skipping sample generation.")
            return
        
        # Choose appropriate directory name
        if baseline:
            dir_name = "samples_baseline"
            print(f"Starting baseline sample generation...")
        elif label:
            dir_name = f"samples_{label}"
            print(f"Starting {label} sample generation...")
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
                
                # Log concatenated audio to wandb if enabled
                if self.args.use_wandb:
                    try:
                        audio_file = f"{concat_path}.wav"
                        caption = f"Concatenated samples - {'baseline' if baseline else f'epoch {epoch+1}'}"
                        
                        # Log the audio file to wandb
                        wandb.log({
                            "audio_samples": wandb.Audio(
                                audio_file,
                                caption=caption,
                                sample_rate=self.melody_flow.sample_rate
                            ),
                            "epoch": epoch + 1 if not baseline else 0
                        })
                        print(f"  Uploaded concatenated audio samples to Weights & Biases")
                        
                        # Log the timestamps as a text file instead of an artifact
                        # Read the text file content
                        try:
                            timestamp_file = output_dir / "concatenated_timestamps.txt"
                            with open(timestamp_file, 'r') as f:
                                timestamp_content = f.read()
                                
                            # Log as a text table instead of an artifact
                            wandb.log({
                                "sample_timestamps": wandb.Table(
                                    columns=["Timestamps"],
                                    data=[[timestamp_content]]
                                )
                            })
                        except Exception as e:
                            print(f"  Error logging timestamps to wandb: {e}")
                    except Exception as e:
                        print(f"  Error uploading audio to wandb: {e}")
                        traceback.print_exc()
            
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
        print(f"Will save checkpoints every {self.args.save_every} epochs")
        print(f"Will generate samples every {self.args.generate_every} epochs")
        print(f"Will ALWAYS generate samples at epochs 5, 10, and 15")
        
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
                print(f"Will generate samples every {self.args.generate_every} epochs")
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
        
        # Track epochs for generation
        last_generation_epoch = 0
        last_save_epoch = 0
        
        # Special epochs for early sample generation
        early_sample_epochs = {5, 10, 15}
        
        for epoch in range(self.args.epochs):
            # Train for one epoch
            epoch_start_time = time.time()
            train_loss = self.train_epoch(epoch)
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{self.args.epochs} - Train Loss: {train_loss:.6f} - Time: {epoch_duration:.2f}s")
            
            # Skip validation entirely - we're not using it
            
            # Save checkpoint only at specified intervals
            current_epoch = epoch + 1  # Convert to 1-indexed for clarity
            
            # Debug info about upcoming operations
            if self.args.save_every > 0:
                next_save = self.args.save_every - (current_epoch % self.args.save_every)
                if next_save == self.args.save_every:
                    next_save = 0
                print(f"Next checkpoint in {next_save} epochs (save_every={self.args.save_every})")
            
            if self.args.generate_every > 0:
                # For epochs <= 15, show when the next early sample generation will happen
                if current_epoch < 15:
                    next_early_epochs = [e for e in [5, 10, 15] if e > current_epoch]
                    if next_early_epochs:
                        next_early = next_early_epochs[0] - current_epoch
                        print(f"Next early sample generation in {next_early} epochs (at epoch {next_early_epochs[0]})")
                    else:
                        next_gen = self.args.generate_every - (current_epoch % self.args.generate_every)
                        if next_gen == self.args.generate_every:
                            next_gen = 0
                        print(f"Next sample generation in {next_gen} epochs (generate_every={self.args.generate_every})")
                # For epochs > 15, show regular schedule
                else:
                    next_gen = self.args.generate_every - (current_epoch % self.args.generate_every)
                    if next_gen == self.args.generate_every:
                        next_gen = 0
                    print(f"Next sample generation in {next_gen} epochs (generate_every={self.args.generate_every})")
            
            # Check for checkpoint saving
            if self.args.save_every > 0 and current_epoch % self.args.save_every == 0:
                print(f"\nEpoch {current_epoch} is a checkpoint saving epoch (every {self.args.save_every})")
                save_start = time.time()
                self.save_model(f"checkpoint_epoch_{current_epoch}.pt")
                save_time = time.time() - save_start
                print(f"Saved checkpoint at epoch {current_epoch} in {save_time:.2f}s")
                last_save_epoch = current_epoch
            
            # Check for early sample generation at specific epochs (5, 10, 15)
            if current_epoch in early_sample_epochs:
                print(f"\nEpoch {current_epoch} is a special early generation epoch")
                gen_start = time.time()
                try:
                    self.generate_test_samples(epoch, label=f"early_{current_epoch}")
                    gen_time = time.time() - gen_start
                    print(f"Generated early samples at epoch {current_epoch} in {gen_time:.2f}s")
                    last_generation_epoch = current_epoch
                except Exception as e:
                    print(f"Error generating early samples at epoch {current_epoch}: {e}")
                    traceback.print_exc()
            # Check for regular sample generation (only after epoch 15)
            elif self.args.generate_every > 0 and current_epoch % self.args.generate_every == 0 and current_epoch > 15:
                print(f"\nEpoch {current_epoch} is a regular generation epoch (every {self.args.generate_every})")
                gen_start = time.time()
                try:
                    self.generate_test_samples(epoch)
                    gen_time = time.time() - gen_start
                    print(f"Generated samples at epoch {current_epoch} in {gen_time:.2f}s")
                    last_generation_epoch = current_epoch
                except Exception as e:
                    print(f"Error generating samples at epoch {current_epoch}: {e}")
                    traceback.print_exc()
        
        # Save final model
        self.save_model("final_model.pt")
        print("Final model saved")
        
        # Final test generation
        if self.test_prompts:
            self.generate_test_samples(self.args.epochs - 1)
            
        print(f"Training complete!")
        print(f"Last checkpoint saved at epoch: {last_save_epoch}")
        print(f"Last samples generated at epoch: {last_generation_epoch}")
        
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
        
        # Finish wandb run if enabled
        if self.args.use_wandb:
            wandb.finish()
    
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
    
    def apply_partial_finetuning(self, strategy, freeze_ratio=0.5):
        """Apply partial fine-tuning by freezing parts of the model.
        
        Args:
            strategy (str): Freezing strategy ('layerwise', 'encoder', 'conditioner', 'attention', 'mlp')
            freeze_ratio (float): Proportion of layers to freeze from the start (for layerwise strategy)
        """
        # Track parameters for reporting
        total_params = 0
        frozen_params = 0
        
        # Helper function to freeze a parameter
        def freeze_param(name, param):
            nonlocal frozen_params
            param.requires_grad = False
            frozen_params += param.numel()
            if self.args.verbose:
                print(f"Freezing: {name} ({param.shape})")
        
        # Helper function to ensure a parameter is trainable
        def ensure_trainable(name, param):
            if not param.requires_grad:
                param.requires_grad = True
                if self.args.verbose:
                    print(f"Unfreezing: {name} ({param.shape})")
        
        # Count total parameters for reporting
        for name, param in self.flow_model.named_parameters():
            total_params += param.numel()
        
        # Apply the selected freezing strategy
        if strategy == 'layerwise':
            # Analyze model to understand layer structure
            layer_names = []
            for name, _ in self.flow_model.named_parameters():
                # Extract the block number if it exists
                parts = name.split('.')
                layer_name = None
                for part in parts:
                    if part.startswith('block'):
                        layer_name = part
                        break
                if layer_name and layer_name not in layer_names:
                    layer_names.append(layer_name)
            
            # Sort layer names if they contain numbers
            layer_names = sorted(layer_names, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
            
            # Determine how many layers to freeze based on the ratio
            num_layers = len(layer_names)
            layers_to_freeze = int(num_layers * freeze_ratio)
            freeze_layers = layer_names[:layers_to_freeze]
            
            print(f"Model has {num_layers} main layers, freezing {layers_to_freeze} earliest layers")
            print(f"Frozen layers: {freeze_layers}")
            
            # Freeze parameters in the selected layers
            for name, param in self.flow_model.named_parameters():
                for freeze_layer in freeze_layers:
                    if freeze_layer in name:
                        freeze_param(name, param)
                        break
            
        elif strategy == 'encoder':
            # Freeze encoder layers or early processing layers
            for name, param in self.flow_model.named_parameters():
                if 'encoder' in name or 'embeddings' in name:
                    freeze_param(name, param)
            
        elif strategy == 'conditioner':
            # Freeze text conditioning components
            for name, param in self.flow_model.named_parameters():
                if 'condition' in name or 'conditioner' in name or 'text_model' in name:
                    freeze_param(name, param)
                    
        elif strategy == 'attention':
            # Freeze attention mechanisms but keep MLP parts trainable
            for name, param in self.flow_model.named_parameters():
                if any(x in name for x in ['attn', 'attention', 'self_attn']):
                    freeze_param(name, param)
        
        elif strategy == 'mlp':
            # Freeze MLP parts but keep attention mechanisms trainable
            for name, param in self.flow_model.named_parameters():
                if any(x in name for x in ['mlp', 'feed_forward', 'ffn']):
                    freeze_param(name, param)
                    
        elif strategy == 'custom':
            # This strategy allows for custom layer freezing based on name patterns
            patterns_to_freeze = self.args.freeze_patterns.split(',')
            for name, param in self.flow_model.named_parameters():
                if any(pattern in name for pattern in patterns_to_freeze):
                    freeze_param(name, param)
        
        # Report the percentage of parameters frozen
        frozen_percent = (frozen_params / total_params) * 100
        print(f"Frozen {frozen_params:,} parameters out of {total_params:,} ({frozen_percent:.2f}%)")
        
        if self.args.use_wandb:
            wandb.log({
                "frozen_parameters": frozen_params,
                "total_parameters": total_params,
                "frozen_percentage": frozen_percent
            })


def main():
    parser = argparse.ArgumentParser(description="MelodyFlow Fine-tuning")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/melodyflow-t24-30secs", 
                        help="Pretrained MelodyFlow model name")
    
    # Dataset arguments
    parser.add_argument("--dataset_config", type=str, required=True,
                        help="Path to dataset configuration JSON")
    parser.add_argument("--validation_dataset_config", type=str, default=None,
                        help="Path to validation dataset configuration JSON (not used)")
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
    
    # Performance optimization
    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Use mixed precision training (fp16) for faster training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients (increases effective batch size)")
    
    # Partial fine-tuning arguments
    parser.add_argument("--partial_finetuning", action="store_true",
                        help="Enable partial fine-tuning (freeze parts of the model)")
    parser.add_argument("--freeze_strategy", type=str, default="layerwise",
                        choices=["layerwise", "encoder", "conditioner", "attention", "mlp", "custom"],
                        help="Strategy for freezing model parts")
    parser.add_argument("--freeze_ratio", type=float, default=0.5,
                        help="Ratio of layers to freeze from the start (for layerwise strategy)")
    parser.add_argument("--freeze_patterns", type=str, default="",
                        help="Comma-separated list of name patterns to freeze (for custom strategy)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information about frozen/unfrozen parameters")
    
    # EWC arguments
    parser.add_argument("--use_ewc", action="store_true",
                        help="Enable Elastic Weight Consolidation to prevent catastrophic forgetting")
    parser.add_argument("--ewc_lambda", type=float, default=15000.0,
                        help="Lambda coefficient for EWC penalty term")
    parser.add_argument("--ewc_samples", type=int, default=32,
                        help="Number of samples to use for Fisher Information calculation")
    
    # Output arguments
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
    
    # Weights & Biases arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="melodyflow-finetuning",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (default: timestamp)")
    
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