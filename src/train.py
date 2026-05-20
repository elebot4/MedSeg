"""
Medical Segmentation Training - nanoGPT style configuration
Simple, transparent, hackable.
"""
import os
from sys import platform
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
#from torch.cuda.amp import GradScaler
from torch.amp import GradScaler, autocast

from dataset import get_dataloaders
from loss import dice_loss
from model import UNet
from optim import get_optimizer, get_scheduler

# -----------------------------------------------------------------------------
# Default config values - just simple variables!
# I/O settings
out_dir = 'checkpoints'
eval_interval = 10
log_interval = 5
save_interval = 10
device = 'cuda'

# Data settings  
data_dir = 'data/processed/Task01_BrainTumour'
input_shape = (64, 64, 64)  # target shape for all inputs
batch_size = 2
slice_mode = 'fullres'  # axi, cor, sag, fullres

# Model architecture
in_channels = 4
out_channels = 4
num_stages = 4
base_chs = 32
dropout = 0.1
norm_groups = 8
deep_supervision = True
act_type = 'relu'  # relu, gelu, leaky  
norm_type = 'group'  # group, batch, instance, none

# Training settings
nb_epochs = 100
learning_rate = 3e-4
weight_decay = 1e-2
beta1 = 0.9 
beta2 = 0.999

# Optimizer
optimizer = 'AdamW'  # AdamW, SGD
momentum = 0.9  # For SGD

# Scheduler
scheduler = 'PolyLR'  # PolyLR, OneCycleLR, MultiStepLR
gamma = 0.9  # For PolyLR decay

# Mixed precision
dtype = 'float16' if torch.cuda.is_available() else 'float32'

# torch.compile() settings (PyTorch 2.0+)
compile_model = True
compile_mode = 'default'  # default, reduce-overhead, max-autotune

# -----------------------------------------------------------------------------

# Load config overrides
import os
import platform

_config_path = os.path.join(os.path.dirname(__file__), 'config.py')
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(_config_path).read())

# Define pytorch dtype used for mixed precision based on config
ptdtype = {'float32': torch.float32, 'float16': torch.float16}[dtype]

def train(out_dir, eval_interval, log_interval, save_interval, device,
         data_dir, input_shape, batch_size, slice_mode,
         in_channels, out_channels, num_stages, base_chs, dropout, 
         norm_groups, deep_supervision, act_type, norm_type, nb_epochs, 
         learning_rate, weight_decay, beta1, beta2, optimizer_type, 
         momentum, scheduler_type, gamma, dtype, compile_model, compile_mode):
    """
    Main training function with explicit parameters.
    
    Args:
        out_dir: Output directory for checkpoints
        eval_interval: Epochs between evaluations
        log_interval: Batches between log prints
        save_interval: Epochs between checkpoint saves
        device: Device to train on ('cuda' or 'cpu')
        data_dir: Path to processed data directory
        input_shape: Target input shape for all samples
        batch_size: Batch size
        slice_mode: Slicing mode ('axi', 'cor', 'sag', 'fullres')
        in_channels, out_channels: Input and output channels
        num_stages: Number of U-Net stages
        base_chs: Base number of channels
        dropout: Dropout rate
        norm_groups: Groups for GroupNorm
        deep_supervision: Enable deep supervision
        act_type: Activation type ('relu', 'gelu', 'leaky')
        norm_type: Normalization type ('group', 'batch', 'instance', 'none')
        nb_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        beta1, beta2: Adam betas
        optimizer_type: Optimizer type ('AdamW', 'SGD')
        momentum: SGD momentum
        scheduler_type: Scheduler type ('PolyLR', 'OneCycleLR', 'MultiStepLR')
        gamma: Scheduler gamma
        dtype: Mixed precision dtype ('float16' or 'float32')
        compile_model: Whether to compile model
        compile_mode: Compilation mode
    """
    # 1. Setup
    device_obj = torch.device(device)
    torch.manual_seed(42)
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Data
    train_loader, val_loader = get_dataloaders(
        data_dir, batch_size, slice_mode=slice_mode, input_shape=input_shape
    )
    
    # 3. Model, Optimizer, Loss, Scheduler, Scaler
    model = UNet(
        input_shape=input_shape,
        in_channels=in_channels, 
        out_channels=out_channels,
        num_stages=num_stages,
        base_chs=base_chs,
        norm_type=norm_type,
        act_type=act_type,
        dropout=dropout,
        norm_groups=norm_groups,
        deep_supervision=deep_supervision
    ).to(device_obj)
    
    if platform.system() != "Windows":
        try:
            print(f"Compiling model with mode='{compile_mode}'...")
            model = torch.compile(model, mode=compile_mode)
            print("Model compilation successful")
        except Exception as e:
            print(f"Warning: Failed to compile model: {e}")
            print("Continuing with uncompiled model...")
            compile_model = False
    else: 
        print("Warning: torch.compile() is not fully supported on Windows. Skipping compilation.")
        compile_model = False
    
    optimizer = get_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type, 
        beta1=beta1,
        beta2=beta2,
        momentum=momentum
    )
    
    # Calculate total training steps for scheduler
    total_steps = nb_epochs * len(train_loader)
    scheduler = get_scheduler(
        optimizer=optimizer,
        num_training_steps=total_steps,
        scheduler_type=scheduler_type,
        learning_rate=learning_rate,
        gamma=gamma
    )
    
    scaler = GradScaler(device = device) # For Mixed Precision
    
    print(f"Starting training on {device}...")
    print(f"- Model: UNet {num_stages} stages, {base_chs} base channels")
    print(f"- Compiled: {'Yes' if compile_model else 'No'} (mode={compile_mode if compile_model else 'N/A'})")
    print(f"- Input: {input_shape}, batch_size={batch_size}")
    print(f"- Slice mode: {slice_mode}")
    print(f"- Training: {nb_epochs} epochs, lr={learning_rate}")
    print(f"- Scheduler: {scheduler.__class__.__name__}")
    print(f"- Optimizer: {optimizer}, weight_decay={weight_decay}")
    print(f"- Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    for epoch in range(nb_epochs):
        model.train()
        t0 = time.time()
        
        for batch_idx, (batch, gt) in enumerate(train_loader):
            batch = batch.to(device_obj, non_blocking=True)
            gt = gt.to(device_obj, non_blocking=True) # Index tensor [B, D, H, W]

            # Mixed Precision Forward Pass
            with autocast(device_type = device, dtype = ptdtype):
                preds = model(batch) # List of [Res1, Res2, ...] 
                
                total_loss = 0
                
                # GPU-side one-hotting for Dice
                # Doing this here on GPU reduce CPU-GPU transfer overhead
                # Doing dice loss without one-hot encoding is very bad depending on the number of classes 
                gt_shape = (gt.shape[0], out_channels, *gt.shape[1:]) 
                gt_onehot = torch.zeros(gt_shape, device = device_obj, dtype = ptdtype)
                gt_onehot.scatter_(1, gt.unsqueeze(1), 1)  # [B, D, H, W] -> [B, C, D, H, W]
                
                for i, p in enumerate(reversed(preds)):
                    stride = 2**i
                    # Strided views for 2D and 3D masks
                    spatial_slices = (slice(None, None, stride),) * (gt.ndim - 1)
                    gt_idx = gt[(slice(None), *spatial_slices)]
                    gt_onehot_idx = gt_onehot[(slice(None), *spatial_slices)]
                    
                    # Loss Calculation
                    loss_ce = F.cross_entropy(p, gt_idx) # remove singleton channel dim
                    loss_dice = dice_loss(F.softmax(p, dim=1), gt_onehot_idx)
                    weight = 1 / (2**i)
                    total_loss += weight * (loss_ce + loss_dice)
            
            # Backward Pass with Scaler
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Log training progress
            if batch_idx % log_interval == 0:
                dt = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{nb_epochs} [{batch_idx}/{len(train_loader)}] Loss: {total_loss.item():.4f} | LR: {lr:.6f} | Time: {dt:.2f}s")
        
        
        # 4. Validation 
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            val_dice = 0
            with torch.no_grad():
                for batch, gt in val_loader:
                    batch, gt = batch.to(device), gt.to(device)
                    # In eval mode, UNet returns only the high-res tensor
                    p = model(batch) 

                    # Simple Dice Metric for monitoring
                    gt_shape = (gt.shape[0], out_channels, *gt.shape[1:]) 
                    gt_onehot = torch.zeros(gt_shape, device = device_obj, dtype = ptdtype)
                    gt_onehot.scatter_(1, gt.unsqueeze(1), 1)
                    
                    p_onehot = p.argmax(dim = 1, keepdim = True)
                    val_dice += (1 - dice_loss(p_onehot, gt_onehot)) # 1 - (1-dice) = dice

            dt = time.time() - t0
            print(f"Epoch {epoch+1}/{nb_epochs} | Loss: {total_loss.item():.4f} | Val Dice: {val_dice/len(val_loader):.4f} | Time: {dt:.2f}s")

        # 5. Save Checkpoint 
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(out_dir, f"ckpt_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    try:
        train(
            out_dir=out_dir,
            eval_interval=eval_interval, 
            log_interval=log_interval,
            save_interval=save_interval,
            device=device,
            data_dir=data_dir,
            input_shape=input_shape,
            batch_size=batch_size,
            slice_mode=slice_mode,
            in_channels=in_channels,
            out_channels=out_channels,
            num_stages=num_stages,
            base_chs=base_chs,
            dropout=dropout,
            norm_groups=norm_groups,
            deep_supervision=deep_supervision,
            act_type=act_type,
            norm_type=norm_type,
            nb_epochs=nb_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            optimizer_type=optimizer,
            momentum=momentum,
            scheduler_type=scheduler,
            gamma=gamma,
            dtype=dtype,
            compile_model=compile_model,
            compile_mode=compile_mode
        )
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\nCUDA out of memory error: {e}")
            print("Try reducing batch_size or input_shape")
        else:
            raise
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise