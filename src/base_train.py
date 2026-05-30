"""
Medical Segmentation Training - nanoGPT style configuration
Simple, transparent, hackable.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from dataset import get_dataloaders
from loss import dice_loss
from model import UNet
from optim import get_optimizer, get_scheduler


class DummyWandb:
    """Useful when we do not want wandb but keep the same call signatures."""

    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


# -----------------------------------------------------------------------------
# Default config values - just simple variables!
# I/O settings
out_dir = "outputs"
eval_interval = 10
log_interval = 5
save_interval = 10
device = "cuda"

# Data settings
data_dir = "data/processed/Task01_BrainTumour"
input_shape = (64, 64, 64)  # target shape for all inputs
batch_size = 2
slice_mode = "fullres"  # axi, cor, sag, fullres

# Model architecture
in_channels = 4
out_channels = 4
num_stages = 5
base_chs = 32
dropout = 0.1
norm_groups = 8
deep_supervision = True
act_type = "relu"  # relu, gelu, leaky
norm_type = "group"  # group, batch, instance, none

# Training settings
nb_epochs = 100
learning_rate = 3e-4
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999

# Optimizer
optimizer = "AdamW"  # AdamW, SGD
momentum = 0.9  # For SGD

# Scheduler
scheduler = "PolyLR"  # PolyLR, OneCycleLR, MultiStepLR
gamma = 0.9  # For PolyLR decay

# Mixed precision
dtype = "float16" if torch.cuda.is_available() else "float32"

# wandb logging
wandb_log = True  # set True to enable wandb
run_name = ""  # prefix for the run name (e.g. '3d_fullres')

# torch.compile() settings (PyTorch 2.0+)
compile_model = True
compile_mode = "default"  # default, reduce-overhead, max-autotune
checkpoint = None  # Path to checkpoint to resume training (optional)

# -----------------------------------------------------------------------------

# Load config overrides

_config_path = os.path.join(os.path.dirname(__file__), "config.py")
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open(_config_path).read())

# Define pytorch dtype used for mixed precision based on config
ptdtype = {"float32": torch.float32, "float16": torch.float16}[dtype]

# configure wandb project and run name
# Build wandb config dict — None disables logging entirely
_run_name = (run_name + "_" if run_name else "") + time.strftime("%b%d").lower()
wb_config = (
    {
        "project": "medseg",
        "name": _run_name,
        "config": {k: globals()[k] for k in config_keys},
    }
    if wandb_log
    else None
)


def train(
    out_dir,
    eval_interval,
    log_interval,
    save_interval,
    device,
    data_dir,
    input_shape,
    batch_size,
    slice_mode,
    in_channels,
    out_channels,
    num_stages,
    base_chs,
    dropout,
    norm_groups,
    deep_supervision,
    act_type,
    norm_type,
    nb_epochs,
    learning_rate,
    weight_decay,
    beta1,
    beta2,
    optimizer_type,
    momentum,
    scheduler_type,
    gamma,
    dtype,
    compile_model,
    compile_mode,
    checkpoint,
    wb_config,
):

    # 1. Setup
    device_obj = torch.device(device)
    best_val_dice = 0.0
    start_epoch = 0
    torch.manual_seed(42)

    # Ensure output directory exists
    run_dir = os.path.join(out_dir, f"{wb_config['name'] if wb_config else 'run_' + time.strftime('%b%d').lower()}")
    os.makedirs(run_dir, exist_ok=True)

    # wandb
    wandb_run = DummyWandb()
    if wb_config is not None:
        import wandb

        wandb_run = wandb.init(**wb_config)
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
        deep_supervision=deep_supervision,
    ).to(device_obj)

    if sys.platform != "win32":
        try:
            print(f"Compiling model with mode='{compile_mode}'...")
            model = torch.compile(model, mode=compile_mode)
            print("Model compilation successful")
        except Exception as e:
            print(f"Warning: Failed to compile model: {e}")
            print("Continuing with uncompiled model...")
            compile_model = False
    else:
        print(
            "Warning: torch.compile() is not fully supported on Windows. Skipping compilation."
        )
        compile_model = False

    optimizer = get_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        beta1=beta1,
        beta2=beta2,
        momentum=momentum,
    )

    # Calculate total training steps for scheduler
    total_steps = nb_epochs * len(train_loader)
    scheduler = get_scheduler(
        optimizer=optimizer,
        num_training_steps=total_steps,
        scheduler_type=scheduler_type,
        learning_rate=learning_rate,
        gamma=gamma,
    )

    scaler = GradScaler(device=device)  # For Mixed Precision

    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=device_obj)

        model_data = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

        model.load_state_dict(model_data)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("best_val_dice", best_val_dice)
        print(f"Resuming training from epoch {start_epoch}...")

    print(f"Starting training on {device}...")
    print(f"- Model: UNet {num_stages} stages, {base_chs} base channels")
    print(
        f"- Compiled: {'Yes' if compile_model else 'No'} (mode={compile_mode if compile_model else 'N/A'})"
    )
    print(f"- Input: {input_shape}, batch_size={batch_size}")
    print(f"- Slice mode: {slice_mode}")
    print(f"- Training: {nb_epochs} epochs, lr={learning_rate}")
    print(f"- Scheduler: {scheduler.__class__.__name__}")
    print(f"- Optimizer: {optimizer}, weight_decay={weight_decay}")
    print(
        f"- Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )
    print(f"- Target device: {device}")

    for epoch in range(start_epoch, nb_epochs):
        model.train()
        t0 = time.time()

        for batch_idx, (batch, gt) in enumerate(train_loader):
            batch = batch.to(device_obj, non_blocking=True)
            gt = gt.to(device_obj, non_blocking=True)  # Index tensor [B, D, H, W]

            # Mixed Precision Forward Pass
            with autocast(device_type=device, dtype=ptdtype):
                preds = model(batch)  # List of [Res1, Res2, ...]

                loss = 0

                # GPU-side one-hotting for Dice
                # Doing this here on GPU reduce CPU-GPU transfer overhead
                # Doing dice loss without one-hot encoding is very bad depending on the number of classes
                gt_shape = (gt.shape[0], out_channels, *gt.shape[1:])
                gt_onehot = torch.zeros(gt_shape, device=device_obj, dtype=ptdtype)
                gt_onehot.scatter_(
                    1, gt.unsqueeze(1), 1
                )  # [B, D, H, W] -> [B, C, D, H, W]
                weights = [2**i for i in range(num_stages - 1)]
                weight_sum = float(sum(weights))
                weights = [w / weight_sum for w in weights]
                for i, (weight, p) in enumerate(zip(weights, reversed(preds))):
                    stride = 2**i
                    # Strided views for 2D and 3D masks
                    spatial_slices = (slice(None, None, stride),) * (gt.ndim - 1)
                    gt_idx = gt[(slice(None), *spatial_slices)]
                    gt_onehot_idx = gt_onehot[
                        (slice(None), slice(None), *spatial_slices)
                    ]

                    loss_ce = F.cross_entropy(p, gt_idx)
                    loss_dice = dice_loss(F.softmax(p, dim=1), gt_onehot_idx)
                    loss += weight * (loss_ce + loss_dice)

            # Backward Pass with Scaler
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # ty:ignore
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Log training progress
            if batch_idx % log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                lr = scheduler.get_last_lr()[0]
                lossf = loss.item()  # ty:ignore
                print(
                    f"Epoch {epoch + 1}/{nb_epochs} [{batch_idx * batch_size}/250] Loss: {lossf:.4f} | LR: {lr:.6f} | Time: {dt:.2f}s"
                )
                wandb_run.log(
                    {
                        "train/loss": lossf,
                        "lr": lr,
                        "epoch": epoch + batch_idx / len(train_loader),
                    }
                )

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
                    gt_onehot = torch.zeros(gt_shape, device=device_obj, dtype=ptdtype)
                    gt_onehot.scatter_(1, gt.unsqueeze(1), 1)

                    p_onehot = torch.zeros_like(gt_onehot)
                    p_onehot.scatter_(1, p.argmax(dim=1, keepdim=True), 1)
                    val_dice += 1 - dice_loss(
                        p_onehot, gt_onehot
                    )  # 1 - (1-dice) = dice

            dt = time.time() - t0
            mean_val_dice = val_dice / len(val_loader)
            best_val_dice = max(best_val_dice, mean_val_dice)

            print(
                f"Epoch {epoch + 1}/{nb_epochs} | EMA Val Dice: {mean_val_dice:.4f} | Time: {dt:.2f}s"
            )
            wandb_run.log(
                {
                    "val/dice": mean_val_dice,
                    "val/best_dice": best_val_dice,
                    "epoch": epoch + 1,
                }
            )

        # 5. Save Training Checkpoint
        if (epoch + 1) % save_interval == 0:
            # HACK: Remove _orig_mod. prefix from state dict keys from torch.compile()
            state_dict = model.state_dict()
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
            }
            checkpoint = {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "scaler": scaler.state_dict(),
                "best_val_dice": best_val_dice,
            }
            os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
            torch.save(
                checkpoint, os.path.join(run_dir, "checkpoints", "ckpt_latest.pt")
            )
            if mean_val_dice == best_val_dice:
                torch.save(
                    checkpoint, os.path.join(run_dir, "checkpoints", "ckpt_best.pt")
                )

    # Save final model checkpoint
    torch.save(state_dict, os.path.join(run_dir, "checkpoints", "ckpt_final.pt"))

    wandb_run.finish()


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
            compile_mode=compile_mode,
            checkpoint=checkpoint,
            wb_config=wb_config,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\nCUDA out of memory error: {e}")
            print("Try reducing batch_size or input_shape")
        else:
            raise
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise
