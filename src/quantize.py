"""
Post-training quantization script (PTQ) for trained UNet segmentation models.
Readability-first, end-to-end pipeline in the same style as base_train.py.
"""

import copy
import os
import time

import torch
import torch.nn as nn
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

from dataset import get_dataloaders
from model import UNet

# -----------------------------------------------------------------------------
# Default config values
# I/O settings
checkpoint = "outputs/2d_axi/checkpoints/ckpt_best.pt"
save_name = "ckpt_quantized.pt"
out_dir = None  # If None, save next to source checkpoint
num_workers = 0

# Quantization settings
backend = "qnnpack"  # qnnpack (CPU-constrained) or x86 (server-class)
nb_steps = 8

# -----------------------------------------------------------------------------
# Load config overrides
_config_path = os.path.join(os.path.dirname(__file__), "config.py")
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open(_config_path).read())


def prepare_ptq(model: nn.Module, backend: str = "qnnpack") -> nn.Module:
    """Prepare a model for torchao PTQ."""
    if backend not in {"qnnpack", "x86"}:
        raise ValueError(f"Unsupported backend: {backend}. Use 'qnnpack' or 'x86'")

    return copy.deepcopy(model).cpu().eval()


def calibrate_ptq(model: nn.Module, calibration_loader, nb_steps: int = 32) -> int:
    """Run bounded calibration/evaluation passes before conversion."""
    if nb_steps <= 0:
        raise ValueError(f"nb_steps must be > 0, got {nb_steps}")

    model.eval()
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_loader):
            if batch_idx >= nb_steps:
                break

            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            _ = model(x.cpu())
            processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError("Calibration loader produced 0 batches")

    return processed_batches


def finalize_ptq(prepared_model: nn.Module) -> nn.Module:
    """Convert model weights/activations to torchao int8 dynamic quantization."""

    def is_quantizable_module(module: nn.Module, _fqn: str) -> bool:
        return isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear))

    quantize_(
        prepared_model,
        Int8DynamicActivationInt8WeightConfig(),
        filter_fn=is_quantizable_module,
    )
    return prepared_model


def run_ptq(
    checkpoint,
    save_name,
    out_dir,
    num_workers,
    backend,
    nb_steps,
):
    """End-to-end PTQ pipeline driven from checkpoint config."""
    t0 = time.time()

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict with at least 'model' and 'config'")
    if "model" not in ckpt:
        raise ValueError("Checkpoint missing 'model' state_dict")
    if "config" not in ckpt:
        raise ValueError("Checkpoint missing 'config' required for PTQ setup")

    config = ckpt["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint 'config' must be a dictionary")

    data_dir = config["data_dir"]
    slice_mode = config["slice_mode"]
    input_shape = tuple(config["input_shape"])
    batch_size = int(config.get("batch_size", 1))
    sample_per_epoch = max(int(nb_steps) * max(batch_size, 1), 32)

    print("Starting PTQ pipeline...")
    print(f"- Checkpoint: {checkpoint}")
    print(f"- Backend: {backend}")
    print(f"- Calibration steps: {nb_steps}")
    print(f"- Calibration batch size: {batch_size}")
    print(f"- Data dir: {data_dir}")
    print(f"- Slice mode: {slice_mode}")
    print(f"- Input shape: {input_shape}")
    print("- Quantization API: torchao.quantization.quantize_ (dynamic int8)")

    train_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        slice_mode=slice_mode,
        input_shape=input_shape,
        num_workers=num_workers,
        sample_per_epoch=sample_per_epoch,
    )

    model = (
        UNet(
            input_shape=tuple(config["input_shape"]),
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            num_stages=config["num_stages"],
            base_chs=config["base_chs"],
            norm_type=config["norm_type"],
            act_type=config["act_type"],
            dropout=config["dropout"],
            norm_groups=config["norm_groups"],
            deep_supervision=config["deep_supervision"],
        )
        .cpu()
        .eval()
    )
    model.load_state_dict(ckpt["model"])

    prepared = prepare_ptq(model, backend=backend)
    print(f"Prepared PTQ model at +{time.time() - t0:.2f}s")

    used_batches = calibrate_ptq(
        model=prepared,
        calibration_loader=train_loader,
        nb_steps=nb_steps,
    )
    print(f"Calibrated with {used_batches} batches at +{time.time() - t0:.2f}s")

    quantized = finalize_ptq(prepared)
    print(f"Converted quantized model at +{time.time() - t0:.2f}s")

    checkpoint_dir = os.path.dirname(checkpoint)
    if out_dir is not None:
        checkpoint_dir = os.path.join(out_dir, slice_mode, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    output_path = os.path.join(checkpoint_dir, save_name)

    artifact = {
        "model": quantized.state_dict(),
        "metadata": {
            "source_checkpoint": checkpoint,
            "backend": backend,
            "nb_steps": used_batches,
            "slice_mode": slice_mode,
            "input_shape": input_shape,
            "batch_size": batch_size,
            "created_unix": int(time.time()),
        },
        "config": config,
    }
    torch.save(artifact, output_path)

    print(f"Saved quantized checkpoint to: {output_path}")
    print(f"PTQ pipeline completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    try:
        config = {k: v for k, v in globals().items() if k in config_keys}
        run_ptq(
            checkpoint=checkpoint,
            save_name=save_name,
            out_dir=out_dir,
            num_workers=num_workers,
            backend=backend,
            nb_steps=nb_steps,
        )
    except KeyboardInterrupt:
        print("\nQuantization interrupted by user")
    except Exception as e:
        print(f"\nQuantization failed: {e}")
        raise
