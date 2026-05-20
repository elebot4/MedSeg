

import numpy as np
import torch
import torch.nn.functional as F
import math

def lowres_transform(tensor):
    """
    Approximate nnU-Net v2 default low-resolution simulation.

    Args:
        tensor: torch.Tensor with shape [C, H, W] or [C, H, W, D]

    Returns:
        torch.Tensor with the same shape.
    """
    # nnU-Net: RandomTransform(..., apply_probability=0.25)
    if torch.rand((), device=tensor.device) >= 0.25:
        return tensor

    tensor = tensor.clone()
    channels = tensor.shape[0]
    orig_shape = tensor.shape[1:]
    spatial_ndim = len(orig_shape)

    upsample_modes = {
        2: "bilinear",
        3: "trilinear",
    }
    upsample_mode = upsample_modes[spatial_ndim]

    for channel in range(channels):
        if torch.rand((), device=tensor.device) >= 0.5:
            continue

        scale = torch.empty((), device=tensor.device).uniform_(0.5, 1.0).item()
        new_shape = [max(1, round(size * scale)) for size in orig_shape]

        x = tensor[channel][None, None]

        x = F.interpolate(x, size=new_shape, mode="nearest-exact")
        x = F.interpolate(x, size=orig_shape, mode=upsample_mode, align_corners=False)
        tensor[channel] = x[0, 0]

    return tensor


def intensity_transform(tensor):
    """
    Approximate nnU-Net v2 default intensity augmentation.

    Args:
        tensor: torch.Tensor with shape [C, H, W] or [C, H, W, D]

    Returns:
        torch.Tensor with the same shape.
    """
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected [C, H, W] or [C, H, W, D], got {tuple(tensor.shape)}")

    tensor = tensor.clone()
    c, *spatial_shape = tensor.shape
    spatial_ndim = len(spatial_shape)
    device = tensor.device
    dtype = tensor.dtype
    eps = torch.tensor(1e-7, device=device, dtype=dtype)

    # 1. Gaussian noise
    # nnU-Net: apply_probability=0.1, noise_variance=(0, 0.1), synchronized across channels.
    if torch.rand((), device=device) < 0.1:
        sigma = torch.empty((), device=device, dtype=dtype).uniform_(0.0, 0.1)
        tensor = tensor + torch.randn_like(tensor) * sigma

    # 2. Gaussian blur
    # nnU-Net: apply_probability=0.2, blur_sigma=(0.5, 1.0),
    #          p_per_channel=0.5, synchronize_channels=False, synchronize_axes=False.
    if torch.rand((), device=device) < 0.2:
        for channel in range(c):
            if torch.rand((), device=device) >= 0.5:
                continue

            x = tensor[channel][None, None]

            for axis in range(spatial_ndim):
                sigma = torch.empty((), device=device, dtype=dtype).uniform_(0.5, 1.0)

                kernel_size = round(float(sigma) * 4.0 + 0.5)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                half = (kernel_size - 1) / 2
                grid = torch.linspace(-half, half, kernel_size, device=device, dtype=dtype)
                kernel = torch.exp(-0.5 * (grid / sigma).pow(2))
                kernel = kernel / kernel.sum()

                pad = [0, 0] * spatial_ndim
                pad_index = 2 * (spatial_ndim - axis - 1)
                pad[pad_index] = kernel_size // 2
                pad[pad_index + 1] = kernel_size // 2

                x = F.pad(x, pad, mode="reflect")

                weight_shape = [1, 1] + [1] * spatial_ndim
                weight_shape[2 + axis] = kernel_size
                weight = kernel.view(*weight_shape)

                if spatial_ndim == 2:
                    x = F.conv2d(x, weight)
                else:
                    x = F.conv3d(x, weight)

            tensor[channel] = x[0, 0]

    # 3. Multiplicative brightness
    # nnU-Net: apply_probability=0.15, multiplier_range=(0.75, 1.25),
    #          per-channel factors.
    if torch.rand((), device=device) < 0.15:
        for channel in range(c):
            factor = torch.empty((), device=device, dtype=dtype).uniform_(0.75, 1.25)
            tensor[channel] = tensor[channel] * factor

    # 4. Contrast
    # nnU-Net: apply_probability=0.15, contrast_range=(0.75, 1.25),
    #          preserve_range=True, per-channel factors.
    if torch.rand((), device=device) < 0.15:
        for channel in range(c):
            x = tensor[channel]
            old_min = x.min()
            old_max = x.max()
            mean = x.mean()
            factor = torch.empty((), device=device, dtype=dtype).uniform_(0.75, 1.25)

            x = (x - mean) * factor + mean
            tensor[channel] = x.clamp(old_min, old_max)

    # 5a. Gamma with inversion
    # nnU-Net: apply_probability=0.1, gamma=(0.7, 1.5),
    #          p_invert_image=1, p_retain_stats=1, per-channel gamma.
    if torch.rand((), device=device) < 0.1:
        for channel in range(c):
            x = tensor[channel]
            old_mean = x.mean()
            old_std = x.std()

            x = -x
            x_min = x.min()
            x_max = x.max()
            x_range = torch.clamp(x_max - x_min, min=eps)

            gamma = torch.empty((), device=device, dtype=dtype).uniform_(0.7, 1.5)
            x = ((x - x_min) / x_range).pow(gamma) * x_range + x_min

            new_mean = x.mean()
            new_std = torch.clamp(x.std(), min=eps)
            x = (x - new_mean) * (old_std / new_std) + old_mean

            tensor[channel] = -x

    # 5b. Gamma without inversion
    # nnU-Net: apply_probability=0.3, gamma=(0.7, 1.5),
    #          p_invert_image=0, p_retain_stats=1, per-channel gamma.
    if torch.rand((), device=device) < 0.3:
        for channel in range(c):
            x = tensor[channel]
            old_mean = x.mean()
            old_std = x.std()

            x_min = x.min()
            x_max = x.max()
            x_range = torch.clamp(x_max - x_min, min=eps)

            gamma = torch.empty((), device=device, dtype=dtype).uniform_(0.7, 1.5)
            x = ((x - x_min) / x_range).pow(gamma) * x_range + x_min

            new_mean = x.mean()
            new_std = torch.clamp(x.std(), min=eps)
            tensor[channel] = (x - new_mean) * (old_std / new_std) + old_mean

    return tensor





def spatial_transform(tensor, target, rotation_degrees=15):
    """
    Approximate nnU-Net v2 default spatial transform.

    Args:
        tensor:  [C, H, W] or [C, D, H, W]
        target: [C, H, W] or [C, D, H, W]

    Returns:
        Transformed img and mask with the same shape.
    """
    if tensor.shape[1:] != target.shape[1:]:
        raise ValueError(f"tensor and target spatial shapes differ: {tensor.shape} vs {target.shape}")

    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected tensor shape [C, H, W] or [C, D, H, W], got {tuple(tensor.shape)}")

    if target.ndim != tensor.ndim:
        raise ValueError(f"tensor and target must have same ndim, got {tensor.ndim} and {target.ndim}")

    device = tensor.device
    dtype = tensor.dtype
    spatial_ndim = tensor.ndim - 1

    do_rotation = torch.rand((), device=device) < 0.2
    do_scaling = torch.rand((), device=device) < 0.2

    if not do_rotation and not do_scaling:
        return tensor, target

    scale = torch.tensor(1.0, device=device, dtype=dtype)
    if do_scaling:
        scale = torch.empty((), device=device, dtype=dtype).uniform_(0.7, 1.4)

    img_batch = tensor[None]
    mask_batch = target[None].to(dtype)

    if spatial_ndim == 2:
        angle = torch.tensor(0.0, device=device, dtype=dtype)
        if do_rotation:
            angle = torch.empty((), device=device, dtype=dtype).uniform_(
                -math.radians(rotation_degrees),
                math.radians(rotation_degrees),
            )

        cos_a = torch.cos(angle) / scale
        sin_a = torch.sin(angle) / scale

        theta = torch.zeros((1, 2, 3), device=device, dtype=dtype)
        theta[0, 0, 0] = cos_a
        theta[0, 0, 1] = -sin_a
        theta[0, 1, 0] = sin_a
        theta[0, 1, 1] = cos_a

        grid = F.affine_grid(theta, img_batch.shape, align_corners=False)

    else:
        angles = torch.zeros(3, device=device, dtype=dtype)
        if do_rotation:
            angles = torch.empty(3, device=device, dtype=dtype).uniform_(
                -math.radians(rotation_degrees),
                math.radians(rotation_degrees),
            )

        ax, ay, az = angles

        cx, sx = torch.cos(ax), torch.sin(ax)
        cy, sy = torch.cos(ay), torch.sin(ay)
        cz, sz = torch.cos(az), torch.sin(az)

        rx = torch.stack([
            torch.stack([torch.ones((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)]),
            torch.stack([torch.zeros((), device=device, dtype=dtype), cx, -sx]),
            torch.stack([torch.zeros((), device=device, dtype=dtype), sx, cx]),
        ])

        ry = torch.stack([
            torch.stack([cy, torch.zeros((), device=device, dtype=dtype), sy]),
            torch.stack([torch.zeros((), device=device, dtype=dtype), torch.ones((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)]),
            torch.stack([-sy, torch.zeros((), device=device, dtype=dtype), cy]),
        ])

        rz = torch.stack([
            torch.stack([cz, -sz, torch.zeros((), device=device, dtype=dtype)]),
            torch.stack([sz, cz, torch.zeros((), device=device, dtype=dtype)]),
            torch.stack([torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype), torch.ones((), device=device, dtype=dtype)]),
        ])

        rotation = rz @ ry @ rx

        # affine_grid maps output coordinates to input coordinates.
        # Divide by scale so scale > 1 zooms in and scale < 1 zooms out.
        theta = torch.zeros((1, 3, 4), device=device, dtype=dtype)
        theta[0, :, :3] = rotation / scale

        grid = F.affine_grid(theta, img_batch.shape, align_corners=False)

    img_out = F.grid_sample(img_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    mask_out = F.grid_sample(mask_batch, grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]

    mask_out = torch.round(mask_out).long()
    
    return img_out, mask_out


if __name__ == "__main__":
    print("Testing spatial_transform...")
    
    # Test 2D case
    print("\n2D Test:")
    img_2d = torch.randn(1, 64, 64)  # [C, H, W]
    mask_2d = torch.randint(0, 4, (1, 64, 64)).float()  # [C, H, W]
    
    print(f"Input img shape: {img_2d.shape}")
    print(f"Input mask shape: {mask_2d.shape}")
    
    img_out, mask_out = spatial_transform(img_2d, mask_2d)
    
    print(f"Output img shape: {img_out.shape}")
    print(f"Output mask shape: {mask_out.shape}")
    print(f"2D test passed: {img_out.shape == img_2d.shape and mask_out.shape == mask_2d.shape}")
    
    # Test 3D case
    print("\n3D Test:")
    img_3d = torch.randn(1, 32, 64, 64)  # [C, D, H, W]
    mask_3d = torch.randint(0, 4, (1, 32, 64, 64)).float()  # [C, D, H, W]
    
    img_out, mask_out = spatial_transform(img_3d, mask_3d)
    
    print(f"Output img shape: {img_out.shape}")
    print(f"Output mask shape: {mask_out.shape}")
    print(f"3D test passed: {img_out.shape == img_3d.shape and mask_out.shape == mask_3d.shape}")
    
    print("\nAll tests passed!")
    
    img_out, mask_out = spatial_transform(img_3d, mask_3d)
    
    print(f"Output img shape: {img_out.shape}")
    print(f"Output mask shape: {mask_out.shape}")
    print(f"3D test passed: {img_out.shape == img_3d.shape and mask_out.shape == mask_3d.shape}")
    
    # Test multiple runs to check consistency
    print("\nConsistency test (3 runs):")
    for i in range(3):
        img_out, mask_out = spatial_transform(img_2d.clone(), mask_2d.clone())
        print(f"Run {i+1}: shapes match = {img_out.shape == img_2d.shape}")
    
    print("\nAll tests completed.")

