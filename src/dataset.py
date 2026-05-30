import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from transforms import intensity_transform, lowres_transform, spatial_transform


class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        file_list: list[str],
        slice_mode: str = "fullres",
        input_shape: tuple = (32, 32, 32),
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.file_list = file_list
        self.slice_mode = slice_mode
        self.input_shape = input_shape
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: int):
        name = self.file_list[index]

        # 1. Load data using memory mapping (no copy yet)
        img_mmap = np.load(
            os.path.join(self.data_dir, name, "image.npy"), mmap_mode="r"
        )
        mask_mmap = np.load(
            os.path.join(self.data_dir, name, "labels.npy"), mmap_mode="r"
        )[None]
        with open(os.path.join(self.data_dir, name, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # 2. Apply slicing based on mode. Ensure meaningful patches by eagerly rejecting empty slices.

        spatial_shape = img_mmap.shape[1:]  # Skip channel dimension

        axial_modes = ["axi", "cor", "sag"]
        axis_by_mode = {"sag": 3, "cor": 2, "axi": 1}
        slicer = [slice(None)] * 4  # [C, H, W, D]

        if self.slice_mode in axial_modes:
            axis = axis_by_mode[self.slice_mode]
            foreground_indexes = metadata["valid_tumor_indices"][self.slice_mode]
            if np.random.rand() < 0.66 and foreground_indexes:
                index = int(np.random.choice(foreground_indexes))
            else:
                index = int(np.random.randint(img_mmap.shape[axis]))
            slicer[axis] = index  # ty:ignore

        elif self.slice_mode == "fullres":
            for mode, size, dim in zip(axial_modes, self.input_shape, spatial_shape):
                axis = axis_by_mode[mode]
                low = size // 2
                high = dim - (size - size // 2) + 1

                foreground_indexes = metadata["valid_tumor_indices"][mode]
                if np.random.rand() < 0.66 and foreground_indexes:
                    center = int(np.random.choice(foreground_indexes))
                    center = int(np.clip(center, low, high - 1))
                else:
                    center = int(np.random.randint(low, high))

                start = center - size // 2
                stop = start + size
                slicer[axis] = slice(start, stop)

        # 3. Extract the slice/patch (this creates a copy of the sliced areas)
        img = torch.from_numpy(img_mmap[tuple(slicer)].copy())
        mask = torch.from_numpy(mask_mmap[tuple(slicer)].copy())

        # np.save(f"{name}_sliced_array.npy", img.numpy())
        #
        # print(f"Image values range: [{img.min():.3f}, {img.max():.3f}]")
        # print(f"Image MMAP values range: [{img_mmap.min():.3f}, {img_mmap.max():.3f}]")
        #
        # np.save(f"{name}_sliced_mask.npy", mask.numpy())
        # print(f"Mask values range: [{mask.min():.3f}, {mask.max():.3f}]")
        # print(f"Mask MMAP values range: [{mask_mmap.min():.3f}, {mask_mmap.max():.3f}]")

        # 5. Resize to exact target shape via padding/cropping
        # This ensures consistent input size for the model
        current_shape = img.shape[1:]  # Skip channel dimension
        target_shape = self.input_shape
        if list(current_shape) != list(target_shape):
            # Calculate padding/cropping for each dimension
            pads = []
            crops = []
            for curr, tgt in zip(reversed(current_shape), reversed(target_shape)):
                if curr < tgt:
                    # Need padding
                    pad_total = tgt - curr
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    pads.extend([pad_before, pad_after])
                    crops.append(slice(None))
                elif curr > tgt:
                    # Need cropping
                    pads.extend([0, 0])
                    crop_total = curr - tgt
                    crop_start = crop_total // 2
                    crop_end = crop_start + tgt
                    crops.append(slice(crop_start, crop_end))
                else:
                    # Same size
                    pads.extend([0, 0])
                    crops.append(slice(None))

            # Apply padding first (if any non-zero pads)
            if any(p > 0 for p in pads):
                img = F.pad(img, pads, mode="constant", value=0)
                mask = F.pad(mask, pads, mode="constant", value=0)

            # Apply cropping (reverse order since we reversed above)
            crops = crops[::-1]
            if any(c != slice(None) for c in crops):
                crop_slice = tuple([slice(None)] + crops)  # Keep channel dim
                img = img[crop_slice]
                mask = mask[crop_slice]

        # 6. Apply Augmentations
        if self.augment:
            img, mask = spatial_transform(img, mask)
            img = intensity_transform(img)
            img = lowres_transform(img)

            # Random flips along spatial dimensions
            spatial_dims = img.ndim - 1  # exclude channel dimension
            for i in range(spatial_dims):
                if torch.rand(1) < 0.5:
                    img = torch.flip(img, dims=[i + 1])  # i+1 to skip channel dim
                    mask = torch.flip(
                        mask, dims=[i + 1]
                    )  # i+1 to skip channel dim in mask

        # remove singleton channel dim & return
        mask = mask[0].long()

        # remap -1 to 0 in mask (if present) to ensure valid class labels
        mask[mask == -1] = 0

        return img, mask


def get_dataloaders(
    data_dir,
    batch_size,
    slice_mode="fullres",
    input_shape=(32, 32, 32),
    train_split=0.8,
    num_workers=0,
    sample_per_epoch=250,
    seed=42,
):
    """
    Create train/validation dataloaders.

    Expected layout:
        data_dir/
          case_name/
            image.npy
            label.npy
            metadata.json
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be between 0 and 1, got {train_split}")

    if slice_mode is 'fullres' and len(input_shape) != 3:
        raise ValueError(f"input_shape must be a 3-tuple for fullres mode, got {input_shape}")
    elif slice_mode in ['axi', 'cor', 'sag'] and len(input_shape) != 2:
        raise ValueError(f"input_shape must be 2-tuple for axial slicing modes, got {input_shape}")

    required_files = ("image.npy", "labels.npy", "metadata.json")
    case_names = []

    for name in sorted(os.listdir(data_dir)):
        case_dir = os.path.join(data_dir, name)
        if not os.path.isdir(case_dir):
            continue

        missing = [
            filename
            for filename in required_files
            if not os.path.isfile(os.path.join(case_dir, filename))
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing files in {case_dir}: {missing}. "
                f"Required: {list(required_files)}"
            )

        case_names.append(name)

    if len(case_names) < 2:
        raise ValueError(
            f"Need at least 2 valid cases for train/val split, found {len(case_names)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(case_names)

    split_idx = int(len(case_names) * train_split)
    train_files = case_names[:split_idx]
    val_files = case_names[split_idx:]

    if not train_files or not val_files:
        raise ValueError(
            f"Empty train/val split with {len(case_names)} cases and train_split={train_split}"
        )

    # Create datasets
    train_dataset = SegmentationDataset(
        data_dir,
        train_files,
        slice_mode=slice_mode,
        input_shape=input_shape,
        augment=True,
    )
    val_dataset = SegmentationDataset(
        data_dir,
        val_files,
        slice_mode=slice_mode,
        input_shape=input_shape,
        augment=False,
    )

    pin_memory = True if torch.cuda.is_available() else False

    # create training sampler
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=sample_per_epoch
    )

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
