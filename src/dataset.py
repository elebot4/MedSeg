
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transforms import intensity_transform, lowres_transform, spatial_transform


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, file_list, slice_mode='fullres', input_shape=(32, 32, 32), augment=False):
        self.data_dir = data_dir
        self.file_list = file_list
        self.slice_mode = slice_mode
        self.input_shape = input_shape
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        
        # 1. Load data using memory mapping (no copy yet)
        img_mmap = np.load(os.path.join(self.data_dir, name, 'image.npy'), mmap_mode='r')
        mask_mmap = np.load(os.path.join(self.data_dir, name, 'label.npy'), mmap_mode='r')[None]
        with open(os.path.join(self.data_dir, name, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        # 2. Apply slicing based on mode. Ensure meaningful patches by eagerly rejecting empty slices. 
        
        spatial_shape = img_mmap.shape[1:]  # Skip channel dimension
        
        axial_modes = ['axi', 'cor', 'sag']
        axis_by_mode = {'sag': 3, 'cor': 2, 'axi': 1}
        slicer = [slice(None)] * 4 # [C, H, W, D]


        if self.slice_mode in axial_modes:
            axis = axis_by_mode[self.slice_mode]
            foreground_indexes = metadata['valid_tumor_indices'][self.slice_mode]
            if np.random.rand() < 0.66 and foreground_indexes:
                index = int(np.random.choice(foreground_indexes))
            else:
                index = int(np.random.randint(img_mmap.shape[axis]))
            slicer[axis] = index
        
        
        elif self.slice_mode == "fullres":
            
            for mode, size, dim in zip(axial_modes, self.input_shape, spatial_shape):
                axis = axis_by_mode[mode]
                low = size // 2
                high = dim - (size - size // 2) + 1
                
                foreground_indexes = metadata['valid_tumor_indices'][mode]
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
        #np.save(f"{name}_sliced_array.npy", img.numpy())
        #
        #print(f"Image values range: [{img.min():.3f}, {img.max():.3f}]")
        #print(f"Image MMAP values range: [{img_mmap.min():.3f}, {img_mmap.max():.3f}]")
        #
        #np.save(f"{name}_sliced_mask.npy", mask.numpy())
        #print(f"Mask values range: [{mask.min():.3f}, {mask.max():.3f}]")
        #print(f"Mask MMAP values range: [{mask_mmap.min():.3f}, {mask_mmap.max():.3f}]")

        
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
                img = F.pad(img, pads, mode='constant', value=0)
                mask = F.pad(mask, pads, mode='constant', value=0)
            
            
            
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
                    img = torch.flip(img, dims=[i + 1])    # i+1 to skip channel dim
                    mask = torch.flip(mask, dims=[i])      # no channel dim in mask

        # remove singleton channel dim & return 
        mask = mask[0].long()
        

        return img, mask


def get_dataloaders(
    data_dir,
    batch_size,
    slice_mode='fullres',
    input_shape=(32, 32, 32),
    train_split=0.8,
    num_workers=0,
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

    required_files = ('image.npy', 'label.npy', 'metadata.json')
    case_names = []

    for name in sorted(os.listdir(data_dir)):
        case_dir = os.path.join(data_dir, name)
        if not os.path.isdir(case_dir):
            continue

        missing = [
            filename for filename in required_files
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

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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


if __name__ == "__main__":
    # Basic testing setup - test on 2-3 samples maximum
    import sys
    print("Testing SegmentationDataset...")
    
    # Set global config for testing
    data_dir = "data/processed/Task01_BrainTumour/imagesTr"
    batch_size = 2
    
    # Test 1: Check if data directory exists
    case_dir = os.path.join(data_dir)    
    if not os.path.exists(case_dir):
        print(f"ERROR: Case directory not found: {case_dir}")
        sys.exit(1)
    
    # Get all cases (limit to first 3 for fast testing)
    all_files = [f for f in os.listdir(case_dir) if f.endswith('.npy')][:3]
    if len(all_files) < 2:
        print(f"ERROR: Need at least 2 .npy files for testing, found {len(all_files)}")
        sys.exit(1)
    
    print(f"Testing with {len(all_files)} files: {all_files}")
    # Test 2: Dataset initialization and basic loading
    dataset = SegmentationDataset(data_dir, all_files, input_shape=(64, 64, 64), augment=False)
    assert len(dataset) == len(all_files), f"Dataset length mismatch: {len(dataset)} vs {len(all_files)}"
    
    # Test 3: Load first sample and check shapes
    img, mask = dataset[0]
    print(f"Sample 0 - Image shape: {img.shape}, Mask shape: {mask.shape}")
    assert img.ndim >= 3, f"Expected at least 3D tensor, got {img.ndim}D"
    assert img.shape[1:] == mask.shape[1:], f"Spatial shape mismatch: {img.shape[1:]} vs {mask.shape[1:]}"
    
    # Test 4: Verify shape matching - should match input_shape exactly
    spatial_dims = img.shape[1:]
    expected_shape = (64, 64, 64)
    assert spatial_dims == expected_shape, f"Shape mismatch: got {spatial_dims}, expected {expected_shape}"
    print(f"✓ Shape correct: matches target input_shape {expected_shape}")
    
    # Test 5: Check data types
    assert img.dtype == torch.float32, f"Expected float32 image, got {img.dtype}"
    assert mask.dtype == torch.int64, f"Expected int64 mask, got {mask.dtype}"
    print(f"✓ Data types correct: image={img.dtype}, mask={mask.dtype}")
    
    # Test 6: Test augmentation toggle  
    dataset_aug = SegmentationDataset(data_dir, all_files[:1], input_shape=(64, 64, 64), augment=True)  # Single file for determinism
    img_aug, mask_aug = dataset_aug[0]
    print(f"Augmented sample - Image shape: {img_aug.shape}, Mask shape: {mask_aug.shape}")
    assert img_aug.shape[1:] == mask_aug.shape[1:], "Augmented shapes should match"
    
    # Test 7: Dataloader creation with small batch
    try:
        train_loader, val_loader = get_dataloaders(data_dir, batch_size, input_shape=(64, 64, 64), train_split=0.7, num_workers=0)  # No multiprocessing for testing
        print(f"✓ Dataloaders created: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
        
        # Test single batch
        batch_img, batch_mask = next(iter(train_loader))
        print(f"✓ Batch loading: {batch_img.shape}, {batch_mask.shape}")
        assert batch_img.shape[0] <= batch_size, f"Batch size exceeded: {batch_img.shape[0]} > {batch_size}"
        
    except Exception as e:
        print(f"ERROR in dataloader creation: {e}")
        sys.exit(1)
    
    print("✓ All tests passed. Dataset is working correctly.")
    print(f"Image value range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask).tolist()}")