"""
Evaluation and metrics for medical segmentation models.
Supports various metrics like Dice, IoU, Hausdorff distance, and inference pipelines.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

from dataset import get_dataloaders
from model import UNet


def dice_score(pred, target, smooth=1e-6):
    """
    Calculate Dice score for segmentation.
    
    Args:
        pred: Predicted probabilities [B, C, *spatial_dims]
        target: Ground truth labels [B, C, *spatial_dims] (one-hot)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice score per class [B, C]
    """
    pred = F.softmax(pred, dim=1)
    
    # Flatten spatial dimensions
    reduce_axis = tuple(range(2, pred.ndim))
    intersection = (pred * target).sum(dim=reduce_axis)
    pred_sum = pred.sum(dim=reduce_axis)
    target_sum = target.sum(dim=reduce_axis)
    
    dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) score.
    
    Args:
        pred: Predicted probabilities [B, C, *spatial_dims]
        target: Ground truth labels [B, C, *spatial_dims] (one-hot)
        smooth: Smoothing factor
        
    Returns:
        IoU score per class [B, C]
    """
    pred = F.softmax(pred, dim=1)
    
    # Flatten spatial dimensions
    reduce_axis = tuple(range(2, pred.ndim))
    intersection = (pred * target).sum(dim=reduce_axis)
    union = pred.sum(dim=reduce_axis) + target.sum(dim=reduce_axis) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate Hausdorff distance between two binary masks.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        Hausdorff distance
    """
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')
    
    # Get surface points
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float('inf')
    
    # Calculate directed Hausdorff distances
    hd1 = directed_hausdorff(pred_surface, target_surface)[0]
    hd2 = directed_hausdorff(target_surface, pred_surface)[0]
    
    return max(hd1, hd2)


def get_surface_points(mask: np.ndarray) -> np.ndarray:
    """
    Extract surface points from a binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Array of surface point coordinates
    """
    # Get binary edges using morphological operations
    eroded = ndimage.binary_erosion(mask)
    edges = mask ^ eroded
    
    # Get coordinates of edge points
    coords = np.where(edges)
    if len(coords[0]) == 0:
        return np.array([])
    
    return np.column_stack(coords)


def sensitivity_specificity(pred: torch.Tensor, target: torch.Tensor):
    """
    Calculate sensitivity (recall) and specificity.
    
    Args:
        pred: Predicted probabilities [B, C, *spatial_dims]
        target: Ground truth labels [B, C, *spatial_dims] (one-hot)
        
    Returns:
        Tuple of (sensitivity, specificity) per class [B, C]
    """
    pred_binary = (F.softmax(pred, dim=1) > 0.5).float()
    
    # Flatten spatial dimensions
    reduce_axis = tuple(range(2, pred.ndim))
    
    # True/False Positives/Negatives
    tp = ((pred_binary == 1) & (target == 1)).sum(dim=reduce_axis).float()
    tn = ((pred_binary == 0) & (target == 0)).sum(dim=reduce_axis).float()
    fp = ((pred_binary == 1) & (target == 0)).sum(dim=reduce_axis).float()
    fn = ((pred_binary == 0) & (target == 1)).sum(dim=reduce_axis).float()
    
    # Sensitivity (Recall) and Specificity
    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    
    return sensitivity, specificity


def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    window_size,
    overlap: float = 0.25
):
    """
    Perform sliding window inference for large images.
    
    Args:
        model: Segmentation model
        image: Input image [C, *spatial_dims]
        window_size: Size of sliding window
        overlap: Overlap ratio between windows (0-1)
        
    Returns:
        Segmentation prediction [C, *spatial_dims]
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Calculate step size
    step_size = tuple(int(w * (1 - overlap)) for w in window_size)
    
    # Get image dimensions
    image_shape = image.shape[1:]  # Skip channel dimension
    
    # Initialize output
    output = torch.zeros((1,) + image_shape, device=device)  # Will be resized after first forward pass
    count_map = torch.zeros(image_shape, device=device)
    
    # Generate sliding windows
    indices = []
    for dim, (img_size, win_size, step) in enumerate(zip(image_shape, window_size, step_size)):
        dim_indices = list(range(0, img_size - win_size + 1, step))
        if dim_indices[-1] + win_size < img_size:
            dim_indices.append(img_size - win_size)
        indices.append(dim_indices)
    
    # Process each window
    import itertools
    with torch.no_grad():
        for window_indices in itertools.product(*indices):
            # Extract window
            slices = tuple(slice(idx, idx + size) for idx, size in zip(window_indices, window_size))
            window = image[(slice(None),) + slices].unsqueeze(0)  # Add batch dimension
            
            # Predict
            pred = model(window)
            if isinstance(pred, list):
                pred = pred[0]  # Use highest resolution prediction
            
            pred = pred.squeeze(0)  # Remove batch dimension
            
            # Add to output
            output[(slice(None),) + slices] += pred
            count_map[slices] += 1
    
    # Average overlapping predictions
    output /= count_map.unsqueeze(0)
    
    return output


def evaluate_model(
    model: torch.nn.Module,
    data_dir: str,
    batch_size: int, 
    num_stages: int,
    out_channels: int,
    save_results: bool = True
) -> dict:
    """
    Comprehensive evaluation of a segmentation model.
    
    Args:
        model: Trained segmentation model
        data_dir: Path to processed data directory
        batch_size: Batch size for data loading
        num_stages: Number of U-Net stages
        out_channels: Number of output channels
        save_results: Whether to save detailed results
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get test data
    _, val_loader = get_dataloaders(data_dir, batch_size, num_stages)
    
    # Initialize metrics
    all_dice = []
    all_iou = []
    all_sensitivity = []
    all_specificity = []
    all_hausdorff = []
    
    print(f"Evaluating model on {len(val_loader)} samples...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            preds = model(images)
            if isinstance(preds, list):
                preds = preds[0]  # Use highest resolution prediction
            
            # Convert masks to one-hot
            masks_onehot = F.one_hot(masks, num_classes=config.out_channels)
            masks_onehot = masks_onehot.permute(0, -1, *range(1, masks_onehot.ndim-1)).float()
            
            # Calculate metrics
            dice = dice_score(preds, masks_onehot)
            iou = iou_score(preds, masks_onehot)
            sens, spec = sensitivity_specificity(preds, masks_onehot)
            
            all_dice.append(dice.cpu())
            all_iou.append(iou.cpu())
            all_sensitivity.append(sens.cpu())
            all_specificity.append(spec.cpu())
            
            # Calculate Hausdorff distance (slower, do for subset)
            if batch_idx < 10:  # Only for first 10 batches
                for b in range(images.shape[0]):
                    pred_np = F.softmax(preds[b], dim=0).argmax(dim=0).cpu().numpy()
                    target_np = masks[b].cpu().numpy()
                    
                    for c in range(1, out_channels):  # Skip background
                        pred_mask = (pred_np == c).astype(np.uint8)
                        target_mask = (target_np == c).astype(np.uint8)
                        
                        if pred_mask.sum() > 0 or target_mask.sum() > 0:
                            hd = hausdorff_distance(pred_mask, target_mask)
                            if hd != float('inf'):
                                all_hausdorff.append(hd)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(val_loader)} batches")
    
    # Aggregate results
    all_dice = torch.cat(all_dice, dim=0)
    all_iou = torch.cat(all_iou, dim=0)
    all_sensitivity = torch.cat(all_sensitivity, dim=0)
    all_specificity = torch.cat(all_specificity, dim=0)
    
    results = {
        'dice_mean': all_dice.mean().item(),
        'dice_std': all_dice.std().item(),
        'iou_mean': all_iou.mean().item(),
        'iou_std': all_iou.std().item(),
        'sensitivity_mean': all_sensitivity.mean().item(),
        'sensitivity_std': all_sensitivity.std().item(),
        'specificity_mean': all_specificity.mean().item(),
        'specificity_std': all_specificity.std().item(),
    }
    
    if all_hausdorff:
        results['hausdorff_mean'] = np.mean(all_hausdorff)
        results['hausdorff_std'] = np.std(all_hausdorff)
    
    # Per-class results
    for c in range(config.out_channels):
        results[f'dice_class_{c}'] = all_dice[:, c].mean().item()
        results[f'iou_class_{c}'] = all_iou[:, c].mean().item()
    
    if save_results:
        results_path = Path("results/evaluation_results.txt")
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=" * 50 + "\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
        
        print(f"Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    config = Config()
    
    # Load trained model
    model = UNet(config)
    checkpoint_path = "checkpoints/best_model.pth"
    
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model = model.to(config.device)
        
        # Run evaluation
        results = evaluate_model(model, config)
        
        print("\nEvaluation Results:")
        print("-" * 30)
        for key, value in results.items():
            if not key.startswith('dice_class') and not key.startswith('iou_class'):
                print(f"{key}: {value:.4f}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")