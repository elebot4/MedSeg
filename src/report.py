"""
Segmentation report generation for medical images. /!\ this file does not include LLM-based reporting /!\
Creates human-readable summaries of segmentation results.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval import dice_score, iou_score, sensitivity_specificity
from model import UNet


class SegmentationAnalysis:
    """Container for segmentation analysis results."""
    def __init__(self):
        self.volume_mm3 = 0.0
        self.confidence_score = 0.0
        self.dice_score = 0.0
        self.iou_score = 0.0
        self.centroid = (0.0, 0.0, 0.0)
        self.bounding_box = (0, 0, 0, 0, 0, 0)  # x1,y1,z1,x2,y2,z2
        self.shape_metrics = {}


def analyze_segmentation_mask(
    pred_mask: np.ndarray,
    target_mask=None,
    voxel_spacing=(1.0, 1.0, 1.0),
    class_name: str = "unknown"
):
    """
    Analyze a segmentation mask to extract quantitative metrics.
    
    Args:
        pred_mask: Predicted segmentation mask (binary)
        target_mask: Ground truth mask (for accuracy metrics)
        voxel_spacing: Physical spacing between voxels (mm)
        class_name: Name of the segmented class
        
    Returns:
        Analysis results
    """
    if pred_mask.sum() == 0:
        return SegmentationAnalysis(
            volume_mm3=0.0,
            confidence_score=0.0,
            dice_score=0.0,
            iou_score=0.0,
            centroid=(0.0, 0.0, 0.0),
            bounding_box=(0, 0, 0, 0, 0, 0),
            shape_metrics={}
        )
    
    # Calculate volume in mm³
    voxel_volume = np.prod(voxel_spacing)
    volume_mm3 = float(pred_mask.sum() * voxel_volume)
    
    # Calculate centroid
    coords = np.where(pred_mask > 0)
    centroid = tuple(float(np.mean(coord)) for coord in coords)
    
    # Calculate bounding box
    min_coords = tuple(int(np.min(coord)) for coord in coords)
    max_coords = tuple(int(np.max(coord)) for coord in coords)
    bounding_box = min_coords + max_coords
    
    # Shape metrics
    shape_metrics = calculate_shape_metrics(pred_mask, voxel_spacing)
    
    # Accuracy metrics (if ground truth available)
    dice_val = 0.0
    iou_val = 0.0
    if target_mask is not None:
        pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target_mask).unsqueeze(0).unsqueeze(0).float()
        dice_val = dice_score(pred_tensor, target_tensor).item()
        iou_val = iou_score(pred_tensor, target_tensor).item()
    
    # Confidence score (placeholder - could use model uncertainty)
    confidence_score = min(dice_val * 1.2, 1.0) if target_mask is not None else 0.8
    
    return SegmentationAnalysis(
        volume_mm3=volume_mm3,
        confidence_score=confidence_score,
        dice_score=dice_val,
        iou_score=iou_val,
        centroid=centroid,
        bounding_box=bounding_box,
        shape_metrics=shape_metrics
    )


def calculate_shape_metrics(mask: np.ndarray, voxel_spacing):
    """
    Calculate shape-based metrics for a segmentation mask.
    
    Args:
        mask: Binary segmentation mask
        voxel_spacing: Physical voxel spacing
        
    Returns:
        Dictionary of shape metrics
    """
    from scipy import ndimage
    
    metrics = {}
    
    # Surface area (approximated)
    # Use binary gradient to find edges
    grad_x = np.abs(np.diff(mask.astype(float), axis=0, prepend=0))
    grad_y = np.abs(np.diff(mask.astype(float), axis=1, prepend=0))
    grad_z = np.abs(np.diff(mask.astype(float), axis=2, prepend=0))
    
    surface_voxels = (grad_x + grad_y + grad_z) > 0
    surface_area = surface_voxels.sum() * np.mean(voxel_spacing)**2  # Rough approximation
    metrics['surface_area_mm2'] = float(surface_area)
    
    # Sphericity (how sphere-like the object is)
    volume = mask.sum() * np.prod(voxel_spacing)
    if volume > 0:
        sphere_surface_area = (36 * np.pi * volume**2)**(1/3)
        sphericity = sphere_surface_area / surface_area if surface_area > 0 else 0
        metrics['sphericity'] = min(float(sphericity), 1.0)
    else:
        metrics['sphericity'] = 0.0
    
    # Extent metrics
    coords = np.where(mask > 0)
    if len(coords[0]) > 0:
        ranges = [np.ptp(coord) * spacing for coord, spacing in zip(coords, voxel_spacing)]
        metrics['extent_x_mm'] = float(ranges[0])
        metrics['extent_y_mm'] = float(ranges[1])
        metrics['extent_z_mm'] = float(ranges[2])
        metrics['max_diameter_mm'] = float(max(ranges))
    
    return metrics


def generate_finding_description(
    analysis: SegmentationAnalysis,
    class_name: str,
    severity_threshold=None
):
    """
    Generate a natural language description of a segmentation finding.
    
    Args:
        analysis: Quantitative analysis results
        class_name: Name of the segmented anatomical structure
        severity_threshold: Volume thresholds for severity assessment
        
    Returns:
        Natural language description
    """
    if analysis.volume_mm3 == 0:
        return f"No {class_name} detected."
    
    # Default thresholds (can be customized per anatomy)
    if severity_threshold is None:
        severity_threshold = {
            'small': 1000,    # mm³
            'medium': 5000,   # mm³
            'large': 15000    # mm³
        }
    
    # Size assessment
    volume_ml = analysis.volume_mm3 / 1000  # Convert to mL
    if analysis.volume_mm3 < severity_threshold['small']:
        size_desc = "small"
    elif analysis.volume_mm3 < severity_threshold['medium']:
        size_desc = "moderate-sized"
    elif analysis.volume_mm3 < severity_threshold['large']:
        size_desc = "large"
    else:
        size_desc = "very large"
    
    # Shape assessment
    sphericity = analysis.shape_metrics.get('sphericity', 0.5)
    if sphericity > 0.8:
        shape_desc = "rounded"
    elif sphericity > 0.6:
        shape_desc = "oval"
    else:
        shape_desc = "irregular"
    
    # Confidence assessment
    confidence_desc = ""
    if hasattr(analysis, 'confidence_score') and analysis.confidence_score > 0:
        if analysis.confidence_score > 0.9:
            confidence_desc = " with high confidence"
        elif analysis.confidence_score > 0.7:
            confidence_desc = " with moderate confidence"
        else:
            confidence_desc = " with low confidence"
    
    # Location description (simplified)
    x, y, z = analysis.centroid
    location_desc = f"centered approximately at coordinates ({x:.1f}, {y:.1f}, {z:.1f})"
    
    # Measurements
    max_diameter = analysis.shape_metrics.get('max_diameter_mm', 0)
    
    description = (
        f"A {size_desc} {shape_desc} {class_name} measuring approximately "
        f"{volume_ml:.1f} mL (volume) with a maximum diameter of {max_diameter:.1f} mm, "
        f"{location_desc}{confidence_desc}."
    )
    
    return description


def generate_comprehensive_report(
    predictions: torch.Tensor,
    class_names,
    voxel_spacing=(1.0, 1.0, 1.0),
    ground_truth=None,
    patient_info=None
):
    """
    Generate a comprehensive radiology-style report from segmentation results.
    
    Args:
        predictions: Model predictions [C, D, H, W] or categorical mask [D, H, W]
        class_names: Names of segmented classes
        voxel_spacing: Physical voxel spacing (mm)
        ground_truth: Ground truth masks for accuracy assessment
        patient_info: Optional patient metadata
        
    Returns:
        Formatted medical report
    """
    # Convert predictions to categorical if needed
    if predictions.ndim == 4 and predictions.shape[0] > 1:
        pred_mask = F.softmax(predictions, dim=0).argmax(dim=0).numpy()
    else:
        pred_mask = predictions.squeeze().numpy()
    
    # Convert ground truth if provided
    gt_mask = ground_truth.numpy() if ground_truth is not None else None
    
    # Analyze each class
    findings = []
    for class_idx, class_name in enumerate(class_names):
        if class_idx == 0:  # Skip background
            continue
            
        # Extract binary mask for this class
        binary_pred = (pred_mask == class_idx).astype(np.uint8)
        binary_gt = (gt_mask == class_idx).astype(np.uint8) if gt_mask is not None else None
        
        # Analyze this finding
        analysis = analyze_segmentation_mask(
            binary_pred, binary_gt, voxel_spacing, class_name
        )
        
        if analysis.volume_mm3 > 0:  # Only include if something was detected
            description = generate_finding_description(analysis, class_name)
            findings.append((class_name, description, analysis))
    
    # Generate report
    report_lines = []
    
    # Header
    report_lines.append("AUTOMATED SEGMENTATION REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Patient info
    if patient_info:
        report_lines.append("PATIENT INFORMATION:")
        for key, value in patient_info.items():
            report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        report_lines.append("")
    
    # Technical parameters
    report_lines.append("TECHNICAL PARAMETERS:")
    report_lines.append(f"  Voxel Spacing: {voxel_spacing[0]:.2f} x {voxel_spacing[1]:.2f} x {voxel_spacing[2]:.2f} mm")
    report_lines.append(f"  Image Dimensions: {pred_mask.shape}")
    report_lines.append("")
    
    # Findings
    report_lines.append("FINDINGS:")
    if not findings:
        report_lines.append("  No significant abnormalities detected.")
    else:
        for i, (class_name, description, analysis) in enumerate(findings, 1):
            report_lines.append(f"  {i}. {description}")
            
            # Add quantitative details
            if analysis.dice_score > 0:  # If ground truth comparison available
                report_lines.append(f"     Segmentation accuracy: Dice={analysis.dice_score:.3f}, IoU={analysis.iou_score:.3f}")
    
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("QUANTITATIVE SUMMARY:")
    total_volume = sum(analysis.volume_mm3 for _, _, analysis in findings)
    report_lines.append(f"  Total abnormal tissue volume: {total_volume/1000:.2f} mL")
    report_lines.append(f"  Number of distinct findings: {len(findings)}")
    
    if findings:
        largest_finding = max(findings, key=lambda x: x[2].volume_mm3)
        report_lines.append(f"  Largest finding: {largest_finding[0]} ({largest_finding[2].volume_mm3/1000:.2f} mL)")
    
    report_lines.append("")
    
    # Disclaimer
    report_lines.append("DISCLAIMER:")
    report_lines.append("This report was generated by an automated segmentation algorithm.")
    report_lines.append("All findings should be verified by a qualified radiologist.")
    report_lines.append("This analysis is for research/educational purposes only.")
    
    return "\n".join(report_lines)


def save_report(
    report_text: str,
    output_path: str,
    format: str = 'txt'
) -> None:
    """
    Save report to file.
    
    Args:
        report_text: Generated report text
        output_path: Path to save the report
        format: Output format ('txt', 'html', 'markdown')
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'txt':
        with open(output_path, 'w') as f:
            f.write(report_text)
    elif format == 'markdown':
        # Convert to markdown format
        markdown_text = report_text.replace("=" * 50, "---")
        with open(output_path, 'w') as f:
            f.write("# " + markdown_text.replace("\n", "\n\n", 1))
    elif format == 'html':
        # Simple HTML conversion
        html_lines = ["<html><head><title>Segmentation Report</title></head><body>"]
        html_lines.append("<pre style='font-family: monospace;'>")
        html_lines.append(report_text.replace("\n", "<br>\n"))
        html_lines.append("</pre></body></html>")
        with open(output_path, 'w') as f:
            f.write("\n".join(html_lines))
    
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Testing report generation with synthetic data...")
    
    # Create synthetic segmentation results
    pred_mask = np.zeros((64, 64, 64), dtype=np.uint8)
    pred_mask[20:40, 20:40, 20:40] = 1  # Tumor
    pred_mask[45:55, 10:20, 10:20] = 2  # Edema
    
    # Convert to tensor
    predictions = torch.from_numpy(pred_mask)
    
    # Generate report
    class_names = ["background", "tumor", "edema"]
    patient_info = {"patient_id": "TEST001", "scan_date": "2024-01-01"}
    
    report = generate_comprehensive_report(
        predictions=predictions,
        class_names=class_names,
        voxel_spacing=(1.0, 1.0, 1.0),
        patient_info=patient_info
    )
    
    print(report)
    
    # Save report
    save_report(report, "reports/example_report.txt")
    save_report(report, "reports/example_report.html", format='html')